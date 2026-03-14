using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using UnityEngine;



namespace GaussianSplatting
{
    /// <summary>
    /// GPU pool-based chunk streamer with async I/O, pooled read buffers, and scatter-based GPU upload.
    /// - Fixed GPU pool: ComputeBuffers are allocated once at startup and divided into chunk-sized slots.
    ///   A free-list manages slots; loading pops a slot, eviction returns it.
    /// - Async I/O thread: a dedicated background thread owns file handles and reads chunk data on demand.
    ///   SH rest data is de-interleaved into per-band arrays on the I/O thread.
    /// - Pooled read buffers: byte[] arrays are rented/returned by size to reduce GC churn; the last chunk
    ///   may be smaller and uses its own size bucket.
    /// - Batched upload: completed reads are compacted into per-frame “patch” batches, then uploaded via
    ///   LockBufferForWrite and scatter compute dispatch.
    /// - Remap indirection: a uint remap buffer maps logical splat indices to physical pool positions,
    ///   updated in a compact dirty range via SetBufferData.
    /// </summary>
    public partial class GSChunkStreamer : IDisposable
    {
        // Per chunk lifecycle states for the slot pipeline
        private enum SlotState
        {
            Unloaded,   // Not loaded, no slot assigned
            Pending,    // Slot assigned, async read in flight
            Ready       // Data uploaded to GPU slot, included in remap
        }

        // Runtime bookkeeping for each chunk
        private class ChunkRuntime
        {
            public readonly ChunkInfo metadata;
            public SlotState state;
            public int slotIndex;           // Pool slot index, negative one when Unloaded
            public int remapStartIdx;       // Starting position in the compact remap buffer, -1 when unmapped
            public int lastVisibleFrame;

            public ChunkRuntime(ChunkInfo info)
            {
                metadata = info;
                state = SlotState.Unloaded;
                slotIndex = -1;
                remapStartIdx = -1;
                lastVisibleFrame = -1;
            }
        }

        // Async I/O request dispatched to the background thread
        private struct ReadRequest
        {
            public int chunkIndex;
            public int slotIndex;   //
        }

        // Result returned from the background thread with pre-split staging data
        private struct ReadResult
        {
            public int chunkIndex;
            public int slotIndex;
            public byte[] positions;
            public byte[] rotations;
            public byte[] scales;
            public byte[] sh;
            public byte[] shBand1;   // L=1 pre-split on IO thread, null when shBands < 1
            public byte[] shBand2;   // L=2 pre-split on IO thread, null when shBands < 2
            public byte[] shBand3;   // L=3 pre-split on IO thread, null when shBands < 3
            public bool ok;
            public string error;
        }

        // Asset reference (read only after construction)
        private readonly ChunkedGSAsset asset;
        private readonly ChunkRuntime[] chunks;

        // Pool sizing (all readonly after construction)
        private readonly int chunkSize;
        private readonly int maxSlots;
        private readonly int poolCapacity;  // maxSlots * chunkSize (total splat capacity)

        // Configuration
        private readonly float loadMargin;
        private readonly float unloadMargin;
        private readonly int evictionDelayFrames;


        // GPU pool buffers (allocated once)
        private ComputeBuffer positionBuffer;
        private ComputeBuffer rotationBuffer;
        private ComputeBuffer scaleBuffer;
        private ComputeBuffer shBuffer;
        private ComputeBuffer shRestBuffer0;  // Band 1 (L=1): full pool when bands >= 1, else 1-element dummy
        private ComputeBuffer shRestBuffer1;  // Band 2 (L=2): full pool when bands >= 2, else 1-element dummy
        private ComputeBuffer shRestBuffer2;  // Band 3 (L=3): full pool when bands >= 3, else 1-element dummy

        // Indirection / remap buffer
        private readonly uint[] remapCPU;
        private ComputeBuffer remapGPU;

        // Per-frame dirty tracking: remap index range
        private int dirtyRemapMin = int.MaxValue;
        private int dirtyRemapMax = -1;

        // Slot management
        private readonly Stack<int> freeSlots;
        private readonly HashSet<int> readyChunkSet;
        private readonly HashSet<int> pendingChunkSet;
        private readonly List<int> activeChunkOrder;

        // Async I/O infrastructure
        private readonly ConcurrentQueue<ReadRequest> requestQueue;
        private readonly ConcurrentQueue<ReadResult> completionQueue; //
        private Thread ioThread;
        private volatile bool ioThreadRunning;
        private readonly ManualResetEventSlim ioWakeEvent;

        // File paths resolved at construction (immutable, safe for I/O thread)
        private readonly string posFilePath;
        private readonly string rotFilePath;
        private readonly string scaleFilePath;
        private readonly string shFilePath;
        private readonly string shRestFilePath;
        private readonly int shRestStride;
        private readonly int shBands;

        // Frustum planes (scratch, reused each frame)
        private readonly Plane[] basePlanes;
        private readonly Plane[] innerPlanes;
        private readonly Plane[] outerPlanes;

        // Scratch lists (reused each frame to avoid allocations)
        private readonly List<int> innerVisibleList;
        private readonly List<int> outerVisibleList;
        private readonly List<int> toEvictList;
        private readonly List<int> toLoadList;

        // Octree for fast frustum culling (null when asset has no octree)
        private readonly ChunkOctreeNode[] octreeNodes;
        private readonly HashSet<int> outerVisibleSet;

        // Current state
        private int currentActiveSplatCount;
        private int lastCompletionDrainFrame = -1;
        private readonly HashSet<int> camerasProcessedThisFrame = new HashSet<int>();
        private bool bufferContentsDirty;

        // Statistics
        private int statLoadCount;
        private int statEvictCount;

        // Public accessors
        public int VisibleSplatCount => currentActiveSplatCount;
        public int LoadedChunkCount => readyChunkSet.Count;
        public int TotalChunkCount => chunks.Length;
        public int LoadCount => statLoadCount;
        public int EvictCount => statEvictCount;
        public int PendingReadCount { get; private set; }
        public bool HasBuffers => positionBuffer != null;
        public int PoolCapacity => poolCapacity;

        public ComputeBuffer PositionBuffer => positionBuffer;
        public ComputeBuffer RotationBuffer => rotationBuffer;
        public ComputeBuffer ScaleBuffer => scaleBuffer;
        public ComputeBuffer SHBuffer => shBuffer;
        public ComputeBuffer SHRestBuffer0 => shRestBuffer0;
        public ComputeBuffer SHRestBuffer1 => shRestBuffer1;
        public ComputeBuffer SHRestBuffer2 => shRestBuffer2;
        public ComputeBuffer RemapBuffer => remapGPU;

        /// <summary>
        /// Returns true once when buffer contents changed since last call. Resets the flag.
        /// </summary>
        public bool ConsumeBufferDirtyFlag()
        {
            bool val = bufferContentsDirty;
            bufferContentsDirty = false;
            return val;
        }

        /// <summary>
        /// Returns true when the given chunk is fully loaded and resident on the GPU.
        /// </summary>
        public bool IsChunkLoaded(int chunkIndex)
        {
            if (chunkIndex < 0 || chunkIndex >= chunks.Length) return false;
            return chunks[chunkIndex].state == SlotState.Ready;
        }


        // ------------------------------------------------------------------
        //  Construction and disposal

        /// <summary> Creates a new chunk streamer with a fixed GPU pool and async I/O thread. </summary>
        /// <param name="asset">Chunked asset to stream from.</param>
        /// <param name="maxVisibleSplats">Upper bound on visible splats. Capped to
        /// the asset's total splat count so small scenes do not over allocate.</param>
        /// <param name="loadMargin">Inner frustum expansion for load decisions (world units).</param>
        /// <param name="unloadMargin">Outer frustum expansion for eviction decisions
        /// (world units). Must exceed loadMargin for hysteresis.</param>
        /// <param name="evictionDelayFrames">Frames a chunk must be outside the outer
        /// frustum before its slot is freed.</param>
        public GSChunkStreamer(
            ChunkedGSAsset asset,
            int maxVisibleSplats = 20_000_000,
            float loadMargin = 2f,
            float unloadMargin = 5f,
            int evictionDelayFrames = 60)
        {
            this.asset = asset;
            this.chunkSize = asset.chunkSize;
            this.shRestStride = asset.SHRestStride;
            this.shBands = asset.SHBands;

            // Cap allocation to what the scene actually needs
            int effectiveMax = Mathf.Min(maxVisibleSplats, asset.totalSplatCount);
            this.maxSlots = Mathf.Max(1, Mathf.CeilToInt((float)effectiveMax / chunkSize));

            // Cap pool so the largest individual ComputeBuffer stays under 2 GB (the DX12 structured-buffer limit). With multi-bank SH rest the
            // bottleneck is the largest per-band stride (84 B for SH3 band 3).
            const long MaxBufferBytes = 2_000_000_000L;
            int largestStride = Mathf.Max(
                Mathf.Max(ChunkedGSAsset.PositionStride, ChunkedGSAsset.RotationStride),
                Mathf.Max(ChunkedGSAsset.ScaleStride, ChunkedGSAsset.SHStride));
            if (shBands >= 1) largestStride = Mathf.Max(largestStride, ChunkedGSAsset.SHBand1Stride);
            if (shBands >= 2) largestStride = Mathf.Max(largestStride, ChunkedGSAsset.SHBand2Stride);
            if (shBands >= 3) largestStride = Mathf.Max(largestStride, ChunkedGSAsset.SHBand3Stride);
            int maxSlotsForPool = (int)(MaxBufferBytes / largestStride / chunkSize);
            if (this.maxSlots > maxSlotsForPool)
            {
                long capSplats = (long)maxSlotsForPool * chunkSize;
                Debug.LogWarning(
                    $"GSChunkStreamer: capping pool from {this.maxSlots} to {maxSlotsForPool} slots " +
                    $"({capSplats} splats) to keep GPU buffers under 2 GB. " +
                    $"Largest stride: {largestStride} B/splat (SH bands: {shBands}). " +
                    $"Scene has {asset.totalSplatCount} splats total; streaming will page the rest.");
                this.maxSlots = maxSlotsForPool;
            }
            this.poolCapacity = this.maxSlots * chunkSize;

            this.loadMargin = loadMargin;
            this.unloadMargin = Mathf.Max(unloadMargin, loadMargin + 0.1f);
            this.evictionDelayFrames = evictionDelayFrames;

            // Build runtime chunk info
            ChunkInfo[] chunkMeta = asset.Chunks;
            chunks = new ChunkRuntime[chunkMeta.Length];
            for (int i = 0; i < chunkMeta.Length; i++)
                chunks[i] = new ChunkRuntime(chunkMeta[i]);

            // Slot free list (push in reverse so slot 0 is popped first)
            freeSlots = new Stack<int>(maxSlots);
            for (int i = maxSlots - 1; i >= 0; i--)
                freeSlots.Push(i);

            readyChunkSet = new HashSet<int>();
            pendingChunkSet = new HashSet<int>();
            activeChunkOrder = new List<int>(maxSlots);

            // Remap array: sized at pool capacity so it can hold the maximum
            // number of active splats when all slots are occupied
            remapCPU = new uint[poolCapacity];

            // Compute per-chunk byte budget for upload throttling
            bytesPerChunk = chunkSize * (ChunkedGSAsset.PositionStride + ChunkedGSAsset.RotationStride +
                                         ChunkedGSAsset.ScaleStride + ChunkedGSAsset.SHStride);
            if (shBands >= 1) bytesPerChunk += chunkSize * ChunkedGSAsset.SHBand1Stride;
            if (shBands >= 2) bytesPerChunk += chunkSize * ChunkedGSAsset.SHBand2Stride;
            if (shBands >= 3) bytesPerChunk += chunkSize * ChunkedGSAsset.SHBand3Stride;
            maxPatchesPerFrame = Mathf.Max(1, MaxUploadBytesPerFrame / bytesPerChunk);

            // Patch CPU staging (compact: sized to maxPatchesPerFrame, not poolCapacity)
            int patchSplats = maxPatchesPerFrame * chunkSize;
            patchHeadersCPU  = new uint[maxPatchesPerFrame * 2];
            patchPosCPU      = new byte[patchSplats * ChunkedGSAsset.PositionStride];
            patchRotCPU      = new byte[patchSplats * ChunkedGSAsset.RotationStride];
            patchScaleCPU    = new byte[patchSplats * ChunkedGSAsset.ScaleStride];
            patchSHCPU       = new byte[patchSplats * ChunkedGSAsset.SHStride];
            patchSHBand1CPU  = shBands >= 1 ? new byte[patchSplats * ChunkedGSAsset.SHBand1Stride] : null;
            patchSHBand2CPU  = shBands >= 2 ? new byte[patchSplats * ChunkedGSAsset.SHBand2Stride] : null;
            patchSHBand3CPU  = shBands >= 3 ? new byte[patchSplats * ChunkedGSAsset.SHBand3Stride] : null;

            // Frustum planes
            basePlanes = new Plane[6];
            innerPlanes = new Plane[6];
            outerPlanes = new Plane[6];

            // Scratch
            innerVisibleList = new List<int>(chunkMeta.Length);
            outerVisibleList = new List<int>(chunkMeta.Length);
            toEvictList = new List<int>();
            toLoadList = new List<int>();

            // Cache octree from asset
            octreeNodes = asset.HasOctree ? asset.OctreeNodes : null;
            outerVisibleSet = new HashSet<int>();

            // Resolve file paths once (immutable, safe for background thread)
            posFilePath = asset.ResolveFilePath(ChunkedGSAsset.PositionsFileName);
            rotFilePath = asset.ResolveFilePath(ChunkedGSAsset.RotationsFileName);
            scaleFilePath = asset.ResolveFilePath(ChunkedGSAsset.ScalesFileName);
            shFilePath = asset.ResolveFilePath(ChunkedGSAsset.SHFileName);
            shRestFilePath = asset.SHRestCount > 0
                ? asset.ResolveFilePath(ChunkedGSAsset.SHRestFileName)
                : null;

            // Async I/O queues
            requestQueue = new ConcurrentQueue<ReadRequest>();
            completionQueue = new ConcurrentQueue<ReadResult>();
            ioWakeEvent = new ManualResetEventSlim(false);

            AllocateGPUBuffers();
            AllocateScatterBuffers();
            StartIOThread();
        }

        private void AllocateGPUBuffers()
        {
            positionBuffer = new ComputeBuffer(poolCapacity, ChunkedGSAsset.PositionStride);
            rotationBuffer = new ComputeBuffer(poolCapacity, ChunkedGSAsset.RotationStride);
            scaleBuffer = new ComputeBuffer(poolCapacity, ChunkedGSAsset.ScaleStride);
            shBuffer = new ComputeBuffer(poolCapacity, ChunkedGSAsset.SHStride);

            // Use float stride (4) so StructuredBuffer<float> and RWStructuredBuffer<float> bindings are correct without stride mismatch 
            // in both precompute and scatter shaders.
            shRestBuffer0 = new ComputeBuffer(shBands >= 1 ? poolCapacity * ChunkedGSAsset.SHBand1Count : 1, sizeof(float));
            shRestBuffer1 = new ComputeBuffer(shBands >= 2 ? poolCapacity * ChunkedGSAsset.SHBand2Count : 1, sizeof(float));
            shRestBuffer2 = new ComputeBuffer(shBands >= 3 ? poolCapacity * ChunkedGSAsset.SHBand3Count : 1, sizeof(float));

            // Remap buffer (one uint per potential active splat)
            remapGPU = new ComputeBuffer(Mathf.Max(1, poolCapacity), sizeof(uint));
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        ~GSChunkStreamer()
        {
            // Destructor: only stop the I/O thread. ComputeBuffer.Release() must
            // happen on the main thread so we cannot call it from the finalizer.
            // If we reach here it means Dispose() was not called -- log a warning.
            if (ioThreadRunning)
            {
                ioThreadRunning = false;
                ioWakeEvent?.Set();
            }
        }

        private void Dispose(bool disposing)
        {
            if (!disposing) return;

            // Shut down I/O thread
            ioThreadRunning = false;
            ioWakeEvent?.Set();
            if (ioThread != null && ioThread.IsAlive)
            {
                if (!ioThread.Join(3000))
                    Debug.LogWarning("GSChunkStreamer: I/O thread did not terminate within timeout.");
            }
            ioWakeEvent?.Dispose();

            // Release GPU resources
            positionBuffer?.Release();
            rotationBuffer?.Release();
            scaleBuffer?.Release();
            shBuffer?.Release();
            shRestBuffer0?.Release();
            shRestBuffer1?.Release();
            shRestBuffer2?.Release();
            remapGPU?.Release();

            positionBuffer = null;
            rotationBuffer = null;
            scaleBuffer = null;
            shBuffer = null;
            shRestBuffer0 = null;
            shRestBuffer1 = null;
            shRestBuffer2 = null;
            remapGPU = null;

            DisposePatchBuffers();

            readyChunkSet.Clear();
            pendingChunkSet.Clear();
            activeChunkOrder.Clear();
            currentActiveSplatCount = 0;
        }


        // ------------------------------------------------------------------
        //  Main thread: visibility update

        /// <summary>
        /// Main per frame update. Processes completed async reads, runs frustum culling, manages evictions and
        /// load requests, and rebuilds the remap buffer. Safe to call multiple times per frame (deduped by frame + camera).
        /// </summary>
        public void UpdateVisibility(Camera camera, Matrix4x4 modelMatrix)
        {
            int frame = Time.frameCount;
            int camId = camera.GetInstanceID();

            // Drain the completion queue exactly once per frame, regardless of how many cameras call this.
            // The first camera to call this frame does the drain; subsequent cameras skip it.
            // This prevents ProcessCompletedReads from resetting pendingPatchCount on a second camera call.
            if (frame != lastCompletionDrainFrame)
            {
                ProcessCompletedReads();
                lastCompletionDrainFrame = frame;
                camerasProcessedThisFrame.Clear();
            }

            // Skip frustum/eviction/load if this exact camera already ran this frame.
            // HashSet.Add returns false if the element was already present.
            if (!camerasProcessedThisFrame.Add(camId))
                return;

            // Step 2: frustum cull chunk AABBs (octree-accelerated)
            GeometryUtility.CalculateFrustumPlanes(camera, basePlanes);
            ExpandPlanes(basePlanes, loadMargin, innerPlanes);
            ExpandPlanes(basePlanes, unloadMargin, outerPlanes);

            innerVisibleList.Clear();
            outerVisibleList.Clear();

            if (octreeNodes != null && octreeNodes.Length > 0)
            {
                // Octree-accelerated frustum culling: O(visible nodes)
                ChunkOctree.QueryFrustum(octreeNodes, innerPlanes, modelMatrix, innerVisibleList);
                ChunkOctree.QueryFrustum(octreeNodes, outerPlanes, modelMatrix, outerVisibleList);

                // Update lastVisibleFrame for all chunks in the outer frustum
                outerVisibleSet.Clear();
                for (int i = 0; i < outerVisibleList.Count; i++)
                {
                    int idx = outerVisibleList[i];
                    chunks[idx].lastVisibleFrame = frame;
                    outerVisibleSet.Add(idx);
                }
            }
            else
            {
                // Fallback: linear scan of all chunks
                Debug.LogWarning("GSChunkStreamer: asset has no octree, falling back to linear chunk culling. " +
                    "Consider generating an octree for better performance on large scenes.");
                for (int i = 0; i < chunks.Length; i++)
                {
                    ChunkRuntime c = chunks[i];

                    if (c.metadata.IsVisibleInFrustum(outerPlanes, modelMatrix))
                    {
                        c.lastVisibleFrame = frame;
                        outerVisibleSet.Add(i);
                    }

                    if (c.metadata.IsVisibleInFrustum(innerPlanes, modelMatrix))
                        innerVisibleList.Add(i);
                }
            }

            // Step 3: determine and execute evictions
            toEvictList.Clear();

            // Check Ready chunks (loaded on GPU)
            for (int i = 0; i < activeChunkOrder.Count; i++)
            {
                int idx = activeChunkOrder[i];
                if ((frame - chunks[idx].lastVisibleFrame) > evictionDelayFrames)
                    toEvictList.Add(idx);
            }

            // Also evict Pending chunks whose reads are no longer needed
            // Iterate only the tracked pending set, not all chunks
            foreach (int pci in pendingChunkSet)
            {
                if ((frame - chunks[pci].lastVisibleFrame) > evictionDelayFrames)
                    toEvictList.Add(pci);
            }

            bool readyChunksEvicted = false;
            for (int i = 0; i < toEvictList.Count; i++)
            {
                int idx = toEvictList[i];
                if (chunks[idx].state == SlotState.Ready)
                    readyChunksEvicted = true;
                EvictChunk(idx);
            }

            // Compact remap on CPU after Ready chunk evictions (no GPU call).
            // O(active splats) but evictions are rare and it's pure CPU work.
            if (readyChunksEvicted)
            {
                CompactRemapCPU();
                dirtyRemapMin = 0;
                dirtyRemapMax = currentActiveSplatCount > 0 ? currentActiveSplatCount - 1 : -1;
            }

            // Step 4: determine new chunks to load
            // Cap new requests so the I/O thread + completion queue don't grow unbounded.
            toLoadList.Clear();
            int available = freeSlots.Count;
            int pendingBudget = MaxPendingRequests - pendingChunkSet.Count;

            for (int i = 0; i < innerVisibleList.Count; i++)
            {
                int idx = innerVisibleList[i];
                if (chunks[idx].state != SlotState.Unloaded) continue;
                if (available <= 0 || pendingBudget <= 0) break;

                toLoadList.Add(idx);
                available--;
                pendingBudget--;
            }

            // Dispatch async read requests
            for (int i = 0; i < toLoadList.Count; i++)
            {
                int idx = toLoadList[i];
                int slot = freeSlots.Pop();
                chunks[idx].state = SlotState.Pending;
                chunks[idx].slotIndex = slot;
                pendingChunkSet.Add(idx);
                requestQueue.Enqueue(new ReadRequest { chunkIndex = idx, slotIndex = slot });
            }

            if (toLoadList.Count > 0)
                ioWakeEvent.Set();

            // Update pending count for diagnostics
            PendingReadCount = requestQueue.Count;
        }


        // ------------------------------------------------------------------
        //  Eviction

        /// <summary>
        /// Evicts a chunk by returning its slot to the free list and clearing tracking state.
        /// Remap compaction (if needed) is handled by the caller after all evictions.
        /// </summary>
        private void EvictChunk(int chunkIndex)
        {
            ChunkRuntime c = chunks[chunkIndex];

            if (c.slotIndex >= 0)
                freeSlots.Push(c.slotIndex);

            c.state = SlotState.Unloaded;
            c.slotIndex = -1;
            c.remapStartIdx = -1;

            readyChunkSet.Remove(chunkIndex);
            pendingChunkSet.Remove(chunkIndex);
            bufferContentsDirty = true;
            statEvictCount++;
        }


        // ------------------------------------------------------------------
        //  Remap buffer

        /// <summary>
        /// Rebuilds the remap buffer on the CPU by iterating active chunks in logical order and writing their slot positions.
        /// Updates each chunk's remapStartIdx for its position in the remap buffer. This is a pure CPU operation that runs after evictions and before uploads, 
        /// so the remap GPU buffer is only updated once per frame with a single SetBufferData call for the dirty range.
        /// </summary>
        private void CompactRemapCPU()
        {
            // Rebuild activeChunkOrder from the authoritative set
            activeChunkOrder.Clear();
            foreach (int idx in readyChunkSet)
                activeChunkOrder.Add(idx);

            // Sort for deterministic logical index assignment across frames
            activeChunkOrder.Sort();

            int logicalIdx = 0;

            for (int i = 0; i < activeChunkOrder.Count; i++)
            {
                int chunkIdx = activeChunkOrder[i];
                ChunkRuntime c = chunks[chunkIdx];

                c.remapStartIdx = logicalIdx;
                int slotOffset = c.slotIndex * chunkSize;
                int splatCount = c.metadata.splatCount;

                for (int s = 0; s < splatCount; s++)
                {
                    if (logicalIdx >= poolCapacity) break;
                    remapCPU[logicalIdx++] = (uint)(slotOffset + s);
                }

                if (logicalIdx >= poolCapacity) break;
            }

            currentActiveSplatCount = logicalIdx;
        }


        // ------------------------------------------------------------------
        //  Frustum plane helpers

        /// <summary>
        /// Expands frustum planes by the given margin. Unity frustum planes normals are pointing inward. 
        /// Increasing distance pushes each plane outward.
        /// </summary>
        private static void ExpandPlanes(Plane[] source, float margin, Plane[] dest)
        {
            for (int i = 0; i < 6; i++)
            {
                Plane p = source[i];
                dest[i] = new Plane(p.normal, p.distance + margin);
            }
        }
    }
}
