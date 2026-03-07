using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using UnityEngine;

namespace GaussianSplatting
{
    /// <summary>
    /// GPU pool based chunk streamer with async I/O, CPU staging arrays, and batched GPU upload.
    /// - Fixed GPU pool: ComputeBuffers are allocated once at startup and divided into chunk sized slots. 
    ///   Slots are managed via a free list: loading a chunk pops a slot, evicting a chunk pushes the slot back. 
    /// - Async I/O: a dedicated background thread owns its own file handles and reads chunk data from disk. 
    ///   Completed reads are copied into CPU staging arrays (pure memcpy, no GPU calls).
    /// - Batched GPU upload: all per-frame changes are flushed in at most 6 SetData calls 
    ///   (5 attribute buffers + 1 remap), regardless of how many chunks changed.
    /// - Indirection buffer (remap): because slots are non contiguous, indexing is abstracted. 
    ///   A uint remap buffer maps each logical splat index to its physical position in the pool.
    /// </summary>
    public class GSChunkStreamer : IDisposable
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
            public int slotIndex;
        }

        // Result returned from the background thread with staging data
        private struct ReadResult
        {
            public int chunkIndex;
            public int slotIndex;
            public byte[] positions;
            public byte[] rotations;
            public byte[] scales;
            public byte[] sh;
            public byte[] shRest;
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


        // GPU pool buffers (allocated once, never reallocated)
        private ComputeBuffer positionBuffer;
        private ComputeBuffer rotationBuffer;
        private ComputeBuffer scaleBuffer;
        private ComputeBuffer shBuffer;
        private ComputeBuffer shRestBuffer;

        // Indirection / remap buffer
        private readonly uint[] remapCPU;
        private ComputeBuffer remapGPU;

        // CPU staging arrays — pre-allocated mirrors of the full GPU pool.
        // Chunk data is copied here on the main thread (CPU memcpy only), then
        // flushed to the GPU in at most 6 SetData calls per frame.
        private readonly byte[] stagingPositions;
        private readonly byte[] stagingRotations;
        private readonly byte[] stagingScales;
        private readonly byte[] stagingSH;
        private readonly byte[] stagingSHRest;   // null when SH rest stride is 0

        // Per-frame dirty tracking — attribute slot range (bounding dirty slots)
        private int dirtySlotMin = int.MaxValue;
        private int dirtySlotMax = -1;

        // Per-frame dirty tracking — remap index range
        private int dirtyRemapMin = int.MaxValue;
        private int dirtyRemapMax = -1;

        // Slot management
        private readonly Stack<int> freeSlots;
        private readonly HashSet<int> readyChunkSet;
        private readonly List<int> activeChunkOrder;

        // Async I/O infrastructure
        private readonly ConcurrentQueue<ReadRequest> requestQueue;
        private readonly ConcurrentQueue<ReadResult> completionQueue;
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

        // Frustum planes (scratch, reused each frame)
        private readonly Plane[] basePlanes;
        private readonly Plane[] innerPlanes;
        private readonly Plane[] outerPlanes;

        // Scratch lists (reused each frame to avoid allocations)
        private readonly List<int> innerVisibleList;
        private readonly List<int> toEvictList;
        private readonly List<int> toLoadList;

        // Current state
        private int currentActiveSplatCount;
        private int lastProcessedFrame = -1;
        private int lastCameraID = -1;
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
        public ComputeBuffer SHRestBuffer => shRestBuffer;
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

        /// <summary>
        /// Creates a new chunk streamer with a fixed GPU pool and async I/O thread.
        /// </summary>
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

            // Cap allocation to what the scene actually needs
            int effectiveMax = Mathf.Min(maxVisibleSplats, asset.totalSplatCount);
            this.maxSlots = Mathf.Max(1, Mathf.CeilToInt((float)effectiveMax / chunkSize));
            int uncappedCapacity = maxSlots * chunkSize;

            // Each staging array is byte[poolCapacity * stride], but arrays are limited to int.MaxValue elements 
            // (~2 GB for byte[]), so we must cap poolCapacity to keep the largest staging array within that limit.
            int largestStride = Mathf.Max(
                Mathf.Max(ChunkedGSAsset.PositionStride, ChunkedGSAsset.RotationStride),
                Mathf.Max(ChunkedGSAsset.ScaleStride,
                    Mathf.Max(ChunkedGSAsset.SHStride, shRestStride)));
            int maxPoolForStaging = int.MaxValue / largestStride;
            // Round down to whole slots so poolCapacity stays a multiple of chunkSize
            int maxSlotsForStaging = maxPoolForStaging / chunkSize;
            if (maxSlots > maxSlotsForStaging)
            {
                Debug.LogWarning(
                    $"GSChunkStreamer: capping pool from {maxSlots} to {maxSlotsForStaging} slots " +
                    $"({maxSlotsForStaging * chunkSize} splats) to keep staging arrays under 2 GB. " +
                    $"Scene has {asset.totalSplatCount} splats total; streaming will page the rest.");
                this.maxSlots = maxSlotsForStaging;
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
            activeChunkOrder = new List<int>(maxSlots);

            // Remap array: sized at pool capacity so it can hold the maximum
            // number of active splats when all slots are occupied
            remapCPU = new uint[poolCapacity];

            // CPU staging arrays: full-pool mirrors for batched GPU uploads.
            // Pre-allocated once to avoid per-frame allocation and GC pressure.
            // Sizes are guaranteed to fit in int by the cap above.
            stagingPositions = new byte[poolCapacity * ChunkedGSAsset.PositionStride];
            stagingRotations = new byte[poolCapacity * ChunkedGSAsset.RotationStride];
            stagingScales    = new byte[poolCapacity * ChunkedGSAsset.ScaleStride];
            stagingSH        = new byte[poolCapacity * ChunkedGSAsset.SHStride];
            stagingSHRest    = shRestStride > 0
                ? new byte[poolCapacity * shRestStride]
                : null;

            // Frustum planes
            basePlanes = new Plane[6];
            innerPlanes = new Plane[6];
            outerPlanes = new Plane[6];

            // Scratch
            innerVisibleList = new List<int>(chunkMeta.Length);
            toEvictList = new List<int>();
            toLoadList = new List<int>();

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
            StartIOThread();
        }

        private void AllocateGPUBuffers()
        {
            positionBuffer = new ComputeBuffer(poolCapacity, ChunkedGSAsset.PositionStride);
            rotationBuffer = new ComputeBuffer(poolCapacity, ChunkedGSAsset.RotationStride);
            scaleBuffer = new ComputeBuffer(poolCapacity, ChunkedGSAsset.ScaleStride);
            shBuffer = new ComputeBuffer(poolCapacity, ChunkedGSAsset.SHStride);

            if (asset.SHRestCount > 0)
                shRestBuffer = new ComputeBuffer(poolCapacity, shRestStride);

            // Remap buffer (one uint per potential active splat)
            remapGPU = new ComputeBuffer(Mathf.Max(1, poolCapacity), sizeof(uint));
        }

        private void StartIOThread()
        {
            ioThreadRunning = true;
            ioThread = new Thread(IOThreadMain)
            {
                Name = "GSChunkIO",
                IsBackground = true,
                Priority = System.Threading.ThreadPriority.BelowNormal
            };
            ioThread.Start();
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
            shRestBuffer?.Release();
            remapGPU?.Release();

            positionBuffer = null;
            rotationBuffer = null;
            scaleBuffer = null;
            shBuffer = null;
            shRestBuffer = null;
            remapGPU = null;

            readyChunkSet.Clear();
            activeChunkOrder.Clear();
            currentActiveSplatCount = 0;
        }


        // ------------------------------------------------------------------
        //  Background I/O thread

        /// <summary>
        /// Entry point for the background I/O thread. Opens its own set of file
        /// handles (independent of the main thread) and reads chunks on demand.
        /// Results are placed in the completion queue for the main thread.
        /// </summary>
        private void IOThreadMain()
        {
            FileStream posFs = OpenFileHandle(posFilePath);
            FileStream rotFs = OpenFileHandle(rotFilePath);
            FileStream scaleFs = OpenFileHandle(scaleFilePath);
            FileStream shFs = OpenFileHandle(shFilePath);
            FileStream shRestFs = !string.IsNullOrEmpty(shRestFilePath)
                ? OpenFileHandle(shRestFilePath)
                : null;

            try
            {
                while (ioThreadRunning)
                {
                    // Drain all queued requests
                    while (requestQueue.TryDequeue(out ReadRequest req))
                    {
                        if (!ioThreadRunning) return;
                        ReadResult result = ExecuteChunkRead(req, posFs, rotFs, scaleFs, shFs, shRestFs);
                        completionQueue.Enqueue(result);
                    }

                    // Sleep until new work arrives or shutdown is signaled.
                    // The 50ms timeout prevents the thread from sleeping forever
                    // if the wake event fires before Wait is called.
                    ioWakeEvent.Wait(50);
                    ioWakeEvent.Reset();
                }
            }
            finally
            {
                posFs?.Dispose();
                rotFs?.Dispose();
                scaleFs?.Dispose();
                shFs?.Dispose();
                shRestFs?.Dispose();
            }
        }

        private static FileStream OpenFileHandle(string path)
        {
            if (string.IsNullOrEmpty(path) || !File.Exists(path)) return null;
            return new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 65536);
        }

        private ReadResult ExecuteChunkRead(
            ReadRequest req,
            FileStream posFs, FileStream rotFs,
            FileStream scaleFs, FileStream shFs, FileStream shRestFs)
        {
            ChunkInfo meta = asset.Chunks[req.chunkIndex];
            int count = meta.splatCount;
            int start = meta.startIndex;

            ReadResult r;
            r.chunkIndex = req.chunkIndex;
            r.slotIndex = req.slotIndex;
            r.positions = ReadRegion(posFs, start, count, ChunkedGSAsset.PositionStride);
            r.rotations = ReadRegion(rotFs, start, count, ChunkedGSAsset.RotationStride);
            r.scales = ReadRegion(scaleFs, start, count, ChunkedGSAsset.ScaleStride);
            r.sh = ReadRegion(shFs, start, count, ChunkedGSAsset.SHStride);
            r.shRest = (shRestFs != null && shRestStride > 0)
                ? ReadRegion(shRestFs, start, count, shRestStride)
                : null;

            bool coreOk =
                r.positions != null &&
                r.rotations != null &&
                r.scales != null &&
                r.sh != null;

            bool shRestOk = shRestStride <= 0 || r.shRest != null;

            r.ok = coreOk && shRestOk;
            r.error = r.ok ? null : "missing or truncated chunk payload";
            return r;
        }

        /// <summary>
        /// Reads a contiguous region of a data file into a new byte array.
        /// Handles partial OS reads with a loop.
        /// </summary>
        private static byte[] ReadRegion(FileStream fs, int startSplat, int splatCount, int stride)
        {
            if (fs == null) return null;

            long byteOffset = startSplat * (long)stride;
            int byteCount = splatCount * stride;
            byte[] data = new byte[byteCount];

            fs.Seek(byteOffset, SeekOrigin.Begin);
            int totalRead = 0;
            while (totalRead < byteCount)
            {
                int read = fs.Read(data, totalRead, byteCount - totalRead);
                if (read <= 0) break;
                totalRead += read;
            }

            // Important: don't return partial payloads
            if (totalRead != byteCount)
                return null;

            return data;
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
            if (frame == lastProcessedFrame && camId == lastCameraID) return;
            lastProcessedFrame = frame;
            lastCameraID = camId;

            // Step 1: drain completed async reads into CPU staging (no GPU calls)
            ProcessCompletedReads();

            // Step 2: frustum cull all chunk AABBs
            GeometryUtility.CalculateFrustumPlanes(camera, basePlanes);
            ExpandPlanes(basePlanes, loadMargin, innerPlanes);
            ExpandPlanes(basePlanes, unloadMargin, outerPlanes);

            innerVisibleList.Clear();
            for (int i = 0; i < chunks.Length; i++)
            {
                ChunkRuntime c = chunks[i];

                if (c.metadata.IsVisibleInFrustum(outerPlanes, modelMatrix))
                    c.lastVisibleFrame = frame;

                if (c.metadata.IsVisibleInFrustum(innerPlanes, modelMatrix))
                    innerVisibleList.Add(i);
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
            for (int i = 0; i < chunks.Length; i++)
            {
                if (chunks[i].state == SlotState.Pending &&
                    (frame - chunks[i].lastVisibleFrame) > evictionDelayFrames)
                {
                    toEvictList.Add(i);
                }
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
            toLoadList.Clear();
            int available = freeSlots.Count;

            for (int i = 0; i < innerVisibleList.Count; i++)
            {
                int idx = innerVisibleList[i];
                if (chunks[idx].state != SlotState.Unloaded) continue;
                if (available <= 0) break;

                toLoadList.Add(idx);
                available--;
            }

            // Dispatch async read requests
            for (int i = 0; i < toLoadList.Count; i++)
            {
                int idx = toLoadList[i];
                int slot = freeSlots.Pop();
                chunks[idx].state = SlotState.Pending;
                chunks[idx].slotIndex = slot;
                requestQueue.Enqueue(new ReadRequest { chunkIndex = idx, slotIndex = slot });
            }

            if (toLoadList.Count > 0)
                ioWakeEvent.Set();

            // Update pending count for diagnostics
            PendingReadCount = requestQueue.Count;

            // Step 5: flush all changes to GPU — max 6 SetData calls (5 attribute buffers covering 
            // the bounding dirty slot range, 1 remap buffer covering the dirty index range)
            FlushToGPU();
        }


        // ------------------------------------------------------------------
        //  Completion processing and GPU upload

        /// <summary>
        /// Drains ALL completed async reads in one pass. Chunk data is copied into the pre-allocated CPU 
        /// staging arrays (pure memcpy, no GPU calls). Stale results for chunks evicted before their read 
        /// finished are discarded.
        /// </summary>
        private void ProcessCompletedReads()
        {
            while (completionQueue.TryDequeue(out ReadResult result))
            {
                if (result.chunkIndex < 0 || result.chunkIndex >= chunks.Length)
                    continue;

                ChunkRuntime c = chunks[result.chunkIndex];
                if (c.state != SlotState.Pending || c.slotIndex != result.slotIndex)
                    continue;

                if (!result.ok)
                {
                    int slot = c.slotIndex;
                    c.state = SlotState.Unloaded;
                    c.slotIndex = -1;
                    if (slot >= 0)
                        freeSlots.Push(slot);
                    Debug.LogError($"GSChunkStreamer: failed to load chunk {result.chunkIndex}: {result.error}");
                    continue;
                }

                // Copy into pre-allocated staging arrays (CPU memcpy, zero GPU cost)
                CopyToStaging(result);

                c.state = SlotState.Ready;
                readyChunkSet.Add(result.chunkIndex);
                activeChunkOrder.Add(result.chunkIndex);

                // Append remap entries for this chunk at the tail
                int slotOffset = c.slotIndex * chunkSize;
                int splatCount = c.metadata.splatCount;
                int appendCount = Mathf.Min(splatCount, poolCapacity - currentActiveSplatCount);
                c.remapStartIdx = currentActiveSplatCount;
                for (int s = 0; s < appendCount; s++)
                    remapCPU[currentActiveSplatCount + s] = (uint)(slotOffset + s);

                // Track dirty remap range (append region)
                if (dirtyRemapMin > currentActiveSplatCount)
                    dirtyRemapMin = currentActiveSplatCount;
                currentActiveSplatCount += appendCount;
                dirtyRemapMax = currentActiveSplatCount - 1;

                bufferContentsDirty = true;
                statLoadCount++;
            }
        }

        /// <summary>
        /// Copies one chunk's payload from the async read result into the pre-allocated CPU staging arrays 
        /// at the correct slot offset. Pure CPU memcpy via Buffer.BlockCopy. Also expands the per-frame dirty slot
        /// range so FlushToGPU uploads the bounding region in one SetData per attribute.
        /// </summary>
        private void CopyToStaging(ReadResult result)
        {
            int slot = result.slotIndex;
            int splatCount = chunks[result.chunkIndex].metadata.splatCount;
            int splatBase = slot * chunkSize;

            CopyRegion(result.positions, stagingPositions, splatBase, splatCount, ChunkedGSAsset.PositionStride);
            CopyRegion(result.rotations, stagingRotations, splatBase, splatCount, ChunkedGSAsset.RotationStride);
            CopyRegion(result.scales,    stagingScales,    splatBase, splatCount, ChunkedGSAsset.ScaleStride);
            CopyRegion(result.sh,        stagingSH,        splatBase, splatCount, ChunkedGSAsset.SHStride);
            CopyRegion(result.shRest,    stagingSHRest,    splatBase, splatCount, shRestStride);

            // Expand dirty slot range
            if (slot < dirtySlotMin) dirtySlotMin = slot;
            if (slot > dirtySlotMax) dirtySlotMax = slot;
        }

        /// <summary>
        /// Copies splatCount * stride bytes from src into dst at the given splat offset.
        /// Null-safe: skips if either array is null (e.g. SH rest when band count is 0).
        /// </summary>
        private static void CopyRegion(byte[] src, byte[] dst, int dstSplatStart, int splatCount, int stride)
        {
            if (src == null || dst == null || stride <= 0) return;
            Buffer.BlockCopy(src, 0, dst, dstSplatStart * stride, splatCount * stride);
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
            bufferContentsDirty = true;
            statEvictCount++;
        }


        // ------------------------------------------------------------------
        //  Remap buffer

        /// <summary>
        /// CPU-only remap compaction: rebuilds remapCPU from the authoritative set of
        /// Ready chunks. No GPU upload — that is deferred to FlushToGPU. Also updates
        /// each chunk's remapStartIdx so future incremental appends are consistent.
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
        //  Batched GPU flush

        /// <summary>
        /// Uploads all per-frame dirty data to the GPU in at most 6 SetData calls: up to 5 for attribute 
        /// buffers (bounding dirty slot range) and 1 for the remap buffer (bounding dirty index range). 
        /// Because the driver scheduling overhead is per-call not per-byte, this avoids the N×5 call storm that
        /// caused TDR crashes with individual per-chunk uploads.
        /// </summary>
        private void FlushToGPU()
        {
            // --- Attribute buffers: one SetData per attribute covering the bounding dirty slot range ---
            if (dirtySlotMin <= dirtySlotMax)
            {
                int startSplat = dirtySlotMin * chunkSize;
                int endSplat = Mathf.Min((dirtySlotMax + 1) * chunkSize, poolCapacity);

                FlushAttribute(positionBuffer, stagingPositions, ChunkedGSAsset.PositionStride, startSplat, endSplat);
                FlushAttribute(rotationBuffer, stagingRotations, ChunkedGSAsset.RotationStride, startSplat, endSplat);
                FlushAttribute(scaleBuffer,    stagingScales,    ChunkedGSAsset.ScaleStride,    startSplat, endSplat);
                FlushAttribute(shBuffer,       stagingSH,        ChunkedGSAsset.SHStride,       startSplat, endSplat);

                if (shRestBuffer != null && stagingSHRest != null)
                    FlushAttribute(shRestBuffer, stagingSHRest, shRestStride, startSplat, endSplat);

                dirtySlotMin = int.MaxValue;
                dirtySlotMax = -1;
            }

            // --- Remap buffer: one SetData covering the dirty index range ---
            if (dirtyRemapMin <= dirtyRemapMax)
            {
                int count = dirtyRemapMax - dirtyRemapMin + 1;
                remapGPU.SetData(remapCPU, dirtyRemapMin, dirtyRemapMin, count);

                dirtyRemapMin = int.MaxValue;
                dirtyRemapMax = -1;
            }
        }

        /// <summary>
        /// Uploads one attribute's bounding dirty range from the CPU staging array
        /// to the corresponding GPU ComputeBuffer. Single SetData call.
        /// </summary>
        private static void FlushAttribute(
            ComputeBuffer gpuBuffer, byte[] staging, int stride,
            int startSplat, int endSplat)
        {
            int startByte = startSplat * stride;
            int byteCount = (endSplat - startSplat) * stride;
            if (byteCount <= 0) return;
            gpuBuffer.SetData(staging, startByte, startByte, byteCount);
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
