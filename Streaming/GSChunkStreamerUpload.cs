using System;
using Unity.Collections;
using UnityEngine;

namespace GaussianSplatting
{
    /// <summary>
    /// Partial class: scatter upload infrastructure, patch batching, GPU flush via LockBufferForWrite + compute dispatch.
    /// </summary>
    public partial class GSChunkStreamer
    {
        // ------------------------------------------------------------------
        //  Scatter upload constants and fields

        /// <summary>
        /// Per-frame byte budget for GPU uploads. 128 MB/frame safely fits in the DX12 upload ring-buffer without stalling GPU.
        /// </summary>
        private const int MaxUploadBytesPerFrame = 128 * 1024 * 1024;

        /// <summary> Maximum outstanding IO requests to bound memory from in-flight byte[] arrays. </summary>
        private const int MaxPendingRequests = 256;

        // Compute shader for GPU-side scatter copy
        private ComputeShader scatterShader;
        private int kernelFloat3, kernelFloat4, kernelFloats;

        // Upload sizing (computed once in constructor)
        private readonly int bytesPerChunk;
        private readonly int maxPatchesPerFrame;

        // Patch GraphicsBuffer: compact buffers sized to maxPatchesPerFrame chunks, uploaded once per frame via LockBufferForWrite.
        private GraphicsBuffer patchHeaders;
        private GraphicsBuffer patchPositions;
        private GraphicsBuffer patchRotations;
        private GraphicsBuffer patchScales;
        private GraphicsBuffer patchSH;
        private GraphicsBuffer patchSHBand1;   // null when shBands < 1
        private GraphicsBuffer patchSHBand2;   // null when shBands < 2
        private GraphicsBuffer patchSHBand3;   // null when shBands < 3

        // CPU-side patch staging: flat arrays sized to maxPatchesPerFrame * chunkSize.
        private readonly uint[] patchHeadersCPU;
        private readonly byte[] patchPosCPU;
        private readonly byte[] patchRotCPU;
        private readonly byte[] patchScaleCPU;
        private readonly byte[] patchSHCPU;
        private readonly byte[] patchSHBand1CPU;   // null when shBands < 1
        private readonly byte[] patchSHBand2CPU;   // null when shBands < 2
        private readonly byte[] patchSHBand3CPU;   // null when shBands < 3

        // How many dirty patches have been queued this frame
        private int pendingPatchCount = 0;

        // ------------------------------------------------------------------
        /// <summary>
        /// Submits scatter dispatches and remap update into the provided CommandBuffer. Must be called after UpdateVisibility 
        /// and before any shader that reads the pool buffers.
        /// </summary>
        public void FlushScatterCommandBuffer(UnityEngine.Rendering.ComputeCommandBuffer cmd)
        {
            if (pendingPatchCount > 0 && scatterShader != null)
            {
                int totalPatchSplats = pendingPatchCount * chunkSize;

                // Upload patch data via LockBufferForWrite (zero-copy, no intermediate staging buffers needed)
                UploadHeaders(patchHeaders, patchHeadersCPU, pendingPatchCount * 2);
                UploadBytes(patchPositions, patchPosCPU,   totalPatchSplats * ChunkedGSAsset.PositionStride);
                UploadBytes(patchRotations, patchRotCPU,   totalPatchSplats * ChunkedGSAsset.RotationStride);
                UploadBytes(patchScales,    patchScaleCPU, totalPatchSplats * ChunkedGSAsset.ScaleStride);
                UploadBytes(patchSH,        patchSHCPU,    totalPatchSplats * ChunkedGSAsset.SHStride);
                if (patchSHBand1 != null && patchSHBand1CPU != null)
                    UploadBytes(patchSHBand1, patchSHBand1CPU, totalPatchSplats * ChunkedGSAsset.SHBand1Stride);
                if (patchSHBand2 != null && patchSHBand2CPU != null)
                    UploadBytes(patchSHBand2, patchSHBand2CPU, totalPatchSplats * ChunkedGSAsset.SHBand2Stride);
                if (patchSHBand3 != null && patchSHBand3CPU != null)
                    UploadBytes(patchSHBand3, patchSHBand3CPU, totalPatchSplats * ChunkedGSAsset.SHBand3Stride);

                // Dispatch scatter compute kernels through the command buffer so that render graph dependencies are tracked correctly.
                cmd.SetComputeIntParam(scatterShader, "_ChunkSize", chunkSize);
                int xGroups = Mathf.CeilToInt(chunkSize / 64f);

                // Bind _PatchHeaders once per kernel
                cmd.SetComputeBufferParam(scatterShader, kernelFloat3, "_PatchHeaders", patchHeaders);
                cmd.SetComputeBufferParam(scatterShader, kernelFloat4, "_PatchHeaders", patchHeaders);

                // Positions (float3)
                cmd.SetComputeBufferParam(scatterShader, kernelFloat3, "_PatchFloat3", patchPositions);
                cmd.SetComputeBufferParam(scatterShader, kernelFloat3, "_PoolFloat3", positionBuffer);
                cmd.DispatchCompute(scatterShader, kernelFloat3, xGroups, pendingPatchCount, 1);

                // Scales (float3, reuse kernel)
                cmd.SetComputeBufferParam(scatterShader, kernelFloat3, "_PatchFloat3", patchScales);
                cmd.SetComputeBufferParam(scatterShader, kernelFloat3, "_PoolFloat3", scaleBuffer);
                cmd.DispatchCompute(scatterShader, kernelFloat3, xGroups, pendingPatchCount, 1);

                // Rotations (float4)
                cmd.SetComputeBufferParam(scatterShader, kernelFloat4, "_PatchFloat4", patchRotations);
                cmd.SetComputeBufferParam(scatterShader, kernelFloat4, "_PoolFloat4", rotationBuffer);
                cmd.DispatchCompute(scatterShader, kernelFloat4, xGroups, pendingPatchCount, 1);

                // SH DC (float4, reuse kernel)
                cmd.SetComputeBufferParam(scatterShader, kernelFloat4, "_PatchFloat4", patchSH);
                cmd.SetComputeBufferParam(scatterShader, kernelFloat4, "_PoolFloat4", shBuffer);
                cmd.DispatchCompute(scatterShader, kernelFloat4, xGroups, pendingPatchCount, 1);

                // SH Band 1 (9 floats)
                if (shBands >= 1 && patchSHBand1 != null)
                {
                    cmd.SetComputeBufferParam(scatterShader, kernelFloats, "_PatchHeaders", patchHeaders);
                    cmd.SetComputeIntParam(scatterShader, "_FloatsPerSplat", ChunkedGSAsset.SHBand1Count);
                    cmd.SetComputeBufferParam(scatterShader, kernelFloats, "_PatchFloats", patchSHBand1);
                    cmd.SetComputeBufferParam(scatterShader, kernelFloats, "_PoolFloats", shRestBuffer0);
                    cmd.DispatchCompute(scatterShader, kernelFloats, xGroups, pendingPatchCount, 1);
                }

                // SH Band 2 (15 floats)
                if (shBands >= 2 && patchSHBand2 != null)
                {
                    cmd.SetComputeIntParam(scatterShader, "_FloatsPerSplat", ChunkedGSAsset.SHBand2Count);
                    cmd.SetComputeBufferParam(scatterShader, kernelFloats, "_PatchFloats", patchSHBand2);
                    cmd.SetComputeBufferParam(scatterShader, kernelFloats, "_PoolFloats", shRestBuffer1);
                    cmd.DispatchCompute(scatterShader, kernelFloats, xGroups, pendingPatchCount, 1);
                }

                // SH Band 3 (21 floats)
                if (shBands >= 3 && patchSHBand3 != null)
                {
                    cmd.SetComputeIntParam(scatterShader, "_FloatsPerSplat", ChunkedGSAsset.SHBand3Count);
                    cmd.SetComputeBufferParam(scatterShader, kernelFloats, "_PatchFloats", patchSHBand3);
                    cmd.SetComputeBufferParam(scatterShader, kernelFloats, "_PoolFloats", shRestBuffer2);
                    cmd.DispatchCompute(scatterShader, kernelFloats, xGroups, pendingPatchCount, 1);
                }

                pendingPatchCount = 0;
            }

            // Remap buffer (simple uint array, single SetData)
            if (dirtyRemapMin <= dirtyRemapMax)
            {
                int count = dirtyRemapMax - dirtyRemapMin + 1;
                cmd.SetBufferData(remapGPU, remapCPU, dirtyRemapMin, dirtyRemapMin, count);
                dirtyRemapMin = int.MaxValue;
                dirtyRemapMax = -1;
            }
        }


        // ------------------------------------------------------------------
        //  Allocation and disposal of scatter resources
        private void AllocateScatterBuffers()
        {
            scatterShader = Resources.Load<ComputeShader>("GSScatterUpload");
            if (scatterShader == null)
            {
                Debug.LogError("GSChunkStreamer: GSScatterUpload compute shader not found in Resources.");
                return;
            }

            kernelFloat3 = scatterShader.FindKernel("ScatterFloat3");
            kernelFloat4 = scatterShader.FindKernel("ScatterFloat4");
            kernelFloats = scatterShader.FindKernel("ScatterFloats");

            int patchSplats = maxPatchesPerFrame * chunkSize;
            var lockFlag = GraphicsBuffer.UsageFlags.LockBufferForWrite;
            var target   = GraphicsBuffer.Target.Structured;

            patchHeaders   = new GraphicsBuffer(target, lockFlag, maxPatchesPerFrame, sizeof(uint) * 2);
            patchPositions = new GraphicsBuffer(target, lockFlag, patchSplats, ChunkedGSAsset.PositionStride);
            patchRotations = new GraphicsBuffer(target, lockFlag, patchSplats, ChunkedGSAsset.RotationStride);
            patchScales    = new GraphicsBuffer(target, lockFlag, patchSplats, ChunkedGSAsset.ScaleStride);
            patchSH        = new GraphicsBuffer(target, lockFlag, patchSplats, ChunkedGSAsset.SHStride);

            if (shBands >= 1)
                patchSHBand1 = new GraphicsBuffer(target, lockFlag,
                    patchSplats * ChunkedGSAsset.SHBand1Count, sizeof(float));
            if (shBands >= 2)
                patchSHBand2 = new GraphicsBuffer(target, lockFlag,
                    patchSplats * ChunkedGSAsset.SHBand2Count, sizeof(float));
            if (shBands >= 3)
                patchSHBand3 = new GraphicsBuffer(target, lockFlag,
                    patchSplats * ChunkedGSAsset.SHBand3Count, sizeof(float));
        }

        private void DisposePatchBuffers()
        {
            patchHeaders?.Dispose();   patchHeaders = null;
            patchPositions?.Dispose(); patchPositions = null;
            patchRotations?.Dispose(); patchRotations = null;
            patchScales?.Dispose();    patchScales = null;
            patchSH?.Dispose();        patchSH = null;
            patchSHBand1?.Dispose();   patchSHBand1 = null;
            patchSHBand2?.Dispose();   patchSHBand2 = null;
            patchSHBand3?.Dispose();   patchSHBand3 = null;
        }


        // ------------------------------------------------------------------
        //  Completion processing

        /// <summary>
        /// Drains completed async reads up to the per-frame byte budget. Each result is appended to the compact 
        /// patch batch for scatter upload. Stale results (evicted chunks) are discarded and their rented buffers returned.
        /// </summary>
        private void ProcessCompletedReads()
        {
            int uploadedBytes = 0;

            while (uploadedBytes + bytesPerChunk <= MaxUploadBytesPerFrame
                && pendingPatchCount < maxPatchesPerFrame
                && completionQueue.TryDequeue(out ReadResult result))
            {
                // Discard results for unknown chunks
                if (result.chunkIndex < 0 || result.chunkIndex >= chunks.Length)
                {
                    ReturnReadResultBuffers(result);
                    continue;
                }

                ChunkRuntime c = chunks[result.chunkIndex];

                // Discard stale results (chunk evicted while read was in flight)
                if (c.state != SlotState.Pending || c.slotIndex != result.slotIndex)
                {
                    ReturnReadResultBuffers(result);
                    continue;
                }

                if (!result.ok)
                {
                    ReturnReadResultBuffers(result);
                    int slot = c.slotIndex;
                    c.state = SlotState.Unloaded;
                    c.slotIndex = -1;
                    pendingChunkSet.Remove(result.chunkIndex);
                    if (slot >= 0) freeSlots.Push(slot);
                    Debug.LogError($"GSChunkStreamer: failed to load chunk {result.chunkIndex}: {result.error}");
                    continue;
                }

                // Append to compact patch batch (CPU memcpy + return rented buffers)
                AppendToPatchBatch(result);

                c.state = SlotState.Ready;
                pendingChunkSet.Remove(result.chunkIndex);
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
                uploadedBytes += bytesPerChunk;
            }
        }

        /// <summary> Returns all rented buffers in a ReadResult to the pool. </summary>
        private void ReturnReadResultBuffers(ReadResult result)
        {
            ReturnBuffer(result.positions);
            ReturnBuffer(result.rotations);
            ReturnBuffer(result.scales);
            ReturnBuffer(result.sh);
            ReturnBuffer(result.shBand1);
            ReturnBuffer(result.shBand2);
            ReturnBuffer(result.shBand3);
        }

        /// <summary>
        /// Copies one chunk's attribute data from the read result into the compact patch CPU arrays at the current 
        /// pendingPatchCount offset, then returns all rented buffers to the pool.
        /// </summary>
        private void AppendToPatchBatch(ReadResult result)
        {
            int patchIdx   = pendingPatchCount;
            int splatCount = chunks[result.chunkIndex].metadata.splatCount;
            int batchBase  = patchIdx * chunkSize;

            // Patch header: (slotIndex, splatCount)
            patchHeadersCPU[patchIdx * 2]     = (uint)result.slotIndex;
            patchHeadersCPU[patchIdx * 2 + 1] = (uint)splatCount;

            // Attribute data into compact patch arrays
            CopyPatch(result.positions, patchPosCPU,      batchBase, splatCount, ChunkedGSAsset.PositionStride);
            CopyPatch(result.rotations, patchRotCPU,      batchBase, splatCount, ChunkedGSAsset.RotationStride);
            CopyPatch(result.scales,    patchScaleCPU,    batchBase, splatCount, ChunkedGSAsset.ScaleStride);
            CopyPatch(result.sh,        patchSHCPU,       batchBase, splatCount, ChunkedGSAsset.SHStride);
            CopyPatch(result.shBand1,   patchSHBand1CPU,  batchBase, splatCount, ChunkedGSAsset.SHBand1Stride);
            CopyPatch(result.shBand2,   patchSHBand2CPU,  batchBase, splatCount, ChunkedGSAsset.SHBand2Stride);
            CopyPatch(result.shBand3,   patchSHBand3CPU,  batchBase, splatCount, ChunkedGSAsset.SHBand3Stride);

            // Return rented buffers: data has been copied into the patch arrays
            ReturnReadResultBuffers(result);

            pendingPatchCount++;
        }

        private static void CopyPatch(byte[] src, byte[] dst, int dstSplatBase, int splatCount, int stride)
        {
            if (src == null || dst == null) return;
            Buffer.BlockCopy(src, 0, dst, dstSplatBase * stride, splatCount * stride);
        }


        // ------------------------------------------------------------------
        //  Zero-copy upload helpers (LockBufferForWrite)

        private static void UploadHeaders(GraphicsBuffer gpuBuffer, uint[] cpuData, int elementCount)
        {
            if (gpuBuffer == null || cpuData == null || elementCount <= 0) return;
            var mapped = gpuBuffer.LockBufferForWrite<uint>(0, elementCount);
            NativeArray<uint>.Copy(cpuData, 0, mapped, 0, elementCount);
            gpuBuffer.UnlockBufferAfterWrite<uint>(elementCount);
        }

        private static void UploadBytes(GraphicsBuffer gpuBuffer, byte[] cpuData, int byteCount)
        {
            if (gpuBuffer == null || cpuData == null || byteCount <= 0) return;
            var mapped = gpuBuffer.LockBufferForWrite<byte>(0, byteCount);
            NativeArray<byte>.Copy(cpuData, 0, mapped, 0, byteCount);
            gpuBuffer.UnlockBufferAfterWrite<byte>(byteCount);
        }
    }
}
