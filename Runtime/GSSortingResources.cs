using UnityEngine;

namespace GaussianSplatting
{
    /// <summary>
    /// Manages GPU resources required for depth-based Gaussian splat sorting (allocation and management of compute buffers used
    /// in the sorting pipeline).
    /// </summary>
    public class GSSortingResources : System.IDisposable
    {
        // From SortCommon.hlsl, constants for radix sort
        private const int RADIX = 256;          // Number of digit bins
        private const int RADIX_PASSES = 4;     // Number of radix passes (32-bit / 8-bit)
        private const int PART_SIZE = 3840;     // Partition size for radix sort


        // Primary sort buffers
        private ComputeBuffer sortIndices;      // uint[splatCount] - Sort indices (payload) - maps sorted position to original splat index
        private ComputeBuffer sortIndicesAlt;   // uint[splatCount] - alternate for ping-pong
        private ComputeBuffer sortKeys;         // float[splatCount] - Sort keys - depth for radix sort
        private ComputeBuffer sortKeysAlt;      // float[splatCount] - alternate for ping-pong

        // View data buffer for caching view-space calculations
        private ComputeBuffer viewData;         // float4[splatCount] - View data - cached view-space positions and visibility flags (xyz=viewPos, w=visible)
        
        // Radix sort histogram buffers
        private ComputeBuffer globalHistogram;  // uint[RADIX * RADIX_PASSES] - global histogram
        private ComputeBuffer passHistogram;    // uint[RADIX * threadBlocks] - per-pass histogram, used during sorting
        
        // Constant buffer for sort parameters
        private ComputeBuffer sortParams;       // cbuffer with numKeys, radixShift, threadBlocks
        
        // Public properties
        public int SplatCount { get; private set; }
        public int ThreadBlocks { get; private set; }
        
        // Legacy tile properties (kept for compatibility, but not used in depth-only sorting)
        public int TileCountX => 1;
        public int TileCountY => 1;
        public int TileSize => 1;
        
        public bool IsInitialized => sortIndices != null && sortKeys != null;
        
        // Public accessors for buffers
        public ComputeBuffer SortIndices => sortIndices;
        public ComputeBuffer SortIndicesAlt => sortIndicesAlt;
        public ComputeBuffer SortKeys => sortKeys;
        public ComputeBuffer SortKeysAlt => sortKeysAlt;
        public ComputeBuffer ViewData => viewData;
        public ComputeBuffer GlobalHistogram => globalHistogram;
        public ComputeBuffer PassHistogram => passHistogram;
        

        /// <summary>
        /// Initialize sorting resources for the given splat count.
        /// </summary>
        /// <param name="splatCount">Number of splats to sort</param>
        public void Initialize(int splatCount)
        {
            // Release existing buffers if sizes changed
            if (SplatCount != splatCount)
            {
                Release();
            }
            
            SplatCount = splatCount;
            
            // Calculate thread blocks for radix sort
            // From the ThirdParty implementation: threadBlocks = ceil(numKeys / PART_SIZE)
            ThreadBlocks = Mathf.CeilToInt((float)splatCount / PART_SIZE);
            
            if (sortIndices != null)
            {
                // Already initialized with same splat count
                return;
            }
            
            // Allocate primary sort buffers
            sortIndices = new ComputeBuffer(splatCount, sizeof(uint));
            sortIndicesAlt = new ComputeBuffer(splatCount, sizeof(uint));
            sortKeys = new ComputeBuffer(splatCount, sizeof(float));
            sortKeysAlt = new ComputeBuffer(splatCount, sizeof(float));
            
            // Allocate view data buffer
            viewData = new ComputeBuffer(splatCount, sizeof(float) * 4);
            
            // Allocate histogram buffers for radix sort
            // Global histogram: RADIX * RADIX_PASSES entries
            globalHistogram = new ComputeBuffer(RADIX * RADIX_PASSES, sizeof(uint));
            
            // Pass histogram: RADIX * threadBlocks entries
            // Need to ensure minimum size
            int passHistSize = Mathf.Max(RADIX * ThreadBlocks, RADIX);
            passHistogram = new ComputeBuffer(passHistSize, sizeof(uint));
        }
        
        /// <summary>
        /// Release all GPU resources.
        /// </summary>
        public void Release()
        {
            ReleaseBuffer(ref sortIndices);
            ReleaseBuffer(ref sortIndicesAlt);
            ReleaseBuffer(ref sortKeys);
            ReleaseBuffer(ref sortKeysAlt);
            ReleaseBuffer(ref viewData);
            ReleaseBuffer(ref globalHistogram);
            ReleaseBuffer(ref passHistogram);
            ReleaseBuffer(ref sortParams);
            
            SplatCount = 0;
        }
        
        private void ReleaseBuffer(ref ComputeBuffer buffer)
        {
            if (buffer != null)
            {
                buffer.Release();
                buffer = null;
            }
        }

        /// <summary>
        /// Allows use with using/IDisposable; calls Release().
        /// </summary>
        public void Dispose()
        {
            Release();
        }


        /// <summary>
        /// Bind sorting input buffers to a compute shader via CommandBuffer.
        /// </summary>
        public void BindSortInputs(UnityEngine.Rendering.ComputeCommandBuffer cmd, ComputeShader shader, int kernel)
        {
            cmd.SetComputeBufferParam(shader, kernel, "b_sortPayload", sortIndices);
            cmd.SetComputeBufferParam(shader, kernel, "b_sort", sortKeys);
            cmd.SetComputeBufferParam(shader, kernel, "_ViewData", viewData);
        }

        /// <summary>
        /// Bind sorting output buffers to a compute shader via CommandBuffer.
        /// </summary>
        public void BindSortOutputs(UnityEngine.Rendering.CommandBuffer cmd, ComputeShader shader, int kernel)
        {
            cmd.SetComputeBufferParam(shader, kernel, "b_altPayload", sortIndicesAlt);
            cmd.SetComputeBufferParam(shader, kernel, "b_alt", sortKeysAlt);
        }
        
        /// <summary>
        /// Bind all radix sort buffers following the ThirdParty convention via CommandBuffer.
        /// </summary>
        public void BindRadixSortBuffers(UnityEngine.Rendering.ComputeCommandBuffer cmd, ComputeShader shader, int kernel, bool useAltAsSource)
        {
            // The ThirdParty radix sort uses:
            // b_sort / b_alt for keys
            // b_sortPayload / b_altPayload for indices
            // b_globalHist for global histogram
            // b_passHist for pass histogram
            
            if (useAltAsSource)
            {
                cmd.SetComputeBufferParam(shader, kernel, "b_sort", sortKeysAlt);
                cmd.SetComputeBufferParam(shader, kernel, "b_alt", sortKeys);
                cmd.SetComputeBufferParam(shader, kernel, "b_sortPayload", sortIndicesAlt);
                cmd.SetComputeBufferParam(shader, kernel, "b_altPayload", sortIndices);
            }
            else
            {
                cmd.SetComputeBufferParam(shader, kernel, "b_sort", sortKeys);
                cmd.SetComputeBufferParam(shader, kernel, "b_alt", sortKeysAlt);
                cmd.SetComputeBufferParam(shader, kernel, "b_sortPayload", sortIndices);
                cmd.SetComputeBufferParam(shader, kernel, "b_altPayload", sortIndicesAlt);
            }
            
            cmd.SetComputeBufferParam(shader, kernel, "b_globalHist", globalHistogram);
            cmd.SetComputeBufferParam(shader, kernel, "b_passHist", passHistogram);
        }
        
        /// <summary>
        /// Set common uniforms for sorting shaders.
        /// </summary>
        public void SetSortUniforms(ComputeShader shader)
        {
            shader.SetInt("_SplatCount", SplatCount);
            shader.SetInt("_TileCountX", TileCountX);
            shader.SetInt("_TileCountY", TileCountY);
            shader.SetInt("_TileSize", TileSize);
        }
        
        /// <summary>
        /// Get the buffer containing sorted indices (after sorting completes).
        /// The result alternates between sortIndices and sortIndicesAlt depending
        /// on the number of radix passes (4 passes = even, so result is in original buffer).
        /// </summary>
        public ComputeBuffer GetSortedIndices()
        {
            // After 4 radix passes, the result is in the original buffer
            // because each pass swaps source and destination
            return sortIndices;
        }
        
        /// <summary>
        /// Get the buffer containing sort keys (for debugging).
        /// </summary>
        public ComputeBuffer GetSortKeys()
        {
            return sortKeys;
        }
        
        /// <summary>
        /// Get the view data buffer (for debugging).
        /// </summary>
        public ComputeBuffer GetViewData()
        {
            return viewData;
        }
    }
}
