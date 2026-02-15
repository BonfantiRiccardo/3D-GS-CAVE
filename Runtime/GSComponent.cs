using System;
using UnityEngine;
        using System.Text;

namespace GaussianSplatting
{   
    /// <summary>
    /// This MonoBehaviour component manages Gaussian Splatting data and sets up compute buffers for rendering.
    /// </summary>
    [ExecuteAlways]
    public class GSComponent : MonoBehaviour
    {
        [Header("Quality Settings")]
        public GSAsset gsAsset;            // Reference to the GSAsset containing splat data
        public int maxSplats = 1000000000;  //1 billion by default
                
        // Gets the model position, rotation and scale from the GameObject's Transform
        public Vector3 modelPosition => transform.position;
        public Quaternion ModelRotation => transform.rotation;
        public Vector3 modelScale => transform.lossyScale;
        
        public int ActiveSplatCount => gsAsset == null ? 0 : Mathf.Min(gsAsset.splatCount, maxSplats);
        /// <summary>
        /// Number of SH bands in the asset. Computed from SHRestCount metadata to avoid
        /// triggering lazy typed-array reconstruction of SH/SHRest byte blobs.
        /// rest count = 3 * (bands^2 - 1), so bands = sqrt(restCount/3 + 1).
        /// </summary>
        public int ShBandsNumber
        {
            get
            {
                if (gsAsset == null || gsAsset.SHRestCount == 0)
                    return 0;
                // Validate that DC (SH) blob exists via byte[] length, not typed array
                if (gsAsset.SHData == null || gsAsset.SHData.Length == 0)
                    return 0;
                // Validate that rest blob exists via byte[] length (4 bytes per float)
                if (gsAsset.SHRestData == null || gsAsset.SHRestData.Length < gsAsset.splatCount * gsAsset.SHRestCount * 4)
                    return 0;
                return (int)Math.Sqrt((gsAsset.SHRestCount / 3) + 1);
            }
        }

        /// <summary>
        /// True when all required GPU buffers are allocated. Uses byte[] metadata checks
        /// to avoid triggering lazy typed-array allocation.
        /// </summary>
        public bool HasBuffers => positionBuffer != null && rotationBuffer != null && scaleBuffer != null && (ShBandsNumber == 0 || shBuffer != null);


        private ComputeBuffer positionBuffer;
        private ComputeBuffer rotationBuffer;
        private ComputeBuffer scaleBuffer;
        private ComputeBuffer shBuffer;
        private ComputeBuffer shRestBuffer;
        // Sorting resources for tile-based rendering
        private GSSortingResources sortingResources;

        
        // Public accessors for compute buffers (needed by sorting pipeline)
        public ComputeBuffer PositionBuffer => positionBuffer;
        public ComputeBuffer RotationBuffer => rotationBuffer;
        public ComputeBuffer ScaleBuffer => scaleBuffer;
        public ComputeBuffer SHBuffer => shBuffer;
        public ComputeBuffer SHRestBuffer => shRestBuffer;
        public GSSortingResources SortingResources => sortingResources;

        /// <summary>
        /// This method is called when the script instance is being loaded.
        /// </summary>
        void OnEnable()
        {
            BuildBuffers();
            GSManager.Register(this);
        }

        /// <summary>
        /// This method is called when the behaviour becomes disabled or inactive.
        /// </summary>
        void OnDisable()
        {
            ReleaseBuffers();
            ReleaseSortingResources();
            GSManager.Unregister(this);
        }

        /// <summary>
        /// This method is called when the MonoBehaviour will be destroyed.
        /// </summary>
        void OnDestroy()
        {
            ReleaseBuffers();
            ReleaseSortingResources();
            GSManager.Unregister(this);
        }

        /// <summary>
        /// This method is called when the script is loaded or a value is changed in the inspector (Called in the editor only).
        /// </summary>
        void OnValidate()
        {
            RebuildBuffers();
            GSManager.Register(this);
        }

        /// <summary>
        /// Builds the compute buffers for positions, rotation, scale, and SH based on the GSAsset data.
        /// Uses direct byte[] upload from GSAsset when possible to avoid reconstructing typed arrays.
        /// </summary>
        void BuildBuffers()
        {
            if (gsAsset == null) 
            {
                Debug.LogWarning("GSComponent: No GSAsset assigned.");
                return;
            }

            int splatCount = Mathf.Min(gsAsset.splatCount, maxSplats);
            if (splatCount <= 0) 
            {
                Debug.LogWarning("GSComponent: GSAsset has no splats.");
                return;
            }

            // Validate using raw byte[] accessors (avoids triggering lazy typed-array reconstruction)
            if (gsAsset.PositionData == null || gsAsset.RotationData == null || gsAsset.ScaleData == null)
            {
                Debug.LogWarning("GSComponent: GSAsset data arrays are null.");
                return;
            }

            // Check if SH data exists (DC term for colors + opacity)
            bool useSH = gsAsset.SHData != null && gsAsset.SHData.Length >= splatCount * 16;
            // Check if higher SH bands exist
            bool useSHRest = gsAsset.SHRestCount > 0 && gsAsset.SHRestData != null &&
                             gsAsset.SHRestData.Length >= splatCount * gsAsset.SHRestCount * 4;

            bool needsRebuild = positionBuffer == null || rotationBuffer == null || scaleBuffer == null ||
                                positionBuffer.count != splatCount ||
                                rotationBuffer.count != splatCount ||
                                scaleBuffer.count != splatCount ||
                                (useSH && (shBuffer == null || shBuffer.count != splatCount)) ||
                                (useSHRest && (shRestBuffer == null || shRestBuffer.count != splatCount));

            if (!needsRebuild) 
            {
                Debug.Log("GSComponent: Compute buffers are already up to date.");
                return;
            }

            ReleaseBuffers();

            positionBuffer = new ComputeBuffer(splatCount, sizeof(float) * 3);
            rotationBuffer = new ComputeBuffer(splatCount, sizeof(float) * 4);
            scaleBuffer = new ComputeBuffer(splatCount, sizeof(float) * 3);
            if (useSH)
            {
                shBuffer = new ComputeBuffer(splatCount, sizeof(float) * 4);
            }
            if (useSHRest)
            {
                shRestBuffer = new ComputeBuffer(splatCount, sizeof(float) * gsAsset.SHRestCount);
            }

            // Upload data to GPU. Use direct byte[] path when uploading the full dataset
            // (avoids allocating typed arrays entirely). Falls back to typed arrays for partial uploads.
            bool fullUpload = splatCount == gsAsset.splatCount;

            if (fullUpload)
            {
                // Zero-copy path: byte[] → GPU. No typed array allocation.
                positionBuffer.SetData(gsAsset.PositionData);
                rotationBuffer.SetData(gsAsset.RotationData);
                scaleBuffer.SetData(gsAsset.ScaleData);
                if (useSH && shBuffer != null)
                    shBuffer.SetData(gsAsset.SHData);
                if (useSHRest && shRestBuffer != null)
                    shRestBuffer.SetData(gsAsset.SHRestData);
            }
            else
            {
                // Partial upload: typed arrays are reconstructed once and cached in GSAsset
                positionBuffer.SetData(gsAsset.Positions, 0, 0, splatCount);
                rotationBuffer.SetData(gsAsset.Rotations, 0, 0, splatCount);
                scaleBuffer.SetData(gsAsset.Scales, 0, 0, splatCount);
                if (useSH && shBuffer != null)
                    shBuffer.SetData(gsAsset.SH, 0, 0, splatCount);
                if (useSHRest && shRestBuffer != null)
                    shRestBuffer.SetData(gsAsset.SHRest, 0, 0, splatCount * gsAsset.SHRestCount);
            }
        }

        /// <summary>
        /// Rebuilds the compute buffers by releasing existing ones and creating new ones.
        /// </summary>
        void RebuildBuffers()
        {
            ReleaseBuffers();
            BuildBuffers();
        }

        /// <summary>
        /// Releases the compute buffers to free up resources.
        /// </summary>
        void ReleaseBuffers()
        {
            if (positionBuffer != null)
            {
                positionBuffer.Release();
                positionBuffer = null;
            }

            if (rotationBuffer != null)
            {
                rotationBuffer.Release();
                rotationBuffer = null;
            }

            if (scaleBuffer != null)
            {
                scaleBuffer.Release();
                scaleBuffer = null;
            }
            if (shBuffer != null)
            {
                shBuffer.Release();
                shBuffer = null;
            }
            if (shRestBuffer != null)
            {
                shRestBuffer.Release();
                shRestBuffer = null;
            }
        }

        /// <summary>
        /// Initialize or update sorting resources for the current screen size.
        /// Called by the render pass before sorting.
        /// </summary>
        public void InitializeSortingResources()
        {
            if (ActiveSplatCount <= 0) 
            {
                Debug.LogWarning("GSComponent: No active splats to initialize sorting resources.");
                return;
            }

            if (sortingResources == null)
            {
                sortingResources = new GSSortingResources();
            }

            sortingResources.Initialize(ActiveSplatCount);
        }

        /// <summary>
        /// Release sorting resources.
        /// </summary>
        void ReleaseSortingResources()
        {
            if (sortingResources != null)
            {
                sortingResources.Dispose();
                sortingResources = null;
            }
        }

        /// <summary>
        /// Binds the compute buffers to the given material for rendering.
        /// </summary>
        public void BindTo(Material material)
        {
            if (positionBuffer != null)
            {
                material.SetBuffer("_Positions", positionBuffer);
            }

            if (rotationBuffer != null)
            {
                material.SetBuffer("_Rotations", rotationBuffer);
            }

            if (scaleBuffer != null)
            {
                material.SetBuffer("_Scales", scaleBuffer);
            }
            // Always bind SH buffer if it exists (contains DC color + opacity)
            // ShBandsNumber only counts higher bands, but we need DC even with 0 bands
            if (shBuffer != null)
            {
                material.SetBuffer("_SH", shBuffer);
            }
            if (ShBandsNumber > 1 && shRestBuffer != null)
            {
                material.SetBuffer("_SHRest", shRestBuffer);
                material.SetInt("_SHRestCount", gsAsset.SHRestCount);
            }

            if (gsAsset != null)
            {
                material.SetInt("_SplatCount", Mathf.Min(gsAsset.splatCount, maxSplats));
            }
        }

        /// <summary>
        /// Binds splat data buffers to a compute shader for sorting/processing via CommandBuffer.
        /// </summary>
        public void BindToCompute(UnityEngine.Rendering.ComputeCommandBuffer cmd, ComputeShader shader, int kernel)
        {
            if (positionBuffer != null)
                cmd.SetComputeBufferParam(shader, kernel, "_Positions", positionBuffer);
            if (rotationBuffer != null)
                cmd.SetComputeBufferParam(shader, kernel, "_Rotations", rotationBuffer);
            if (scaleBuffer != null)
                cmd.SetComputeBufferParam(shader, kernel, "_Scales", scaleBuffer);
        }

        /*
            // Start is called once before the first execution of Update after the MonoBehaviour is created
            void Start()
            {

            }

            // Update is called once per frame
            void Update()
            {

            }
        */


        public void DebugPrintSortedIndices(int count = 10)
        {
            if (sortingResources == null || !sortingResources.IsInitialized)
            {
                Debug.Log("Sorting resources not initialized");
                return;
            }

            var sortedBuffer = sortingResources.GetSortedIndices();
            var keysBuffer = sortingResources.GetSortKeys();
            
            uint[] indices = new uint[Mathf.Min(count, sortedBuffer.count)];
            float[] keys = new float[Mathf.Min(count, keysBuffer.count)];
            
            sortedBuffer.GetData(indices);
            keysBuffer.GetData(keys);

            var cam = Camera.main;  // Use Camera.main instead of Camera.current for more reliable access

            // Also print the actual positions to verify depth ordering
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"First {count} sorted indices (SplatCount={sortingResources.SplatCount}):");
            for (int i = 0; i < indices.Length; i++)
            {
                uint idx = indices[i];
                float key = keys[i];
                if (idx < gsAsset.Positions.Length)
                {
                    Vector3 pos = gsAsset.Positions[idx];
                    if (cam != null)
                    {
                        Vector3 camSpacePos = cam.worldToCameraMatrix.MultiplyPoint(pos);
                        float depth = camSpacePos.z;
                        sb.AppendLine($"  [{i}] Index: {idx}, Key: {key:F4}, Position: {pos}, CamRelDepth: {depth:F2}");
                    }
                    else
                    {
                        sb.AppendLine($"  [{i}] Index: {idx}, Key: {key:F4}, Position: {pos}");
                    }
                }
            }
            Debug.Log(sb.ToString());
        }

        //Debug call to read what is inside the position buffer, 
        // the world positions buffer, and the sorted indices after kernel execution
        public void DebugPrintBuffers()
        {
            if (positionBuffer == null)
            {
                Debug.Log("Position buffer not initialized");
                return;
            }

            int count = Mathf.Min(10, positionBuffer.count);
            Vector3[] positions = new Vector3[count];
            positionBuffer.GetData(positions);

            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"=== DEBUG BUFFER DUMP (count={positionBuffer.count}) ===");
            
            // 1. Positions from GPU buffer
            sb.AppendLine("\n[1] Position Buffer (GPU):");
            for (int i = 0; i < positions.Length; i++)
            {
                sb.AppendLine($"  [{i}] Position: {positions[i]}");
            }

            // 2. Compute world positions (applying model transform)
            sb.AppendLine("\n[2] World Positions (after model transform):");
            for (int i = 0; i < positions.Length; i++)
            {
                Vector3 worldPos = ModelRotation * Vector3.Scale(positions[i], modelScale) + modelPosition;
                sb.AppendLine($"  [{i}] WorldPos: {worldPos}");
            }

            // 3. Camera data
            var cam = Camera.main;
            if (cam != null)
            {
                sb.AppendLine("\n[3] Camera Data:");
                sb.AppendLine($"  Camera Position: {cam.transform.position}");
                sb.AppendLine($"  Camera Forward: {cam.transform.forward}");
                sb.AppendLine($"  Near Plane: {cam.nearClipPlane}");
                sb.AppendLine($"  Far Plane: {cam.farClipPlane}");
                sb.AppendLine($"  View Matrix:\n{cam.worldToCameraMatrix}");
                
                // 4. View-space positions (what the compute shader should compute)
                sb.AppendLine("\n[4] Expected View-Space Positions (CPU calculation):");
                for (int i = 0; i < positions.Length; i++)
                {
                    Vector3 worldPos = ModelRotation * Vector3.Scale(positions[i], modelScale) + modelPosition;
                    Vector3 viewPos = cam.worldToCameraMatrix.MultiplyPoint(worldPos);
                    float expectedKey = ComputeExpectedSortKey(viewPos.z, cam.nearClipPlane, cam.farClipPlane);
                    sb.AppendLine($"  [{i}] ViewPos: {viewPos}, ViewZ: {viewPos.z:F4}, ExpectedKey: {expectedKey:F4}");
                }
            }
            else
            {
                sb.AppendLine("\n[3] Camera: NOT FOUND (Camera.main is null)");
            }

            // 5. Sorting resources data
            if (sortingResources != null && sortingResources.IsInitialized)
            {
                // View data from GPU
                var viewDataBuffer = sortingResources.GetViewData();
                if (viewDataBuffer != null)
                {
                    Vector4[] viewData = new Vector4[count];
                    viewDataBuffer.GetData(viewData);
                    sb.AppendLine("\n[5] ViewData Buffer (GPU - from CalcViewData kernel):");
                    for (int i = 0; i < viewData.Length; i++)
                    {
                        sb.AppendLine($"  [{i}] ViewPos: ({viewData[i].x:F4}, {viewData[i].y:F4}, {viewData[i].z:F4}), Visible: {viewData[i].w}");
                    }
                }

                // Sort keys from GPU
                var keysBuffer = sortingResources.GetSortKeys();
                if (keysBuffer != null)
                {
                    float[] keys = new float[count];
                    keysBuffer.GetData(keys);
                    sb.AppendLine("\n[6] SortKeys Buffer (GPU - from CalcSortKeys kernel):");
                    for (int i = 0; i < keys.Length; i++)
                    {
                        sb.AppendLine($"  [{i}] Key: {keys[i]:F4}");
                    }
                }

                // Sorted indices from GPU
                var sortedBuffer = sortingResources.GetSortedIndices();
                if (sortedBuffer != null)
                {
                    uint[] indices = new uint[count];
                    sortedBuffer.GetData(indices);
                    sb.AppendLine("\n[7] SortedIndices Buffer (GPU - after radix sort):");
                    for (int i = 0; i < indices.Length; i++)
                    {
                        sb.AppendLine($"  [{i}] Index: {indices[i]}");
                    }
                }
            }
            else
            {
                sb.AppendLine("\n[5-7] Sorting resources not initialized");
            }

            Debug.Log(sb.ToString());
        }
        
        /// <summary>
        /// Compute expected sort key using the same formula as the compute shader.
        /// </summary>
        private float ComputeExpectedSortKey(float viewZ, float nearPlane, float farPlane)
        {
            // Simply return viewZ for maximum precision
            // Ascending sort with negative viewZ: far objects (more negative) come first
            return viewZ;
        }

        // Add at the top of the class
        private float lastDebugTime = 0f;
        private const float DEBUG_INTERVAL = 1.0f; // Print every 1 second

        // Add this method
        void Update()
        {
            // Only debug in editor and at intervals
#if UNITY_EDITOR
            if (Time.time - lastDebugTime > DEBUG_INTERVAL)
            {
                lastDebugTime = Time.time;
                DebugPrintBuffers();  // Comprehensive debug output
            }
#endif
        }
    }
}