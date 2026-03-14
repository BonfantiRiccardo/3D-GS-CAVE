/* External data loading through .bytes files inspired by aras-p/UnityGaussianSplatting */
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
        public int maxSplats = 20000000;  //20 million by default
        
        [Header("Rendering")]
        [Tooltip("Color space conversion mode for splat colors.\nAuto: uses Unity's current color space.\nForceLinear: always convert gamma->linear.\nForceGamma: no conversion (keep gamma).")]
        public ColorSpaceMode colorSpaceMode = ColorSpaceMode.Auto;

        [Range(0.1f, 3.0f)]
        [Tooltip("Global scale multiplier for splat size.")]
        public float splatScale = 1.0f;

        [Header("Debugging")]
        [Tooltip("Render splat centers as small points instead of full Gaussians.")]
        public bool showSplatCenters = false;

        [Range(1.0f, 20.0f)]
        [Tooltip("Size of the debug center points in pixels.")]
        public float centerPointSize = 1.0f;

        /// <summary>
        /// Color space conversion mode.
        /// </summary>
        public enum ColorSpaceMode
        {
            Auto,           // Use Unity's project color space setting
            ForceLinear,    // Always convert gamma to linear
            ForceGamma      // Never convert (keep gamma-space colors)
        }
                
        // Gets the model position, rotation and scale from the GameObject's Transform
        public Vector3 modelPosition => transform.position;
        public Quaternion ModelRotation => transform.rotation;
        public Vector3 modelScale => transform.lossyScale;
        
        public int ActiveSplatCount => gsAsset == null ? 0 : Mathf.Min(gsAsset.splatCount, maxSplats);

        /// <summary>
        /// Number of SH bands in the asset. Computed from SHRestCount metadata to avoid triggering lazy data loading from external .bytes files.
        /// rest count = 3 * (bands^2 - 1), so bands = sqrt(restCount/3 + 1).
        /// </summary>
        public int ShBandsNumber
        {
            get
            {
                if (gsAsset == null || gsAsset.SHRestCount == 0)
                    return 0;
                return (int)Math.Sqrt((gsAsset.SHRestCount / 3) + 1);
            }
        }

        /// <summary>
        /// True when all required GPU buffers are allocated. Uses metadata checks only
        /// to avoid triggering data loading from external files.
        /// </summary>
        public bool HasBuffers => positionBuffer != null && rotationBuffer != null && scaleBuffer != null && shBuffer != null;


        private ComputeBuffer positionBuffer;
        private ComputeBuffer rotationBuffer;
        private ComputeBuffer scaleBuffer;
        private ComputeBuffer shBuffer;
        private ComputeBuffer shRestBuffer0;
        private ComputeBuffer shRestBuffer1;
        private ComputeBuffer shRestBuffer2;

        private GSSortingResources sortingResources;

        
        // Public accessors for compute buffers (needed by sorting pipeline)
        public ComputeBuffer PositionBuffer => positionBuffer;
        public ComputeBuffer RotationBuffer => rotationBuffer;
        public ComputeBuffer ScaleBuffer => scaleBuffer;
        public ComputeBuffer SHBuffer => shBuffer;
        public ComputeBuffer SHRestBuffer0 => shRestBuffer0;
        public ComputeBuffer SHRestBuffer1 => shRestBuffer1;
        public ComputeBuffer SHRestBuffer2 => shRestBuffer2;
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
        /// Uses metadata to decide what buffers are needed, and loads data from external .bytes files to fill the buffers. 
        /// Called on enable and when validating in the editor.
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

            // Use metadata to determine what data exists (no file loading)
            int shBands = ShBandsNumber;

            bool needsRebuild = positionBuffer == null || rotationBuffer == null || scaleBuffer == null ||
                                positionBuffer.count != splatCount ||
                                rotationBuffer.count != splatCount ||
                                scaleBuffer.count != splatCount ||
                                shBuffer == null || shBuffer.count != splatCount ||
                                (shBands >= 1 && (shRestBuffer0 == null || shRestBuffer0.count != splatCount * ChunkedGSAsset.SHBand1Count));

            if (!needsRebuild) 
            {
                Debug.Log("GSComponent: Compute buffers are already up to date.");
                return;
            }

            ReleaseBuffers();

            positionBuffer = new ComputeBuffer(splatCount, sizeof(float) * 3);
            rotationBuffer = new ComputeBuffer(splatCount, sizeof(float) * 4);
            scaleBuffer = new ComputeBuffer(splatCount, sizeof(float) * 3);
            shBuffer = new ComputeBuffer(splatCount, sizeof(float) * 4);
            // Always allocate all 3 per-band buffers (1-element dummy when unused) so the compute shader binding is always satisfied.
            // Use float stride (4) to match StructuredBuffer<float> / RWStructuredBuffer<float> in shaders.
            shRestBuffer0 = new ComputeBuffer(shBands >= 1 ? splatCount * ChunkedGSAsset.SHBand1Count : 1, sizeof(float));
            shRestBuffer1 = new ComputeBuffer(shBands >= 2 ? splatCount * ChunkedGSAsset.SHBand2Count : 1, sizeof(float));
            shRestBuffer2 = new ComputeBuffer(shBands >= 3 ? splatCount * ChunkedGSAsset.SHBand3Count : 1, sizeof(float));

            // Load byte[] data from external .bytes files and upload to GPU
            byte[] posData = gsAsset.PositionData;
            byte[] rotData = gsAsset.RotationData;
            byte[] scData = gsAsset.ScaleData;
            byte[] shDCData = gsAsset.SHData;

            if (posData == null || rotData == null || scData == null)
            {
                Debug.LogError("GSComponent: Essential data missing. Check external .bytes files in StreamingAssets.");
                ReleaseBuffers();
                return;
            }

            positionBuffer.SetData(posData);
            rotationBuffer.SetData(rotData);
            scaleBuffer.SetData(scData);

            if (shDCData != null && shDCData.Length >= splatCount * 16)
                shBuffer.SetData(shDCData);

            if (shBands >= 1)
            {
                byte[] shRData = gsAsset.SHRestData;
                if (shRData != null)
                {
                    int totalBytesPerSplat = gsAsset.SHRestCount * sizeof(float);
                    int band1B = ChunkedGSAsset.SHBand1Stride;
                    int band2B = ChunkedGSAsset.SHBand2Stride;
                    int band3B = ChunkedGSAsset.SHBand3Stride;

                    byte[] b1 = new byte[splatCount * band1B];
                    byte[] b2 = shBands >= 2 ? new byte[splatCount * band2B] : null;
                    byte[] b3 = shBands >= 3 ? new byte[splatCount * band3B] : null;

                    for (int i = 0; i < splatCount; i++)
                    {
                        int src = i * totalBytesPerSplat;
                        Buffer.BlockCopy(shRData, src, b1, i * band1B, band1B);
                        if (b2 != null)
                            Buffer.BlockCopy(shRData, src + band1B, b2, i * band2B, band2B);
                        if (b3 != null)
                            Buffer.BlockCopy(shRData, src + band1B + band2B, b3, i * band3B, band3B);
                    }

                    shRestBuffer0.SetData(b1);
                    if (shRestBuffer1 != null) shRestBuffer1.SetData(b2);
                    if (shRestBuffer2 != null) shRestBuffer2.SetData(b3);
                }
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
            if (shRestBuffer0 != null) { shRestBuffer0.Release(); shRestBuffer0 = null; }
            if (shRestBuffer1 != null) { shRestBuffer1.Release(); shRestBuffer1 = null; }
            if (shRestBuffer2 != null) { shRestBuffer2.Release(); shRestBuffer2 = null; }
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

    }
}
