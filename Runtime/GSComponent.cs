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
        public int maxSplats = 1000000000;  //1 billion by default
        
        [Header("Rendering")]
        [Tooltip("Color space conversion mode for splat colors.\nAuto: uses Unity's current color space.\nForceLinear: always convert gamma→linear.\nForceGamma: no conversion (keep gamma).")]
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
        /// Uses metadata to decide what buffers are needed (no data loading), then loads data from external 
        /// .bytes files or inline byte[] only when uploading to GPU.
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
            bool useSHRest = gsAsset.SHRestCount > 0;

            bool needsRebuild = positionBuffer == null || rotationBuffer == null || scaleBuffer == null ||
                                positionBuffer.count != splatCount ||
                                rotationBuffer.count != splatCount ||
                                scaleBuffer.count != splatCount ||
                                shBuffer == null || shBuffer.count != splatCount ||
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
            shBuffer = new ComputeBuffer(splatCount, sizeof(float) * 4);
            if (useSHRest)
            {
                shRestBuffer = new ComputeBuffer(splatCount, sizeof(float) * gsAsset.SHRestCount);
            }

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

            if (useSHRest && shRestBuffer != null)
            {
                byte[] shRData = gsAsset.SHRestData;
                if (shRData != null && shRData.Length >= splatCount * gsAsset.SHRestCount * 4)
                    shRestBuffer.SetData(shRData);
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

    }
}
