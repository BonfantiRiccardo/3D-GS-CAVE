using System;
using UnityEngine;

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
        public int maxSplats = 100000000;  //100 million by default
        
        [Header("Model Transform")]
        [Tooltip("Position offset for the splat model")]
        public Vector3 modelPosition = Vector3.zero;
        
        [Tooltip("Rotation of the splat model (Euler angles)")]
        public Vector3 modelRotationEuler = Vector3.zero;
        
        [Tooltip("Scale of the splat model")]
        public Vector3 modelScale = Vector3.one;
        
        /// <summary>
        /// Gets the model rotation as a quaternion
        /// </summary>
        public Quaternion ModelRotation => Quaternion.Euler(modelRotationEuler);
        public int ActiveSplatCount => gsAsset == null ? 0 : Mathf.Min(gsAsset.splatCount, maxSplats);
        public int ShBandsNumber
        {
            get         
            {       // Gets the number of SH bands used in the asset
                if (gsAsset == null || gsAsset.SH == null || gsAsset.SH.Length == 0 || gsAsset.SHRest == null || gsAsset.SHRest.Length == 0 || gsAsset.SHRestCount == 0)
                {
                    return 0;
                }
                // rest count = 3 * (bands^2 -1) because we store 3 color channels per coefficient and exclude the first band stored in sh (direct color)
                return (int)Math.Sqrt( (gsAsset.SHRestCount / 3) + 1);
            }
        }
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

            if (gsAsset.Positions == null || gsAsset.Rotations == null || gsAsset.Scales == null) {
                Debug.LogWarning("GSComponent: GSAsset data arrays are null.");
                return;
            }
            if (gsAsset.Positions.Length < splatCount ||
                gsAsset.Rotations.Length < splatCount ||
                gsAsset.Scales.Length < splatCount)
            {
                Debug.LogWarning("GSComponent: GSAsset data arrays are smaller than the splat count.");
                return;
            }

            // Check if SH data exists (DC term for colors + opacity)
            bool useSH = gsAsset.SH != null && gsAsset.SH.Length >= splatCount;
            // Check if higher SH bands exist
            bool useSHRest = ShBandsNumber > 1 && gsAsset.SHRest != null && gsAsset.SHRestCount > 0 &&
                             gsAsset.SHRest.Length >= splatCount * gsAsset.SHRestCount;

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

            positionBuffer.SetData(gsAsset.Positions, 0, 0, splatCount);
            rotationBuffer.SetData(gsAsset.Rotations, 0, 0, splatCount);
            scaleBuffer.SetData(gsAsset.Scales, 0, 0, splatCount);
            if (useSH && shBuffer != null)
            {
                shBuffer.SetData(gsAsset.SH, 0, 0, splatCount);
            }
            if (useSHRest && shRestBuffer != null)
            {
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
        public void InitializeSortingResources(int screenWidth, int screenHeight)
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

            sortingResources.Initialize(ActiveSplatCount, screenWidth, screenHeight);
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

            // Bind sorting indices buffer if available (for sorted rendering)
            if (sortingResources != null && sortingResources.IsInitialized)
            {
                material.SetBuffer("_SortedIndices", sortingResources.GetSortedIndices());
                material.SetInt("_UseSortedIndices", 1);
            }
            else
            {
                material.SetInt("_UseSortedIndices", 0);
            }
        }

        /// <summary>
        /// Binds splat data buffers to a compute shader for sorting/processing.
        /// </summary>
        public void BindToCompute(ComputeShader shader, int kernel)
        {
            if (positionBuffer != null)
                shader.SetBuffer(kernel, "_Positions", positionBuffer);
            if (rotationBuffer != null)
                shader.SetBuffer(kernel, "_Rotations", rotationBuffer);
            if (scaleBuffer != null)
                shader.SetBuffer(kernel, "_Scales", scaleBuffer);
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
    }
}