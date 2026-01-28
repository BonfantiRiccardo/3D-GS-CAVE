using UnityEngine;

namespace GaussianSplatting
{   // This MonoBehaviour component manages Gaussian Splatting data and sets up compute buffers for rendering
    public class GSComponent : MonoBehaviour
    {
        [Header("Quality Settings")]
        public GSAsset gsAsset;
        public int maxSplats = 100000000;  //100 million by default

        ComputeBuffer positionBuffer;
        ComputeBuffer rotationBuffer;
        ComputeBuffer scaleBuffer;

        ComputeBuffer shBuffer;
        ComputeBuffer shRestBuffer;

        public int ActiveSplatCount => gsAsset == null ? 0 : Mathf.Min(gsAsset.splatCount, maxSplats);
        public int ShBandsNumber
        {
            get         
            {       // Gets the number of SH bands used in the asset
                if (gsAsset == null || gsAsset.SH == null || gsAsset.SH.Length == 0)
                {
                    return 0;
                }

                return gsAsset.SHRestCount > 0 ? 3 : 1;
            }
        }
        public bool HasBuffers => positionBuffer != null && rotationBuffer != null && scaleBuffer != null && (ShBandsNumber == 0 || shBuffer != null);

        /**
        * This method is called when the script instance is being loaded.
        */
        void OnEnable()
        {
            BuildBuffers();
            GSManager.Register(this);
        }

        /**
        * This method is called when the behaviour becomes disabled or inactive.
        */
        void OnDisable()
        {
            ReleaseBuffers();
            GSManager.Unregister(this);
        }

        /**
        * This method is called when the script is loaded or a value is changed in the inspector (Called in the editor only).
        */
        void OnValidate()
        {
            RebuildBuffers();
        }

        /**
        * Builds the compute buffers for positions, rotation, scale, and SH based on the GSAsset data.
        */
        void BuildBuffers()
        {
            if (gsAsset == null) return;

            int splatCount = Mathf.Min(gsAsset.splatCount, maxSplats);
            if (splatCount <= 0) return;

            if (gsAsset.Positions == null || gsAsset.Rotations == null || gsAsset.Scales == null) return;
            if (gsAsset.Positions.Length < splatCount ||
                gsAsset.Rotations.Length < splatCount ||
                gsAsset.Scales.Length < splatCount)
            {
                return;
            }

            bool useSH = ShBandsNumber > 0 && gsAsset.SH != null && gsAsset.SH.Length >= splatCount;
            bool useSHRest = ShBandsNumber > 1 && gsAsset.SHRest != null && gsAsset.SHRestCount > 0 &&
                             gsAsset.SHRest.Length >= splatCount * gsAsset.SHRestCount;

            bool needsRebuild = positionBuffer == null || rotationBuffer == null || scaleBuffer == null ||
                                positionBuffer.count != splatCount ||
                                rotationBuffer.count != splatCount ||
                                scaleBuffer.count != splatCount ||
                                (useSH && (shBuffer == null || shBuffer.count != splatCount)) ||
                                (useSHRest && (shRestBuffer == null || shRestBuffer.count != splatCount));

            if (!needsRebuild) return;

            ReleaseBuffers();

            positionBuffer = new ComputeBuffer(splatCount, sizeof(float) * 3);
            rotationBuffer = new ComputeBuffer(splatCount, sizeof(float) * 4);
            scaleBuffer = new ComputeBuffer(splatCount, sizeof(float) * 3);
            if (useSH)
            {
                shBuffer = new ComputeBuffer(splatCount, sizeof(float) * 3);
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

        /**
        * Rebuilds the compute buffers by releasing existing ones and creating new ones.
        */
        void RebuildBuffers()
        {
            ReleaseBuffers();
            BuildBuffers();
        }

        /**
        * Releases the compute buffers to free up resources.
        */
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

        /**
        * Binds the compute buffers to the given material for rendering.
        */
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
            if (ShBandsNumber > 0 && shBuffer != null)
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
