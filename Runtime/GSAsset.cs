using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting
{
    // This ScriptableObject holds the data for a Gaussian Splatting asset
    [CreateAssetMenu(fileName = "GSAsset", menuName = "Scriptable Objects/GSAsset")]
    public class GSAsset : ScriptableObject
    {
        [Header("Gaussian Splatting Settings")]
        [SerializeField, HideInInspector] private Vector3[] positions;      // Splat center positions
        [SerializeField, HideInInspector] private quaternion[] rotations;   // Splat orientations
        [SerializeField, HideInInspector] private Vector3[] scales;         // Splat scale factors
        [SerializeField, HideInInspector] private Vector3[] sh;             // direct color (rgb for sh0)
        [SerializeField, HideInInspector] private float[] shRest;           //other SH coefficients (1,2,3) if used
        [SerializeField, HideInInspector] private int shRestCount;          //number of SH rest bands per splat (0,1,2,3)
        public int splatCount;                                   // Total number of splats in the asset
        public Bounds bounds;                         // Axis-aligned bounding box of the splats

        // Properties to access private fields
        public Vector3[] Positions => positions;
        public quaternion[] Rotations => rotations;
        public Vector3[] Scales => scales;

        public Vector3[] SH => sh;
        public float[] SHRest => shRest;
        public int SHRestCount => shRestCount;

        /**
        * Initializes the GSAsset with the provided data arrays and parameters.
        */
        public void Initialize(
            Vector3[] positions,
            quaternion[] rotations,
            Vector3[] scales,
            Vector3[] sh,
            float[] shRest,
            int shRestCount,
            Bounds bounds)
        {       //Verifies data is not null and assigns to fields
            this.positions = positions ?? System.Array.Empty<Vector3>();
            this.rotations = rotations ?? System.Array.Empty<quaternion>();
            this.scales = scales ?? System.Array.Empty<Vector3>();
            this.sh = sh ?? System.Array.Empty<Vector3>();
            this.shRest = shRest ?? System.Array.Empty<float>();
            this.shRestCount = shRestCount;
            this.splatCount = this.positions.Length;
            this.bounds = bounds;
        }
    }


/**
Possible improvements:
    1. Data Compression: Implement compression techniques for the position, color, rotation, 
        scale, and SH data to reduce memory usage.
        Example: use color palette for SH and point to shared colors instead of storing full color values.
    2. Level of Detail (LOD): Introduce LOD mechanisms to adjust the number of splats based on the 
        camera distance, improving performance.
    3. Streaming: Implement data streaming to load and unload splat data dynamically based on the 
        camera's position and view frustum.
    
*/
}