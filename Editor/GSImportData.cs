using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    /// <summary>
    /// GSImportData holds the data imported from a PLY file.
    /// </summary>
    public class GSImportData
    {
        public Vector3[] Positions;
        public quaternion[] Rotations;
        public Vector3[] Scales;
        public Vector4[] SH;  // SH DC coefficients (xyz) + opacity (w) - Vector4 to preserve raw values
        public float[] SHRest;
        public int SHRestCount;
        public Bounds Bounds;
    }
}
