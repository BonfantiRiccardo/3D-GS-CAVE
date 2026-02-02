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

    /// <summary>
    /// Enum to specify rotation order
    /// </summary>
    public enum RotationOrder
    {
        XYZW,
        WXYZ
    }

    /// <summary>
    /// This class holds the import options for PLY files
    /// </summary>
    public class PLYImportOptions
    {
        public bool ImportRotations = true;
        public bool ImportScales = true;
        public bool ImportSH = false;
        public RotationOrder RotationOrder = RotationOrder.WXYZ;
    }
}
