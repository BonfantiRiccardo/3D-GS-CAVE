using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    // This class holds the data imported from a PLY file
    public class GSImportData
    {
        public Vector3[] Positions;
        public quaternion[] Rotations;
        public Vector3[] Scales;
        public Vector3[] SH;
        public float[] SHRest;
        public int SHRestCount;
        public Bounds Bounds;
    }

    // Enum to specify rotation order
    public enum RotationOrder
    {
        XYZW,
        WXYZ
    }

    // This class holds the import options for PLY files
    public class PLYImportOptions
    {
        public bool ImportRotations = true;
        public bool ImportScales = true;
        public bool ImportSH = false;
        public RotationOrder RotationOrder = RotationOrder.WXYZ;
    }
}
