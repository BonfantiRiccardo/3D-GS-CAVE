using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    /// <summary>
    /// SH import quality levels controlling how many spherical harmonic
    /// coefficients are imported from the PLY file.
    /// Concept inspired by aras-p/UnityGaussianSplatting.
    /// </summary>
    public enum SHImportQuality
    {
        Full,               //Import all SH rest coefficients (all available bands).
        Reduced,            // Import only SH Band 1 (9 coefficients: 3 per channel).
        None                // No SH rest coefficients. DC color and opacity only.>

    }

    /// <summary>
    /// Coordinate axis flip mode for converting between coordinate system handedness.
    /// </summary>
    public enum CoordinateFlip
    {
        None,           // No axis flip — import coordinates as-is.
        FlipZ,          //Negate the Z axis (common for right-hand to left-hand conversion).
        FlipX           //Negate the X axis.

    }

    /// <summary>
    /// GSImportData holds the data imported from a PLY file.
    /// </summary>
    public class GSImportData
    {
        public Vector3[] Positions;
        public quaternion[] Rotations;
        public Vector3[] Scales;
        public Vector4[] SH;  // SH DC coefficients (xyz) + opacity (w)
        public float[] SHRest;
        public int SHRestCount;
        public Bounds Bounds;

        // Metadata. What was found in the file and how it was imported
        public bool HasRotations;
        public bool HasScales;
        public bool HasColors;
        public bool HasOpacity;
        public bool HasSHRest;
        public SHImportQuality ImportedSHQuality;
        public CoordinateFlip AppliedFlip;
    }
}
