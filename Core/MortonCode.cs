using UnityEngine;

namespace GaussianSplatting
{
    /// <summary>
    /// Provides utilities for computing Morton codes (Z-order curve indices) from 3D positions. Map 3D points to 1D indices while preserving 
    /// spatial locality. This is used to spatially sort splats, enabling efficient frustum culling at the chunk level.
    /// </summary>
    public static class MortonCode
    {
        /// <summary>
        /// Computes a 30 bit Morton code for a position within a normalized bounding box.
        /// The position is quantized to 10 bits per axis (1024 levels).
        /// </summary>
        /// <param name="position">The 3D position to encode.</param>
        /// <param name="boundsMin">Minimum corner of the bounding box.</param>
        /// <param name="boundsSize">Size of the bounding box.</param>
        /// <returns>A 32 bit Morton code (only lower 30 bits used).</returns>
        public static uint Encode(Vector3 position, Vector3 boundsMin, Vector3 boundsSize)
        {
            // Normalize position to [0,1] within bounds. Avoid division by zero for degenerate bounds
            float nx = boundsSize.x > 0f ? (position.x - boundsMin.x) / boundsSize.x : 0f;
            float ny = boundsSize.y > 0f ? (position.y - boundsMin.y) / boundsSize.y : 0f;
            float nz = boundsSize.z > 0f ? (position.z - boundsMin.z) / boundsSize.z : 0f;

            // Clamp to [0,1] to handle floating point imprecision
            nx = Mathf.Clamp01(nx);
            ny = Mathf.Clamp01(ny);
            nz = Mathf.Clamp01(nz);

            // Quantize to 10 bits (0 to 1023)
            uint qx = (uint)(nx * 1023f);
            uint qy = (uint)(ny * 1023f);
            uint qz = (uint)(nz * 1023f);

            // Interleave the bits to form Morton code
            return Interleave3(qx, qy, qz);
        }

        /// <summary>
        /// Spreads out the lower 10 bits of a value across a 30 bit result, placing each bit in every third position.
        /// </summary>
        private static uint SpreadBits3(uint x)
        {
            // Mask to lower 10 bits
            x &= 0x000003FFu;

            // Spread bits using the standard Morton bit interleaving technique
            x = (x | (x << 16)) & 0x030000FFu;
            x = (x | (x << 8)) & 0x0300F00Fu;
            x = (x | (x << 4)) & 0x030C30C3u;
            x = (x | (x << 2)) & 0x09249249u;

            return x;
        }

        /// <summary> Interleaves three 10 bit values into a 30 bit Morton code. </summary>
        private static uint Interleave3(uint x, uint y, uint z)
        {
            return SpreadBits3(x) | (SpreadBits3(y) << 1) | (SpreadBits3(z) << 2);
        }

        /// <summary> Computes Morton codes for an array of positions and returns sorted indices. </summary>
        /// <param name="positions">Array of positions to sort.</param>
        /// <param name="bounds">Bounding box containing all positions.</param>
        /// <returns>Array of indices that would sort the positions by Morton code.</returns>
        public static int[] ComputeSortedIndices(Vector3[] positions, Bounds bounds)
        {
            int count = positions.Length;
            if (count == 0)
            {
                return System.Array.Empty<int>();
            }

            Vector3 boundsMin = bounds.min;
            Vector3 boundsSize = bounds.size;

            // Compute Morton codes for all positions
            MortonSortEntry[] entries = new MortonSortEntry[count];
            for (int i = 0; i < count; i++)
            {
                entries[i] = new MortonSortEntry
                {
                    originalIndex = i,
                    mortonCode = Encode(positions[i], boundsMin, boundsSize)
                };
            }

            // Sort by Morton code (stable sort preserving original order for equal codes)
            System.Array.Sort(entries, (a, b) => a.mortonCode.CompareTo(b.mortonCode));

            // Extract sorted indices
            int[] sortedIndices = new int[count];
            for (int i = 0; i < count; i++)
            {
                sortedIndices[i] = entries[i].originalIndex;
            }

            return sortedIndices;
        }

        private struct MortonSortEntry
        {
            public int originalIndex;
            public uint mortonCode;
        }
    }
}
