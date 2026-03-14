using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    /// <summary>
    /// Provides utilities for sorting splat data by spatial locality
    /// using Morton codes, and partitioning the sorted data into chunks.
    /// </summary>
    public static class SpatialSorter
    {
        /// <summary>
        /// Default number of splats per chunk. This balances culling granularity
        /// against chunk management overhead.
        /// </summary>
        public const int DefaultChunkSize = 4096;

        /// <summary>
        /// Reorders GSImportData arrays by Morton code and partitions into chunks.
        /// Modifies the input data in place and returns chunk metadata.
        /// </summary>
        /// <param name="data">Import data to reorder. Arrays will be modified in place.</param>
        /// <param name="chunkSize">Number of splats per chunk.</param>
        /// <returns>Array of ChunkInfo describing each chunk.</returns>
        public static ChunkInfo[] SortAndChunk(GSImportData data, int chunkSize = DefaultChunkSize)
        {
            if (data.Positions == null || data.Positions.Length == 0)
            {
                return Array.Empty<ChunkInfo>();
            }

            if (chunkSize <= 0)
            {
                Debug.LogWarning($"SpatialSorter: invalid chunkSize={chunkSize}. Using {DefaultChunkSize}.");
                chunkSize = DefaultChunkSize;
            }

            int splatCount = data.Positions.Length;

            // Step 1: Compute Morton sorted indices
            int[] sortedIndices = MortonCode.ComputeSortedIndices(data.Positions, data.Bounds);

            // Step 2: Reorder all arrays according to sorted indices
            ReorderArrays(data, sortedIndices);

            // Step 3: Partition into chunks and compute per chunk bounds
            int chunkCount = (splatCount + chunkSize - 1) / chunkSize;
            ChunkInfo[] chunks = new ChunkInfo[chunkCount];

            for (int c = 0; c < chunkCount; c++)
            {
                int startIndex = c * chunkSize;
                int endIndex = Math.Min(startIndex + chunkSize, splatCount);
                int count = endIndex - startIndex;

                // Compute tight bounds for this chunk
                Bounds chunkBounds = ComputeBounds(data.Positions, startIndex, count);

                chunks[c] = new ChunkInfo
                {
                    bounds = chunkBounds,
                    splatCount = count,
                    startIndex = startIndex,
                    dataOffset = (long)startIndex  // Will be multiplied by stride when reading
                };
            }

            return chunks;
        }

        /// <summary>
        /// Reorders all arrays in GSImportData according to the given index mapping.
        /// After reordering, data[i] = originalData[sortedIndices[i]].
        /// </summary>
        private static void ReorderArrays(GSImportData data, int[] sortedIndices)
        {
            int count = sortedIndices.Length;

            // Reorder positions
            data.Positions = ReorderArray(data.Positions, sortedIndices);

            // Reorder rotations
            if (data.Rotations != null && data.Rotations.Length == count)
            {
                data.Rotations = ReorderArray(data.Rotations, sortedIndices);
            }

            // Reorder scales
            if (data.Scales != null && data.Scales.Length == count)
            {
                data.Scales = ReorderArray(data.Scales, sortedIndices);
            }

            // Reorder SH DC (color + opacity)
            if (data.SH != null && data.SH.Length == count)
            {
                data.SH = ReorderArray(data.SH, sortedIndices);
            }

            // Reorder SHRest (higher order SH coefficients)
            // SHRest is a flat array with SHRestCount floats per splat
            if (data.SHRest != null && data.SHRestCount > 0 && data.SHRest.Length == count * data.SHRestCount)
            {
                data.SHRest = ReorderFlatArray(data.SHRest, sortedIndices, data.SHRestCount);
            }
        }

        /// <summary>
        /// Creates a new array with elements reordered according to sortedIndices.
        /// </summary>
        private static T[] ReorderArray<T>(T[] original, int[] sortedIndices)
        {
            if (original == null) return null;

            T[] reordered = new T[sortedIndices.Length];
            for (int i = 0; i < sortedIndices.Length; i++)
            {
                reordered[i] = original[sortedIndices[i]];
            }
            return reordered;
        }

        /// <summary>
        /// Reorders a flat array where each element spans multiple values.
        /// Used for SHRest which has SHRestCount floats per splat.
        /// </summary>
        private static float[] ReorderFlatArray(float[] original, int[] sortedIndices, int elementsPerSplat)
        {
            if (original == null) return null;

            float[] reordered = new float[original.Length];
            for (int i = 0; i < sortedIndices.Length; i++)
            {
                int srcOffset = sortedIndices[i] * elementsPerSplat;
                int dstOffset = i * elementsPerSplat;
                for (int j = 0; j < elementsPerSplat; j++)
                {
                    reordered[dstOffset + j] = original[srcOffset + j];
                }
            }
            return reordered;
        }

        /// <summary>
        /// Computes the axis aligned bounding box for a range of positions.
        /// </summary>
        private static Bounds ComputeBounds(Vector3[] positions, int startIndex, int count)
        {
            if (count <= 0)
            {
                return new Bounds(Vector3.zero, Vector3.zero);
            }

            Vector3 min = positions[startIndex];
            Vector3 max = positions[startIndex];

            int endIndex = startIndex + count;
            for (int i = startIndex + 1; i < endIndex; i++)
            {
                Vector3 p = positions[i];
                min = Vector3.Min(min, p);
                max = Vector3.Max(max, p);
            }

            Vector3 center = (min + max) * 0.5f;
            Vector3 size = max - min;

            return new Bounds(center, size);
        }
    }
}
