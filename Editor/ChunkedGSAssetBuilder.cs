using System;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    /// <summary>
    /// Creates ChunkedGSAsset from imported data. It spatially sorts the data using Morton codes,
    /// partitions into chunks, and writes external .bytes files with the reordered data.
    /// </summary>
    public static class ChunkedGSAssetBuilder
    {
        /// <summary>
        /// Writes chunked external .bytes data files for a ChunkedGSAsset.
        /// Data should already be spatially sorted. Uses streaming writes to handle 
        /// very large files without allocating massive intermediate byte arrays.
        /// </summary>
        public static void WriteExternalData(GSImportData data, string absoluteDirectoryPath)
        {
            Directory.CreateDirectory(absoluteDirectoryPath);

            // Write position data with streaming to avoid large allocations
            WriteTypedArrayStreaming(
                Path.Combine(absoluteDirectoryPath, ChunkedGSAsset.PositionsFileName),
                data.Positions,
                ChunkedGSAsset.PositionStride);

            WriteTypedArrayStreaming(
                Path.Combine(absoluteDirectoryPath, ChunkedGSAsset.RotationsFileName),
                data.Rotations,
                ChunkedGSAsset.RotationStride);

            WriteTypedArrayStreaming(
                Path.Combine(absoluteDirectoryPath, ChunkedGSAsset.ScalesFileName),
                data.Scales,
                ChunkedGSAsset.ScaleStride);

            WriteTypedArrayStreaming(
                Path.Combine(absoluteDirectoryPath, ChunkedGSAsset.SHFileName),
                data.SH,
                ChunkedGSAsset.SHStride);

            // Write SH rest data (flat float array, same streaming path)
            WriteTypedArrayStreaming(
                Path.Combine(absoluteDirectoryPath, ChunkedGSAsset.SHRestFileName),
                data.SHRest,
                sizeof(float));
        }

        /// <summary>
        /// Builds a ChunkedGSAsset with the given chunk metadata.
        /// Data files must already be written via WriteExternalData.
        /// </summary>
        public static ChunkedGSAsset BuildAsset(
            GSImportData data,
            ChunkInfo[] chunks,
            int chunkSize,
            string relativeDataPath)
        {
            var asset = ScriptableObject.CreateInstance<ChunkedGSAsset>();

            int splatCount = data.Positions?.Length ?? 0;

            asset.Initialize(
                totalSplatCount: splatCount,
                chunkSize: chunkSize,
                shRestCount: data.SHRestCount,
                globalBounds: data.Bounds,
                chunks: chunks,
                externalDataPath: relativeDataPath);

            return asset;
        }

        /// <summary>
        /// Complete pipeline: sort, chunk, write data, and build asset.
        /// </summary>
        public static (ChunkedGSAsset asset, ChunkInfo[] chunks) ProcessAndBuild(
            GSImportData data,
            string absoluteDataPath,
            string relativeDataPath,
            int chunkSize = SpatialSorter.DefaultChunkSize,
            IProgress<float> progress = null)
        {
            progress?.Report(0f);

            // Step 1: Sort by Morton code and partition into chunks
            // This modifies data arrays in place
            progress?.Report(0.1f);
            ChunkInfo[] chunks = SpatialSorter.SortAndChunk(data, chunkSize);

            // Step 2: Write reordered data to external files
            progress?.Report(0.3f);
            WriteExternalData(data, absoluteDataPath);

            // Step 3: Build the asset
            progress?.Report(0.9f);
            ChunkedGSAsset asset = BuildAsset(data, chunks, chunkSize, relativeDataPath);

            progress?.Report(1f);
            return (asset, chunks);
        }

        /// <summary>
        /// Writes a typed struct array to a binary file using streaming to avoid
        /// massive intermediate allocations. Writes in chunks of 16MB at a time.
        /// Uses long arithmetic for byte offsets to support very large files.
        /// </summary>
        private static void WriteTypedArrayStreaming<T>(string filePath, T[] data, int expectedStride) where T : struct
        {
            if (data == null || data.Length == 0)
            {
                File.WriteAllBytes(filePath, Array.Empty<byte>());
                return;
            }

            int elementSize = Marshal.SizeOf<T>();
            if (elementSize != expectedStride)
            {
                Debug.LogWarning($"WriteTypedArrayStreaming: Element size {elementSize} != expected stride {expectedStride}");
            }

            const int maxBytesPerChunk = 16 * 1024 * 1024;
            int elementsPerChunk = maxBytesPerChunk / elementSize;
            if (elementsPerChunk < 1) elementsPerChunk = 1;

            using (FileStream fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, 65536))
            {
                int offset = 0;
                while (offset < data.Length)
                {
                    int countThisChunk = Math.Min(elementsPerChunk, data.Length - offset);
                    int byteSizeThisChunk = countThisChunk * elementSize;

                    byte[] buffer = new byte[byteSizeThisChunk];

                    GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
                    try
                    {
                        long byteOffset = (long)offset * elementSize;
                        IntPtr srcPtr = new IntPtr(handle.AddrOfPinnedObject().ToInt64() + byteOffset);
                        Marshal.Copy(srcPtr, buffer, 0, byteSizeThisChunk);
                    }
                    finally
                    {
                        handle.Free();
                    }

                    fs.Write(buffer, 0, byteSizeThisChunk);
                    offset += countThisChunk;
                }
            }
        }
    }
}
