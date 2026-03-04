using System;
using System.IO;
using UnityEngine;

namespace GaussianSplatting
{
    /// <summary>
    /// ChunkedGSAsset stores spatially chunked Gaussian splat data. It supports streaming individual
    /// chunks based on visibility, enabling rendering of very large scenes that exceed GPU memory.
    /// 
    /// Data is stored externally in the Assets/StreamingAssets folder:
    ///   Assets/StreamingAssets/{assetName}/chunks.json      Chunk metadata (bounds, counts, offsets)
    ///   /positions.bytes  Position data for all chunks contiguously
    ///   /rotations.bytes  Rotation data for all chunks contiguously
    ///   /scales.bytes     Scale data for all chunks contiguously
    ///   /sh.bytes         SH DC data for all chunks contiguously
    ///   /shrest.bytes     SH rest data for all chunks contiguously
    /// 
    /// Chunks are spatially sorted using Morton codes so that each chunk contains
    /// nearby splats. This enables efficient frustum culling at the chunk level.
    /// </summary>
    [CreateAssetMenu(fileName = "ChunkedGSAsset", menuName = "Scriptable Objects/ChunkedGSAsset")]
    public class ChunkedGSAsset : ScriptableObject
    {
        [Header("Global Metadata")]
        [Tooltip("Total number of splats across all chunks")]
        public int totalSplatCount;

        [Tooltip("Global bounding box encompassing all splats")]
        public Bounds globalBounds;

        [Header("Chunk Configuration")]
        [Tooltip("Number of splats per chunk (except possibly the last chunk)")]
        public int chunkSize = 4096;

        [Tooltip("Total number of chunks")]
        public int chunkCount;

        [Header("SH Configuration")]
        [SerializeField, HideInInspector] 
        private int shRestCount;

        [Header("Data Location")]
        [SerializeField, HideInInspector] 
        private string externalDataPath;

        // Chunk metadata array (serialized in the asset for fast access)
        [SerializeField, HideInInspector]
        private ChunkInfo[] chunks;

        // File names for external data
        public const string ChunkMetadataFileName = "chunks.json";
        public const string PositionsFileName = "positions.bytes";
        public const string RotationsFileName = "rotations.bytes";
        public const string ScalesFileName = "scales.bytes";
        public const string SHFileName = "sh.bytes";
        public const string SHRestFileName = "shrest.bytes";

        // Strides for each data type (bytes per splat)
        public const int PositionStride = 12;   // float3: 3 * 4 bytes
        public const int RotationStride = 16;   // float4: 4 * 4 bytes
        public const int ScaleStride = 12;      // float3: 3 * 4 bytes
        public const int SHStride = 16;         // float4: 4 * 4 bytes

        // Public accessors
        public int SHRestCount => shRestCount;
        public string ExternalDataPath => externalDataPath;
        public ChunkInfo[] Chunks => chunks;
        public int SHRestStride => shRestCount * sizeof(float);

        /// <summary>
        /// Number of SH bands. Computed from SHRestCount.
        /// SHRestCount = 3 * (bands^2 - 1), so bands = sqrt(SHRestCount/3 + 1).
        /// </summary>
        public int SHBands
        {
            get
            {
                if (shRestCount == 0) return 0;
                return (int)Math.Sqrt((shRestCount / 3) + 1);
            }
        }

        /// <summary>
        /// Initializes the asset with chunk data. Called by the importer after
        /// data has been written to external files.
        /// </summary>
        public void Initialize(
            int totalSplatCount,
            int chunkSize,
            int shRestCount,
            Bounds globalBounds,
            ChunkInfo[] chunks,
            string externalDataPath)
        {
            this.totalSplatCount = totalSplatCount;
            this.chunkSize = chunkSize;
            this.shRestCount = shRestCount;
            this.globalBounds = globalBounds;
            this.chunks = chunks;
            this.chunkCount = chunks?.Length ?? 0;
            this.externalDataPath = externalDataPath;
        }

        /// <summary>
        /// Resolves the external data path to an absolute file path for a given filename.
        /// </summary>
        public string ResolveFilePath(string fileName)
        {
            if (string.IsNullOrEmpty(externalDataPath)) return null;
            string projectRoot = GetProjectRoot();
            return Path.Combine(projectRoot, externalDataPath, fileName);
        }

        /// <summary>
        /// Reads position data for a specific chunk from the external file.
        /// </summary>
        /// <param name="chunkIndex">Index of the chunk to read.</param>
        /// <returns>Byte array containing position data for the chunk.</returns>
        public byte[] ReadChunkPositions(int chunkIndex)
        {
            return ReadChunkData(chunkIndex, PositionsFileName, PositionStride);
        }

        /// <summary>
        /// Reads rotation data for a specific chunk from the external file.
        /// </summary>
        public byte[] ReadChunkRotations(int chunkIndex)
        {
            return ReadChunkData(chunkIndex, RotationsFileName, RotationStride);
        }

        /// <summary>
        /// Reads scale data for a specific chunk from the external file.
        /// </summary>
        public byte[] ReadChunkScales(int chunkIndex)
        {
            return ReadChunkData(chunkIndex, ScalesFileName, ScaleStride);
        }

        /// <summary>
        /// Reads SH DC data for a specific chunk from the external file.
        /// </summary>
        public byte[] ReadChunkSH(int chunkIndex)
        {
            return ReadChunkData(chunkIndex, SHFileName, SHStride);
        }

        /// <summary>
        /// Reads SH rest data for a specific chunk from the external file.
        /// </summary>
        public byte[] ReadChunkSHRest(int chunkIndex)
        {
            if (shRestCount == 0) return null;
            return ReadChunkData(chunkIndex, SHRestFileName, SHRestStride);
        }

        /// <summary>
        /// Reads data for multiple chunks at once (for batch loading).
        /// Returns combined byte array for all specified chunks.
        /// </summary>
        public byte[] ReadChunksPositions(int[] chunkIndices)
        {
            return ReadMultipleChunksData(chunkIndices, PositionsFileName, PositionStride);
        }

        public byte[] ReadChunksRotations(int[] chunkIndices)
        {
            return ReadMultipleChunksData(chunkIndices, RotationsFileName, RotationStride);
        }

        public byte[] ReadChunksScales(int[] chunkIndices)
        {
            return ReadMultipleChunksData(chunkIndices, ScalesFileName, ScaleStride);
        }

        public byte[] ReadChunksSH(int[] chunkIndices)
        {
            return ReadMultipleChunksData(chunkIndices, SHFileName, SHStride);
        }

        public byte[] ReadChunksSHRest(int[] chunkIndices)
        {
            if (shRestCount == 0) return null;
            return ReadMultipleChunksData(chunkIndices, SHRestFileName, SHRestStride);
        }

        /// <summary>
        /// Internal helper to read chunk data from an external file.
        /// </summary>
        private byte[] ReadChunkData(int chunkIndex, string fileName, int stride)
        {
            if (chunks == null || chunkIndex < 0 || chunkIndex >= chunks.Length)
            {
                Debug.LogError($"ChunkedGSAsset: Invalid chunk index {chunkIndex}");
                return null;
            }

            string filePath = ResolveFilePath(fileName);
            if (filePath == null || !File.Exists(filePath))
            {
                Debug.LogError($"ChunkedGSAsset: Data file not found: {filePath}");
                return null;
            }

            ChunkInfo chunk = chunks[chunkIndex];
            long byteOffset = chunk.startIndex * (long)stride;
            int byteCount = chunk.splatCount * stride;

            byte[] data = new byte[byteCount];

            using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                fs.Seek(byteOffset, SeekOrigin.Begin);
                int bytesRead = fs.Read(data, 0, byteCount);
                if (bytesRead != byteCount)
                {
                    Debug.LogWarning($"ChunkedGSAsset: Expected {byteCount} bytes but read {bytesRead}");
                }
            }

            return data;
        }

        /// <summary>
        /// Reads data for multiple chunks and concatenates them.
        /// </summary>
        private byte[] ReadMultipleChunksData(int[] chunkIndices, string fileName, int stride)
        {
            if (chunks == null || chunkIndices == null || chunkIndices.Length == 0)
            {
                return null;
            }

            string filePath = ResolveFilePath(fileName);
            if (filePath == null || !File.Exists(filePath))
            {
                Debug.LogError($"ChunkedGSAsset: Data file not found: {filePath}");
                return null;
            }

            // Calculate total byte count
            int totalSplats = 0;
            foreach (int idx in chunkIndices)
            {
                if (idx >= 0 && idx < chunks.Length)
                {
                    totalSplats += chunks[idx].splatCount;
                }
            }

            byte[] combinedData = new byte[totalSplats * stride];
            int writeOffset = 0;

            using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                foreach (int chunkIndex in chunkIndices)
                {
                    if (chunkIndex < 0 || chunkIndex >= chunks.Length) continue;

                    ChunkInfo chunk = chunks[chunkIndex];
                    long byteOffset = chunk.startIndex * (long)stride;
                    int byteCount = chunk.splatCount * stride;

                    fs.Seek(byteOffset, SeekOrigin.Begin);
                    fs.Read(combinedData, writeOffset, byteCount);
                    writeOffset += byteCount;
                }
            }

            return combinedData;
        }

        private static string GetProjectRoot()
        {
            // Application.dataPath = "{ProjectRoot}/Assets"
            return Directory.GetParent(Application.dataPath).FullName;
        }
    }
}
