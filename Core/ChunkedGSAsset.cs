using System;
using System.IO;
using UnityEngine;

namespace GaussianSplatting
{
    /// <summary>
    /// ChunkedGSAsset stores spatially chunked Gaussian splat data. It supports streaming individual
    /// chunks based on visibility, enabling rendering of very large scenes that exceed GPU memory.
    /// 
    /// Data is stored externally in the StreamingAssets folder:
    ///   {StreamingAssets}/{assetName}/positions.bytes  Position data for all chunks contiguously
    ///   {StreamingAssets}/{assetName}/rotations.bytes  Rotation data for all chunks contiguously
    ///   {StreamingAssets}/{assetName}/scales.bytes     Scale data for all chunks contiguously
    ///   {StreamingAssets}/{assetName}/sh.bytes         SH DC data for all chunks contiguously
    ///   {StreamingAssets}/{assetName}/shrest.bytes     SH rest data for all chunks contiguously
    /// 
    /// Chunk metadata (bounds, counts, offsets) is serialized directly in this ScriptableObject
    /// via the ChunkInfo[] array.
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
        [Tooltip("Folder name inside StreamingAssets where .bytes files are stored")]
        [SerializeField, HideInInspector] 
        private string assetFolderName;

        // Chunk metadata array (serialized in the asset for fast access)
        [SerializeField, HideInInspector]
        private ChunkInfo[] chunks;

        [SerializeField, HideInInspector]
        private ChunkOctreeNode[] octreeNodes;

        // File names for external data
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

        // Per-band SH rest: splitting across 3 buffers keeps every
        // individual ComputeBuffer under the 2 GB structured-buffer limit.
        public const int SHBand1Count  = 9;                                 // 3 channels × 3 coefficients (L=1)
        public const int SHBand2Count  = 15;                                // 3 channels × 5 coefficients (L=2)
        public const int SHBand3Count  = 21;                                // 3 channels × 7 coefficients (L=3)
        public const int SHBand1Stride = SHBand1Count * sizeof(float);      // 36
        public const int SHBand2Stride = SHBand2Count * sizeof(float);      // 60
        public const int SHBand3Stride = SHBand3Count * sizeof(float);      // 84

        // Public accessors
        public int SHRestCount => shRestCount;
        public string AssetFolderName => assetFolderName;
        public ChunkInfo[] Chunks => chunks;
        public ChunkOctreeNode[] OctreeNodes => octreeNodes;
        public bool HasOctree => octreeNodes != null && octreeNodes.Length > 0;
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
            string assetFolderName)
        {
            this.totalSplatCount = totalSplatCount;
            this.chunkSize = chunkSize;
            this.shRestCount = shRestCount;
            this.globalBounds = globalBounds;
            this.chunks = chunks;
            this.chunkCount = chunks?.Length ?? 0;
            this.assetFolderName = assetFolderName;

            // Build octree over chunk AABBs for fast frustum culling
            if (chunks != null && chunks.Length > 0)
            {
                this.octreeNodes = ChunkOctree.Build(chunks, globalBounds);
                Debug.Log($"ChunkedGSAsset: built octree with {octreeNodes.Length} nodes over {chunks.Length} chunks");
            }
            else
            {
                this.octreeNodes = System.Array.Empty<ChunkOctreeNode>();
            }
        }

        /// <summary>
        /// Resolves a data file name to an absolute path using Application.streamingAssetsPath.
        /// Works both in the editor and in builds.
        /// </summary>
        public string ResolveFilePath(string fileName)
        {
            if (string.IsNullOrEmpty(assetFolderName)) return null;
            return Path.Combine(Application.streamingAssetsPath, assetFolderName, fileName);
        }

    }
}
