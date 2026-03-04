using System;
using System.IO;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    /// <summary>
    /// PLYImporter is responsible for importing PLY files as ChunkedGSAsset ScriptableObjects.
    /// Uses the buffered AsyncPLYParser for high performance import of large files.
    /// Data is spatially sorted using Morton codes and partitioned into chunks for
    /// efficient frustum culling and streaming at runtime.
    /// </summary>
    [ScriptedImporter(5, "ply")]
    public class PLYImporter : ScriptedImporter
    {
        [SerializeField]
        [Tooltip("Controls how many spherical harmonic (SH) coefficients are imported.\n" +
                 "Full: All SH bands (highest quality, most memory)\n" +
                 "Reduced: Band 1 only (good quality, less memory)\n" +
                 "None: DC color only (fastest import, least memory)")]
        private SHImportQuality m_SHQuality = SHImportQuality.Full;

        [SerializeField]
        [Tooltip("Coordinate system handedness flip applied during import.\n" +
                 "None: No flip (data used as-is)\n" +
                 "FlipZ: Negate Z axis (common for COLMAP / 3DGS right-hand to Unity left-hand)\n" +
                 "FlipX: Negate X axis (alternative handedness conversion)")]
        private CoordinateFlip m_CoordinateFlip = CoordinateFlip.FlipZ;

        [SerializeField]
        [Tooltip("Number of splats per spatial chunk. Smaller values give finer culling " +
                 "granularity but increase chunk management overhead. Recommended: 2048 to 8192.")]
        private int m_ChunkSize = SpatialSorter.DefaultChunkSize;

        private sealed class ImportProgress : IProgress<float>, IDisposable
        {
            private readonly string _fileName;
            private bool _disposed;

            public ImportProgress(string fileName)
            {
                _fileName = fileName;
            }

            public void Report(float value)
            {
                if (_disposed || !CanShowProgressBar())
                {
                    return;
                }

                float clamped = Mathf.Clamp01(value);
                EditorUtility.DisplayProgressBar(
                    "Importing PLY",
                    $"{_fileName} - {clamped * 100f:F0}%",
                    clamped);
            }

            public void Dispose()
            {
                if (_disposed)
                {
                    return;
                }

                _disposed = true;
                if (!CanShowProgressBar())
                {
                    return;
                }

                EditorUtility.ClearProgressBar();
                EditorApplication.delayCall += EditorUtility.ClearProgressBar;
            }
        }

        private static bool CanShowProgressBar()
        {
            if (Application.isBatchMode)
            {
                return false;
            }

#if UNITY_2021_2_OR_NEWER
            if (AssetDatabase.IsAssetImportWorkerProcess())
            {
                return false;
            }
#endif

            return true;
        }

        [InitializeOnLoadMethod]
        private static void ClearStaleProgressBarAfterDomainReload()
        {
            if (!CanShowProgressBar())
            {
                return;
            }

            EditorApplication.delayCall += EditorUtility.ClearProgressBar;
        }

        /// <summary>
        /// Called when the asset is imported. Creates a ChunkedGSAsset from the PLY file.
        /// Data is spatially sorted and partitioned into chunks, then written to
        /// external .bytes files in Assets/StreamingAssets/{plyName}/ at the project root.
        /// </summary>
        public override void OnImportAsset(AssetImportContext ctx)
        {
            string fileName = Path.GetFileName(ctx.assetPath);
            using var progress = new ImportProgress(fileName);
            progress.Report(0f);

            GSImportData data;
            try
            {
                data = AsyncPLYParser.Parse(ctx.assetPath, progress, m_SHQuality, m_CoordinateFlip);
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"Buffered parser failed, falling back to legacy parser: {ex.Message}");
                data = PLYParser.Parse(ctx.assetPath);
            }

            // Create external data directory: {ProjectRoot}/Assets/StreamingAssets/{plyName}/
            string plyName = Path.GetFileNameWithoutExtension(ctx.assetPath);
            string relativeDataPath = "Assets/StreamingAssets/" + plyName;
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            string absoluteDataPath = Path.Combine(projectRoot,  "Assets", "StreamingAssets", plyName);

            // Use the chunked pipeline: sort spatially, partition, and write
            progress.Report(0.5f);
            var (asset, chunks) = ChunkedGSAssetBuilder.ProcessAndBuild(
                data,
                absoluteDataPath,
                relativeDataPath,
                m_ChunkSize);

            asset.name = plyName;

            Debug.Log(
                $"PLY import (chunked): {ctx.assetPath} | Splats={data.Positions?.Length ?? 0} | " +
                $"Chunks={chunks.Length} | ChunkSize={m_ChunkSize} | " +
                $"SH={data.ImportedSHQuality} | SHRestCount={data.SHRestCount} | " +
                $"Flip={data.AppliedFlip} | " +
                $"HasRot={data.HasRotations} | HasScale={data.HasScales} | " +
                $"HasColor={data.HasColors} | HasOpacity={data.HasOpacity} | " +
                $"ExternalData={absoluteDataPath}");

            ctx.AddObjectToAsset("ChunkedGSAsset", asset);
            ctx.SetMainObject(asset);
            progress.Report(1f);
        }
    }
}
