using System;
using System.IO;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    /// <summary>
    /// PLYImporter is responsible for importing PLY files as GSAsset ScriptableObjects.
    /// Uses the buffered AsyncPLYParser for high-performance import of large files.
    /// </summary>
    [ScriptedImporter(3, "ply")]
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
                 "FlipZ: Negate Z axis (common for COLMAP / 3DGS right-hand → Unity left-hand)\n" +
                 "FlipX: Negate X axis (alternative handedness conversion)")]
        private CoordinateFlip m_CoordinateFlip = CoordinateFlip.FlipZ;

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
        /// Called when the asset is imported. Creates a GSAsset from the PLY file.
        /// Uses AsyncPLYParser with buffered I/O and editor progress bar.
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

            GSAsset asset = GSAssetBuilder.BuildAsset(data);
            asset.name = Path.GetFileNameWithoutExtension(ctx.assetPath);

            Debug.Log(
                $"PLY import: {ctx.assetPath} | Splats={data.Positions?.Length ?? 0} | " +
                $"SH={data.ImportedSHQuality} | SHRestCount={data.SHRestCount} | " +
                $"Flip={data.AppliedFlip} | " +
                $"HasRot={data.HasRotations} | HasScale={data.HasScales} | " +
                $"HasColor={data.HasColors} | HasOpacity={data.HasOpacity}");

            ctx.AddObjectToAsset("GSAsset", asset);
            ctx.SetMainObject(asset);
            progress.Report(1f);
        }
    }
}
