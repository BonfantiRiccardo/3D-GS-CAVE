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
    [ScriptedImporter(6, "ply")]
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

        /// <summary>
        /// IProgress wrapper using the Unity Progress API (task-based progress system).
        /// Progress tasks are always visible in the Background Tasks window and the status bar.
        /// </summary>
        private sealed class ImportProgress : IProgress<float>, IDisposable
        {
            private readonly int _progressId;
            private bool _disposed;
            private string _phase;

            public ImportProgress(string fileName)
            {
                _progressId = Progress.Start("Importing PLY", fileName, Progress.Options.Synchronous);
                _phase = fileName;
            }

            /// <summary>Sets the current phase description shown in the progress UI.</summary>
            public void SetPhase(string phase)
            {
                _phase = phase;
            }

            public void Report(float value)
            {
                if (_disposed) return;
                Progress.Report(_progressId, Mathf.Clamp01(value), _phase);
            }

            public void Dispose()
            {
                if (_disposed) return;
                _disposed = true;
                Progress.Remove(_progressId);
            }
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

            // Phase 1: Parse PLY file
            progress.SetPhase("Parsing PLY...");
            progress.Report(0f);

            GSImportData data;
            try
            {
                // Create a sub-progress that maps parser's 0-1 range to our 0-0.5 range
                var parseProgress = new ScaledProgress(progress, 0f, 0.5f);
                data = AsyncPLYParser.Parse(ctx.assetPath, parseProgress, m_SHQuality, m_CoordinateFlip);
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"Buffered parser failed, falling back to legacy parser: {ex.Message}");
                data = PLYParser.Parse(ctx.assetPath);
            }

            // Phase 2: Sort, chunk, write external data, build asset
            string plyName = Path.GetFileNameWithoutExtension(ctx.assetPath);
            string absoluteDataPath = Path.Combine(Application.streamingAssetsPath, plyName);

            progress.SetPhase("Chunking and building octree...");
            progress.Report(0.5f);

            // Create a sub-progress that maps builder's 0-1 range to our 0.5-0.95 range
            var buildProgress = new ScaledProgress(progress, 0.5f, 0.95f);
            var (asset, chunks) = ChunkedGSAssetBuilder.ProcessAndBuild(
                data,
                absoluteDataPath,
                plyName,
                m_ChunkSize,
                buildProgress);

            asset.name = plyName;

            progress.SetPhase("Finalising asset...");
            progress.Report(0.95f);

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

    /// <summary>
    /// Maps an inner IProgress{float} 0-1 range to a sub-range of an outer IProgress{float}.
    /// Used to compose multi-phase progress from independent phases that each report 0-1.
    /// </summary>
    internal sealed class ScaledProgress : IProgress<float>
    {
        private readonly IProgress<float> _outer;
        private readonly float _start;
        private readonly float _range;

        public ScaledProgress(IProgress<float> outer, float start, float end)
        {
            _outer = outer;
            _start = start;
            _range = end - start;
        }

        public void Report(float value)
        {
            _outer.Report(_start + Mathf.Clamp01(value) * _range);
        }
    }
}
