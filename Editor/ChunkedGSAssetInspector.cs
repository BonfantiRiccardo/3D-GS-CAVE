using UnityEditor;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    /// <summary>
    /// Custom inspector for ChunkedGSAsset that displays chunk statistics
    /// and provides visualization options.
    /// </summary>
    [CustomEditor(typeof(ChunkedGSAsset))]
    public class ChunkedGSAssetInspector : UnityEditor.Editor
    {
        private bool showChunkList = false;
        private Vector2 chunkListScrollPos;

        public override void OnInspectorGUI()
        {
            ChunkedGSAsset asset = (ChunkedGSAsset)target;

            // Draw default inspector for basic fields
            DrawDefaultInspector();

            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Chunked Asset Statistics", EditorStyles.boldLabel);

            EditorGUI.indentLevel++;

            // Total statistics
            EditorGUILayout.LabelField("Total Splats", asset.totalSplatCount.ToString("N0"));
            EditorGUILayout.LabelField("Total Chunks", asset.chunkCount.ToString());
            EditorGUILayout.LabelField("Chunk Size", asset.chunkSize.ToString("N0"));
            EditorGUILayout.LabelField("SH Rest Count", asset.SHRestCount.ToString());
            EditorGUILayout.LabelField("SH Bands", asset.SHBands.ToString());

            EditorGUILayout.Space(5);

            // Memory estimate
            long bytesPerSplat = ChunkedGSAsset.PositionStride + 
                                 ChunkedGSAsset.RotationStride + 
                                 ChunkedGSAsset.ScaleStride + 
                                 ChunkedGSAsset.SHStride + 
                                 asset.SHRestStride;
            long totalDataSize = (long)asset.totalSplatCount * bytesPerSplat;
            EditorGUILayout.LabelField("Data Size", FormatBytes(totalDataSize));
            EditorGUILayout.LabelField("Bytes per Splat", $"{bytesPerSplat} bytes");

            EditorGUILayout.Space(5);

            // Bounds
            EditorGUILayout.LabelField("Global Bounds");
            EditorGUI.indentLevel++;
            EditorGUILayout.LabelField("Center", asset.globalBounds.center.ToString("F2"));
            EditorGUILayout.LabelField("Size", asset.globalBounds.size.ToString("F2"));
            EditorGUI.indentLevel--;

            EditorGUILayout.Space(5);

            // External data path
            EditorGUILayout.LabelField("Asset Folder", asset.AssetFolderName);

            EditorGUI.indentLevel--;

            EditorGUILayout.Space(10);

            // Expandable chunk list
            if (asset.Chunks != null && asset.Chunks.Length > 0)
            {
                showChunkList = EditorGUILayout.Foldout(showChunkList, $"Chunks ({asset.Chunks.Length})");
                if (showChunkList)
                {
                    EditorGUI.indentLevel++;

                    // Scrollable list for large chunk counts
                    int maxVisible = 20;
                    if (asset.Chunks.Length > maxVisible)
                    {
                        chunkListScrollPos = EditorGUILayout.BeginScrollView(chunkListScrollPos, 
                            GUILayout.MaxHeight(400));
                    }

                    for (int i = 0; i < asset.Chunks.Length; i++)
                    {
                        ChunkInfo chunk = asset.Chunks[i];
                        EditorGUILayout.BeginHorizontal();
                        EditorGUILayout.LabelField($"[{i}]", GUILayout.Width(40));
                        EditorGUILayout.LabelField($"Splats: {chunk.splatCount}", GUILayout.Width(100));
                        EditorGUILayout.LabelField($"Start: {chunk.startIndex}", GUILayout.Width(100));
                        EditorGUILayout.LabelField($"Center: {chunk.bounds.center:F1}", GUILayout.ExpandWidth(true));
                        EditorGUILayout.EndHorizontal();
                    }

                    if (asset.Chunks.Length > maxVisible)
                    {
                        EditorGUILayout.EndScrollView();
                    }

                    EditorGUI.indentLevel--;
                }
            }
        }

        private static string FormatBytes(long bytes)
        {
            string[] suffixes = { "B", "KB", "MB", "GB", "TB" };
            int suffixIndex = 0;
            double size = bytes;

            while (size >= 1024 && suffixIndex < suffixes.Length - 1)
            {
                size /= 1024;
                suffixIndex++;
            }

            return $"{size:F2} {suffixes[suffixIndex]}";
        }
    }
}
