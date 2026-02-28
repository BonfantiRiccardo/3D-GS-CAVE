using UnityEditor;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    /// <summary>
    /// Custom inspector for GSAsset ScriptableObjects.
    /// Uses metadata-only accessors to avoid triggering data loading from external .bytes files.
    /// </summary>
    [CustomEditor(typeof(GSAsset))]
    public class GSAssetInspector : UnityEditor.Editor
    {
        private Vector2 scroll; // Scroll position for bounds display

        /// <summary>
        /// Overrides the default inspector GUI to display GSAsset information.
        /// </summary>
        public override void OnInspectorGUI()
        {
            GSAsset asset = (GSAsset)target;

            EditorGUILayout.LabelField("GS Asset", EditorStyles.boldLabel);

            using (new EditorGUI.DisabledScope(true))
            {
                EditorGUILayout.IntField("Splat Count", asset.splatCount);
                EditorGUILayout.IntField("SH Rest Count/Splat", asset.SHRestCount);

                // Compute estimated total data size from metadata (no file loading)
                EditorGUILayout.TextField("Total Data", $"{asset.EstimatedTotalDataSize / (1024.0 * 1024.0):F1} MB");
            }

            // Show data storage location
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Data Storage", EditorStyles.boldLabel);

            using (new EditorGUI.DisabledScope(true))
            {
                EditorGUILayout.TextField("Location", asset.ExternalDataPath ?? "(not set)");
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Bounds", EditorStyles.boldLabel);

            using (var scrollView = new EditorGUILayout.ScrollViewScope(scroll, GUILayout.Height(70)))
            {
                scroll = scrollView.scrollPosition;

                using (new EditorGUI.DisabledScope(true))
                {
                    Vector3 center = asset.bounds.center;
                    Vector3 extents = asset.bounds.extents;

                    EditorGUILayout.TextField("Center", $"X {center.x}   Y {center.y}   Z {center.z}");
                    EditorGUILayout.TextField("Extents", $"X {extents.x}   Y {extents.y}   Z {extents.z}");
                }
            }
        }
    }
}
