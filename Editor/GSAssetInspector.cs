using UnityEditor;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    /// <summary>
    /// Custom inspector for GSAsset ScriptableObjects.
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
                // Use byte[] lengths divided by element stride to avoid reconstructing typed arrays
                EditorGUILayout.IntField("Position Data", asset.PositionData != null ? asset.PositionData.Length / 12 : 0);
                EditorGUILayout.IntField("Rotation Data", asset.RotationData != null ? asset.RotationData.Length / 16 : 0);
                EditorGUILayout.IntField("Scale Data", asset.ScaleData != null ? asset.ScaleData.Length / 12 : 0);
                EditorGUILayout.IntField("SH DC Data", asset.SHData != null ? asset.SHData.Length / 16 : 0);
                EditorGUILayout.IntField("SH Rest Count/Splat", asset.SHRestCount);
                EditorGUILayout.IntField("SH Rest Total", asset.SHRestData != null ? asset.SHRestData.Length / 4 : 0);
                
                // Estimate total asset memory in MB
                long totalBytes = (asset.PositionData?.Length ?? 0) + (asset.RotationData?.Length ?? 0) +
                                  (asset.ScaleData?.Length ?? 0) + (asset.SHData?.Length ?? 0) +
                                  (asset.SHRestData?.Length ?? 0);
                EditorGUILayout.TextField("Total Data", $"{totalBytes / (1024.0 * 1024.0):F1} MB");
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
