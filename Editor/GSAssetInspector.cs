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
                EditorGUILayout.IntField("Position Count", asset.Positions != null ? asset.Positions.Length : -1);
                EditorGUILayout.IntField("Rotation Count", asset.Rotations != null ? asset.Rotations.Length : -1);
                EditorGUILayout.IntField("Scale Count", asset.Scales != null ? asset.Scales.Length : -1);
                EditorGUILayout.IntField("SH Rest Count", asset.SHRestCount);
                EditorGUILayout.IntField("SH Rest Total", asset.SHRestCount > 0 ? asset.splatCount * asset.SHRestCount : 0);
                EditorGUILayout.IntField("SH DC Count", asset.SH != null ? asset.SH.Length : -1);
                EditorGUILayout.IntField("SH Rest Array Length", asset.SHRest != null ? asset.SHRest.Length : -1);
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
