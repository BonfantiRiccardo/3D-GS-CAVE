using UnityEngine;

namespace GaussianSplatting.Editor
{
    /// <summary>
    /// GSAssetBuilder is responsible for creating GSAsset ScriptableObjects from imported data.
    /// </summary>
    public static class GSAssetBuilder
    {
        /// <summary>
        /// Builds a GSAsset from the provided GSImportData.
        /// </summary>
        /// <param name="data">The imported data used to build the asset.</param>
        /// <returns>The constructed GSAsset.</returns>
        public static GSAsset BuildAsset(GSImportData data)
        {
            // Create a new GSAsset ScriptableObject
            var asset = ScriptableObject.CreateInstance<GSAsset>();

            // Initialize the asset with the provided data
            Vector3[] positions = data.Positions ?? System.Array.Empty<Vector3>();
            Unity.Mathematics.quaternion[] rotations = data.Rotations ?? System.Array.Empty<Unity.Mathematics.quaternion>();
            Vector3[] scales = data.Scales ?? System.Array.Empty<Vector3>();
            Vector4[] sh = data.SH ?? System.Array.Empty<Vector4>();
            float[] shRest = data.SHRest ?? System.Array.Empty<float>();

            asset.Initialize(
                positions,
                rotations,
                scales,
                sh,
                shRest,
                data.SHRestCount,
                data.Bounds);

            return asset;
        }
    }
}
