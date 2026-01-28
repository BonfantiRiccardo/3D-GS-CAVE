using UnityEngine;

namespace GaussianSplatting.Editor
{
    // This class is responsible for building a GSAsset from imported data
    public static class GSAssetBuilder
    {
        public static GSAsset BuildAsset(GSImportData data)
        {
            // Create a new GSAsset ScriptableObject
            var asset = ScriptableObject.CreateInstance<GSAsset>();

            // Initialize the asset with the provided data
            Vector3[] positions = data.Positions ?? System.Array.Empty<Vector3>();
            Unity.Mathematics.quaternion[] rotations = data.Rotations ?? System.Array.Empty<Unity.Mathematics.quaternion>();
            Vector3[] scales = data.Scales ?? System.Array.Empty<Vector3>();
            Vector3[] sh = data.SH ?? System.Array.Empty<Vector3>();
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
