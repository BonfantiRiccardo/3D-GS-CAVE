using UnityEditor.AssetImporters;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    [ScriptedImporter(1, "ply")]
    public class PLYImporter : ScriptedImporter     //Imports PLY files as GSAsset
    {
        [Header("Parsing Options")]         // Specify import options in the inspector
        [SerializeField] private bool importRotations = true;
        [SerializeField] private bool importScales = true;
        [SerializeField] private bool importSH = false;
        [SerializeField] private RotationOrder rotationOrder = RotationOrder.WXYZ;

        // This method is called when the asset is imported and is responsible for creating the GSAsset
        // It uses the PLYParser to read the PLY file and GSAssetBuilder to create the asset
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var options = new PLYImportOptions
            {
                ImportRotations = importRotations,
                ImportScales = importScales,
                ImportSH = importSH,
                RotationOrder = rotationOrder
            };

            GSImportData data = PLYParser.Parse(ctx.assetPath, options);
            GSAsset asset = GSAssetBuilder.BuildAsset(data);
            asset.name = System.IO.Path.GetFileNameWithoutExtension(ctx.assetPath);

            Debug.Log($"PLY import: {ctx.assetPath} | ImportSH={options.ImportSH} | SHRestCount={data.SHRestCount} | SHRestLen={data.SHRest?.Length ?? 0} | SHDCLen={data.SH?.Length ?? 0}");

            ctx.AddObjectToAsset("GSAsset", asset);         // Add the created asset to the import context
            ctx.SetMainObject(asset);
        }
    }
}