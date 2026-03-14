using System;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    /// <summary>
    /// GSAssetBuilder is responsible for creating GSAsset ScriptableObjects from imported data.
    /// Supports external .bytes file storage.
    /// </summary>
    public static class GSAssetBuilder
    {
        /// <summary>
        /// Writes external .bytes data files for a GSAsset to the specified directory.
        /// Creates the directory if it doesn't exist.
        /// </summary>
        public static void WriteExternalData(GSImportData data, string absoluteDirectoryPath)
        {
            Directory.CreateDirectory(absoluteDirectoryPath);

            WriteTypedArrayToFile(Path.Combine(absoluteDirectoryPath, GSAsset.PositionsFileName),
                data.Positions);
            WriteTypedArrayToFile(Path.Combine(absoluteDirectoryPath, GSAsset.RotationsFileName),
                data.Rotations);
            WriteTypedArrayToFile(Path.Combine(absoluteDirectoryPath, GSAsset.ScalesFileName),
                data.Scales);
            WriteTypedArrayToFile(Path.Combine(absoluteDirectoryPath, GSAsset.SHFileName),
                data.SH);
            WriteFloatArrayToFile(Path.Combine(absoluteDirectoryPath, GSAsset.SHRestFileName),
                data.SHRest);
        }

        /// <summary>
        /// Builds a GSAsset with external data storage. Data files must already be written
        /// to the relativeDataPath folder via WriteExternalData() before calling this.
        /// </summary>
        public static GSAsset BuildAsset(GSImportData data, string assetFolderName)
        {
            var asset = ScriptableObject.CreateInstance<GSAsset>();

            Vector3[] positions = data.Positions ?? Array.Empty<Vector3>();

            asset.InitializeExternal(
                positions.Length,
                data.SHRestCount,
                data.Bounds,
                assetFolderName);

            return asset;
        }


        /// <summary>
        /// Write a typed struct array to a binary file.
        /// </summary>
        private static void WriteTypedArrayToFile<T>(string filePath, T[] data) where T : struct
        {
            if (data == null || data.Length == 0)
            {
                File.WriteAllBytes(filePath, Array.Empty<byte>());
                return;
            }

            int elementSize = Marshal.SizeOf<T>();
            byte[] bytes = new byte[data.Length * elementSize];

            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                Marshal.Copy(handle.AddrOfPinnedObject(), bytes, 0, bytes.Length);
            }
            finally
            {
                handle.Free();
            }

            File.WriteAllBytes(filePath, bytes);
        }

        /// <summary>
        /// Write a float array to a binary file using Buffer.BlockCopy.
        /// </summary>
        private static void WriteFloatArrayToFile(string filePath, float[] data)
        {
            if (data == null || data.Length == 0)
            {
                File.WriteAllBytes(filePath, Array.Empty<byte>());
                return;
            }

            byte[] bytes = new byte[data.Length * sizeof(float)];
            Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(filePath, bytes);
        }
    }
}
