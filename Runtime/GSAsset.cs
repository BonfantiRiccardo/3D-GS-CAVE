using System;
using System.IO;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting
{
    /// <summary>
    /// GSAsset ScriptableObject for storing Gaussian Splatting data.
    /// 
    /// Data is stored externally as .bytes files in a Assets/StreamingAssets/ folder at the project root. (Inspired by github.com/aras-p/UnityGaussianSplatting)
    /// 
    /// At runtime, data is loaded on-demand from external files and cached.
    /// ComputeBuffers can be populated directly from byte[] via SetData.
    /// </summary>
    [CreateAssetMenu(fileName = "GSAsset", menuName = "Scriptable Objects/GSAsset")]
    public class GSAsset : ScriptableObject
    {
        [Header("Gaussian Splatting Settings")]
        public int splatCount;                                              // Total number of splats
        public Bounds bounds;                                               // Axis-Aligned Bounding Box (AABB)

        [SerializeField, HideInInspector] private int shRestCount;         // number of SH rest coefficients per splat

        /// <summary>
        /// Path to external data folder, relative to the project root.
        /// Splat data is loaded from .bytes files in this folder.
        /// </summary>
        [SerializeField, HideInInspector] private string externalDataPath;

        // Cached typed arrays: reconstructed on first access from byte data
        [NonSerialized] private Vector3[] _positions;
        [NonSerialized] private quaternion[] _rotations;
        [NonSerialized] private Vector3[] _scales;
        [NonSerialized] private Vector4[] _sh;
        [NonSerialized] private float[] _shRest;

        // Cached raw byte arrays loaded from external files
        [NonSerialized] private byte[] _extPositionData;
        [NonSerialized] private byte[] _extRotationData;
        [NonSerialized] private byte[] _extScaleData;
        [NonSerialized] private byte[] _extSHData;
        [NonSerialized] private byte[] _extSHRestData;

        // External data file names

        public const string PositionsFileName = "positions.bytes";
        public const string RotationsFileName = "rotations.bytes";
        public const string ScalesFileName = "scales.bytes";
        public const string SHFileName = "sh.bytes";
        public const string SHRestFileName = "shrest.bytes";

        // Metadata accessors (never trigger data loading)

        public int SHRestCount => shRestCount;
        public string ExternalDataPath => externalDataPath;

        /// <summary>
        /// Estimated total data size in bytes, computed from metadata.
        /// Does not trigger data loading from external files.
        /// </summary>
        public long EstimatedTotalDataSize => (long)splatCount * (12 + 16 + 12 + 16 + shRestCount * 4);

        // Data accessors (load from external files on first access)

        public byte[] PositionData => LoadExternal(ref _extPositionData, PositionsFileName);
        public byte[] RotationData => LoadExternal(ref _extRotationData, RotationsFileName);
        public byte[] ScaleData => LoadExternal(ref _extScaleData, ScalesFileName);
        public byte[] SHData => LoadExternal(ref _extSHData, SHFileName);
        public byte[] SHRestData => LoadExternal(ref _extSHRestData, SHRestFileName);

        // Typed-array accessors (lazy reconstruction from byte data)
        public Vector3[] Positions => _positions ??= ReinterpretArray<Vector3>(PositionData);
        public quaternion[] Rotations => _rotations ??= ReinterpretArray<quaternion>(RotationData);
        public Vector3[] Scales => _scales ??= ReinterpretArray<Vector3>(ScaleData);
        public Vector4[] SH => _sh ??= ReinterpretArray<Vector4>(SHData);
        public float[] SHRest => _shRest ??= ReinterpretArray<float>(SHRestData);

        /// <summary>
        /// Initialize with external data storage. Data files should already be written
        /// to the externalDataPath folder before calling this method.
        /// </summary>
        public void InitializeExternal(
            int splatCount,
            int shRestCount,
            Bounds bounds,
            string externalDataPath)
        {
            this.splatCount = splatCount;
            this.shRestCount = shRestCount;
            this.bounds = bounds;
            this.externalDataPath = externalDataPath;
            ClearCaches();
        }

        /// <summary>
        /// Resolves the project-root-relative externalDataPath to an absolute file path
        /// for a given data file name.
        /// </summary>
        public string ResolveFilePath(string fileName)
        {
            if (string.IsNullOrEmpty(externalDataPath)) return null;
            string projectRoot = GetProjectRoot();
            return Path.Combine(projectRoot, externalDataPath, fileName);
        }

        private void ClearCaches()
        {
            _positions = null;
            _rotations = null;
            _scales = null;
            _sh = null;
            _shRest = null;
            _extPositionData = null;
            _extRotationData = null;
            _extScaleData = null;
            _extSHData = null;
            _extSHRestData = null;
        }

        /// <summary>
        /// Load data from external .bytes file. Cached after first load.
        /// </summary>
        private byte[] LoadExternal(ref byte[] cache, string fileName)
        {
            if (cache != null) return cache;

            string fullPath = ResolveFilePath(fileName);
            if (fullPath != null && File.Exists(fullPath))
            {
                cache = File.ReadAllBytes(fullPath);
                return cache;
            }

            Debug.LogWarning($"GSAsset: External data file not found: {fullPath}");
            return null;
        }

        private static string GetProjectRoot()
        {
            // Application.dataPath = "{ProjectRoot}/Assets"
            return Directory.GetParent(Application.dataPath).FullName;
        }

        /// <summary>
        /// Reinterpret a byte array as a typed array of blittable structs.
        /// </summary>
        private static T[] ReinterpretArray<T>(byte[] data) where T : struct
        {
            if (data == null || data.Length == 0)
                return Array.Empty<T>();

            int elementSize = Marshal.SizeOf<T>();
            int count = data.Length / elementSize;
            T[] result = new T[count];

            GCHandle handle = GCHandle.Alloc(result, GCHandleType.Pinned);
            try
            {
                Marshal.Copy(data, 0, handle.AddrOfPinnedObject(), data.Length);
            }
            finally
            {
                handle.Free();
            }
            return result;
        }
    }
}