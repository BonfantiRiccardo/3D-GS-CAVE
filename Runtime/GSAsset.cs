using System;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting
{
    /// <summary>
    /// GSAsset ScriptableObject for storing Gaussian Splatting data.
    /// 
    /// Data is stored as raw byte[] blobs for fast serialization.
    /// Unity serializes byte[] as a single base64 block in text mode or a flat blob in binary mode,
    /// which is dramatically faster than serializing millions of individual Vector3/quaternion/Vector4 elements.
    /// 
    /// At runtime, typed arrays are reconstructed on first access and cached.
    /// ComputeBuffers can also be populated directly from byte[] via SetData.
    /// </summary>
    [CreateAssetMenu(fileName = "GSAsset", menuName = "Scriptable Objects/GSAsset")]
    public class GSAsset : ScriptableObject
    {
        [Header("Gaussian Splatting Settings")]
        [SerializeField, HideInInspector] private byte[] positionData;     // float3 per splat (12 bytes each)
        [SerializeField, HideInInspector] private byte[] rotationData;     // quaternion per splat (16 bytes each)
        [SerializeField, HideInInspector] private byte[] scaleData;        // float3 per splat (12 bytes each)
        [SerializeField, HideInInspector] private byte[] shData;           // float4 per splat (16 bytes each) — DC(xyz) + opacity(w)
        [SerializeField, HideInInspector] private byte[] shRestData;       // float per coeff (4 bytes each), total = splatCount * shRestCount
        [SerializeField, HideInInspector] private int shRestCount;         // number of SH rest coefficients per splat
        public int splatCount;                                              // Total number of splats
        public Bounds bounds;                                               // Axis-Aligned Bounding Box (AABB)

        // Cached typed arrays — reconstructed on first access
        [NonSerialized] private Vector3[] _positions;
        [NonSerialized] private quaternion[] _rotations;
        [NonSerialized] private Vector3[] _scales;
        [NonSerialized] private Vector4[] _sh;
        [NonSerialized] private float[] _shRest;

        // Public typed-array accessors (lazy reconstruction from byte blobs)
        public Vector3[] Positions => _positions ??= ReinterpretArray<Vector3>(positionData);
        public quaternion[] Rotations => _rotations ??= ReinterpretArray<quaternion>(rotationData);
        public Vector3[] Scales => _scales ??= ReinterpretArray<Vector3>(scaleData);
        public Vector4[] SH => _sh ??= ReinterpretArray<Vector4>(shData);
        public float[] SHRest => _shRest ??= ReinterpretArray<float>(shRestData);
        public int SHRestCount => shRestCount;

        // Direct byte[] access for ComputeBuffer.SetData(byte[]) — zero-copy GPU upload
        public byte[] PositionData => positionData;
        public byte[] RotationData => rotationData;
        public byte[] ScaleData => scaleData;
        public byte[] SHData => shData;
        public byte[] SHRestData => shRestData;

        /// <summary>
        /// Initializes the GSAsset from typed arrays (called by GSAssetBuilder during import).
        /// Converts typed arrays to byte blobs for efficient serialization.
        /// </summary>
        public void Initialize(
            Vector3[] positions,
            quaternion[] rotations,
            Vector3[] scales,
            Vector4[] sh,
            float[] shRest,
            int shRestCount,
            Bounds bounds)
        {
            this.positionData = ToByteArray(positions);
            this.rotationData = ToByteArray(rotations);
            this.scaleData = ToByteArray(scales);
            this.shData = ToByteArray(sh);
            this.shRestData = ToByteArray(shRest);
            this.shRestCount = shRestCount;
            this.splatCount = positions?.Length ?? 0;
            this.bounds = bounds;

            // Cache the source arrays directly (they're already in memory)
            _positions = positions;
            _rotations = rotations;
            _scales = scales;
            _sh = sh;
            _shRest = shRest;
        }

        /// <summary>
        /// Reinterpret a byte array as a typed array of blittable structs.
        /// Uses GCHandle pinning because Buffer.BlockCopy only works with primitive arrays,
        /// not struct arrays like Vector3[], quaternion[], Vector4[].
        /// </summary>
        private static T[] ReinterpretArray<T>(byte[] data) where T : struct
        {
            if (data == null || data.Length == 0)
                return Array.Empty<T>();

            int elementSize = Marshal.SizeOf<T>();
            int count = data.Length / elementSize;
            T[] result = new T[count];

            // Pin the destination struct array and copy bytes into it
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

        /// <summary>
        /// Convert a typed array of blittable structs to a raw byte array.
        /// Uses GCHandle pinning because Buffer.BlockCopy only works with primitive arrays,
        /// not struct arrays like Vector3[], quaternion[], Vector4[].
        /// </summary>
        private static byte[] ToByteArray<T>(T[] source) where T : struct
        {
            if (source == null || source.Length == 0)
                return Array.Empty<byte>();

            int elementSize = Marshal.SizeOf<T>();
            byte[] result = new byte[source.Length * elementSize];

            // Pin the source struct array and copy its raw bytes out
            GCHandle handle = GCHandle.Alloc(source, GCHandleType.Pinned);
            try
            {
                Marshal.Copy(handle.AddrOfPinnedObject(), result, 0, result.Length);
            }
            finally
            {
                handle.Free();
            }
            return result;
        }
    }
}