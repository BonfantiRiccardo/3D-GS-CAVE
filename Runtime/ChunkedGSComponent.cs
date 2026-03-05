using System;
using UnityEngine;

namespace GaussianSplatting
{   
    /// <summary>
    /// ChunkedGSComponent manages spatially chunked Gaussian Splatting data with frustum based streaming. 
    /// This component only loads chunks that are visible to the camera, enabling rendering of very large scenes.
    /// 
    /// It performs frustum culling at chunk level (typically 4096 splats per chunk) and dynamic loading and 
    /// unloading based on camera movement. Implements margin delta for preloading chunks slightly outside the frustum
    /// </summary>
    [ExecuteAlways]
    public class ChunkedGSComponent : MonoBehaviour
    {
        [Header("Asset")]
        [Tooltip("Reference to the ChunkedGSAsset containing spatially partitioned splat data")]
        public ChunkedGSAsset asset;

        [Header("Streaming Settings")]
        [Tooltip("Maximum number of splats that can be visible at once. " +
                 "Determines GPU buffer sizes. Capped to the scene's total " +
                 "splat count at runtime so small scenes do not over allocate. ")]
        public int maxVisibleSplats = 20_000_000;

        [Tooltip("Inner frustum margin for deciding when to load chunks (world units). " +
                 "Chunks entering this expanded frustum get loaded.")]
        [Range(0f, 50f)]
        public float frustumMargin = 3f;

        [Tooltip("Outer frustum margin for deciding when to unload chunks (world units). " +
                 "Must be larger than the inner margin to create a hysteresis band " +
                 "that prevents load/unload thrashing.")]
        [Range(0f, 100f)]
        public float unloadMargin = 5f;

        [Tooltip("Number of frames a chunk must be outside the outer frustum before eviction. " +
                 "Higher values reduce costly repacks at the cost of extra GPU memory.")]
        [Range(0, 300)]
        public int evictionDelayFrames = 60;

        [Tooltip("Maximum number of chunk GPU uploads (SetData calls) per frame. " +
                 "Async I/O dispatch is unlimited so the background thread reads ahead. " +
                 "Higher values reduce pop-in but cost more per frame.")]
        [Range(1, 256)]
        public int maxUploadsPerFrame = 128;

        [Header("Rendering")]
        [Tooltip("Color space conversion mode for splat colors.\n" +
                 "Auto: uses Unity's current color space.\n" +
                 "ForceLinear: always convert gamma to linear.\n" +
                 "ForceGamma: no conversion (keep gamma).")]
        public GSComponent.ColorSpaceMode colorSpaceMode = GSComponent.ColorSpaceMode.Auto;

        [Range(0.1f, 3.0f)]
        [Tooltip("Global scale multiplier for splat size.")]
        public float splatScale = 1.0f;

        [Header("Debugging")]
        [Tooltip("Render splat centers as small points instead of full Gaussians.")]
        public bool showSplatCenters = false;

        [Range(1.0f, 20.0f)]
        [Tooltip("Size of the debug center points in pixels.")]
        public float centerPointSize = 1.0f;

        [Tooltip("Show chunk bounding boxes as debug gizmos.")]
        public bool showChunkBounds = false;

        // Streaming manager
        private GSChunkStreamer streamer;

        // Sorting resources for the visible splats
        private GSSortingResources sortingResources;

        // Cached camera for updates
        private Camera cachedCamera;

        // Properties
        public Vector3 modelPosition => transform.position;
        public Quaternion ModelRotation => transform.rotation;
        public Vector3 modelScale => transform.lossyScale;

        /// <summary>
        /// Number of splats currently visible and loaded.
        /// </summary>
        public int ActiveSplatCount => streamer?.VisibleSplatCount ?? 0;

        /// <summary>
        /// Number of chunks currently loaded.
        /// </summary>
        public int LoadedChunkCount => streamer?.LoadedChunkCount ?? 0;

        /// <summary>
        /// Total number of chunks in the asset.
        /// </summary>
        public int TotalChunkCount => asset?.chunkCount ?? 0;

        /// <summary>
        /// Cumulative count of async chunk loads completed.
        /// </summary>
        public int LoadCount => streamer?.LoadCount ?? 0;

        /// <summary>
        /// Cumulative count of chunk evictions.
        /// </summary>
        public int EvictCount => streamer?.EvictCount ?? 0;

        /// <summary>
        /// Estimated GPU memory in bytes for the allocated splat data buffers.
        /// Uses the actual pool capacity (may be less than maxVisibleSplats
        /// for small scenes).
        /// </summary>
        public long EstimatedGPUMemoryBytes
        {
            get
            {
                if (asset == null || streamer == null) return 0;
                long capacity = streamer.PoolCapacity;
                long perSplat = ChunkedGSAsset.PositionStride + ChunkedGSAsset.RotationStride +
                                ChunkedGSAsset.ScaleStride + ChunkedGSAsset.SHStride;
                if (asset.SHRestCount > 0)
                    perSplat += asset.SHRestStride;
                // Add remap buffer: one uint per splat
                perSplat += sizeof(uint);
                return capacity * perSplat;
            }
        }

        /// <summary>
        /// Number of async chunk reads still pending on the I/O thread.
        /// </summary>
        public int PendingReadCount => streamer?.PendingReadCount ?? 0;

        /// <summary>
        /// Number of SH bands in the asset.
        /// </summary>
        public int ShBandsNumber => asset?.SHBands ?? 0;

        /// <summary>
        /// True when the streamer has allocated buffers.
        /// </summary>
        public bool HasBuffers => streamer?.HasBuffers ?? false;

        /// <summary>
        /// Returns true once when chunk data changed since last call.
        /// Used by the render feature to force a re sort after chunk load/evict.
        /// </summary>
        public bool ConsumeBufferDirtyFlag() => streamer?.ConsumeBufferDirtyFlag() ?? false;

        /// <summary>
        /// Indirection buffer mapping logical splat indices to physical pool positions.
        /// Bound to the sort compute shader as _SplatRemap.
        /// </summary>
        public ComputeBuffer RemapBuffer => streamer?.RemapBuffer;

        // Public buffer accessors for the render pipeline
        public ComputeBuffer PositionBuffer => streamer?.PositionBuffer;
        public ComputeBuffer RotationBuffer => streamer?.RotationBuffer;
        public ComputeBuffer ScaleBuffer => streamer?.ScaleBuffer;
        public ComputeBuffer SHBuffer => streamer?.SHBuffer;
        public ComputeBuffer SHRestBuffer => streamer?.SHRestBuffer;
        public GSSortingResources SortingResources => sortingResources;

        void OnEnable()
        {
            InitializeStreamer();
            ChunkedGSManager.Register(this);
        }

        void OnDisable()
        {
            DisposeStreamer();
            ChunkedGSManager.Unregister(this);
        }

        void OnDestroy()
        {
            DisposeStreamer();
            ChunkedGSManager.Unregister(this);
        }

        void OnApplicationQuit()
        {
            // Belt-and-suspenders: ensure GPU resources are released even if
            // OnDisable is skipped during an abrupt quit.
            DisposeStreamer();
        }

        void OnValidate()
        {
            // Reinitialize if settings changed
            if (streamer != null)
            {
                DisposeStreamer();
                InitializeStreamer();
            }
            ChunkedGSManager.Register(this);
        }

        void Update()
        {
            // In edit mode visibility is driven entirely by the render feature using
            // the Scene camera, so we skip here to avoid competing with it.
            if (!Application.isPlaying) return;

            if (cachedCamera == null)
            {
                cachedCamera = Camera.main;
            }

            if (cachedCamera != null && streamer != null)
            {
                // Update visibility based on camera frustum
                Matrix4x4 modelMatrix = transform.localToWorldMatrix;
                streamer.UpdateVisibility(cachedCamera, modelMatrix);
            }
        }

        /// <summary>
        /// Initializes the chunk streamer with current settings.
        /// </summary>
        private void InitializeStreamer()
        {
            if (asset == null)
            {
                Debug.LogWarning("ChunkedGSComponent: No ChunkedGSAsset assigned.");
                return;
            }

            if (asset.chunkCount == 0)
            {
                Debug.LogWarning("ChunkedGSComponent: Asset has no chunks.");
                return;
            }

            streamer = new GSChunkStreamer(asset, maxVisibleSplats, frustumMargin, unloadMargin, evictionDelayFrames, maxUploadsPerFrame);

            // Pre allocate sort resources at pool capacity so they are never
            // reallocated during rendering when the active splat count changes.
            if (sortingResources == null)
            {
                sortingResources = new GSSortingResources();
            }
            sortingResources.EnsureCapacity(streamer.PoolCapacity);
        }

        /// <summary>
        /// Disposes the streamer and releases resources.
        /// </summary>
        private void DisposeStreamer()
        {
            sortingResources?.Dispose();
            sortingResources = null;

            streamer?.Dispose();
            streamer = null;
        }

        /// <summary>
        /// Updates visibility for a specific camera. Call this from the render pass
        /// to ensure visibility is up to date before sorting and rendering.
        /// </summary>
        public void UpdateVisibilityForCamera(Camera camera)
        {
            if (streamer == null || camera == null) return;

            Matrix4x4 modelMatrix = transform.localToWorldMatrix;
            streamer.UpdateVisibility(camera, modelMatrix);
        }

        /// <summary>
        /// Updates the active splat count on the pre allocated sorting resources.
        /// Called by the render pass before sorting. No GPU allocation occurs here.
        /// </summary>
        public void InitializeSortingResources()
        {
            int splatCount = ActiveSplatCount;
            if (splatCount <= 0) return;

            if (sortingResources == null)
            {
                sortingResources = new GSSortingResources();
                sortingResources.EnsureCapacity(streamer?.PoolCapacity ?? maxVisibleSplats);
            }

            sortingResources.SetActiveSplatCount(splatCount);
        }

        /// <summary>
        /// Binds splat data buffers to a compute shader for sorting and processing.
        /// </summary>
        public void BindToCompute(UnityEngine.Rendering.ComputeCommandBuffer cmd, ComputeShader shader, int kernel)
        {
            if (streamer == null) return;

            if (streamer.PositionBuffer != null)
                cmd.SetComputeBufferParam(shader, kernel, "_Positions", streamer.PositionBuffer);
            if (streamer.RotationBuffer != null)
                cmd.SetComputeBufferParam(shader, kernel, "_Rotations", streamer.RotationBuffer);
            if (streamer.ScaleBuffer != null)
                cmd.SetComputeBufferParam(shader, kernel, "_Scales", streamer.ScaleBuffer);
        }

        /// <summary>
        /// Binds buffers to a material for rendering.
        /// </summary>
        public void BindTo(Material material)
        {
            if (streamer == null) return;

            if (streamer.PositionBuffer != null)
                material.SetBuffer("_Positions", streamer.PositionBuffer);
            if (streamer.RotationBuffer != null)
                material.SetBuffer("_Rotations", streamer.RotationBuffer);
            if (streamer.ScaleBuffer != null)
                material.SetBuffer("_Scales", streamer.ScaleBuffer);
            if (streamer.SHBuffer != null)
                material.SetBuffer("_SH", streamer.SHBuffer);

            if (ShBandsNumber > 1 && streamer.SHRestBuffer != null)
            {
                material.SetBuffer("_SHRest", streamer.SHRestBuffer);
                material.SetInt("_SHRestCount", asset.SHRestCount);
            }

            material.SetInt("_SplatCount", ActiveSplatCount);
        }

        void OnDrawGizmosSelected()
        {
            if (!showChunkBounds || asset == null || asset.Chunks == null) return;

            Matrix4x4 modelMatrix = transform.localToWorldMatrix;
            Gizmos.matrix = modelMatrix;

            // Draw all chunk bounds
            for (int i = 0; i < asset.Chunks.Length; i++)
            {
                ChunkInfo chunk = asset.Chunks[i];

                // Color based on load state
                bool isLoaded = streamer != null && streamer.IsChunkLoaded(i);
                Gizmos.color = isLoaded ? new Color(0f, 1f, 0f, 0.3f) : new Color(1f, 0f, 0f, 0.1f);

                Gizmos.DrawWireCube(chunk.bounds.center, chunk.bounds.size);
            }

            Gizmos.matrix = Matrix4x4.identity;
        }
    }
}
