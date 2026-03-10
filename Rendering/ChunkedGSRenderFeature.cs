using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using System.Collections.Generic;


namespace GaussianSplatting
{
    /// <summary>
    /// Render Feature for URP. Renders ChunkedGSComponent instances which use frustum based 
    /// chunk streaming for large scenes.
    /// 
    /// This feature handles:
    /// - Updating chunk visibility based on camera frustum
    /// - Sorting only the visible splats
    /// - Rendering visible splats with proper depth ordering
    /// </summary>
    public class ChunkedGSRenderFeature : ScriptableRendererFeature
    {
        [SerializeField] private ChunkedGSRenderSettings settings = new();
        private ChunkedGSRenderPass renderPass;

        // Track the material we create so we can clean it up on dispose
        private Material ownedMaterial;

        public override void Create()
        {
            if (settings == null)
            {
                settings = new ChunkedGSRenderSettings();
            }

            if (settings.splatMaterial == null)
            {
                Shader shader = Shader.Find(ChunkedGSRenderSettings.DefaultShaderName);
                if (shader != null)
                {
                    var mat = new Material(shader) { hideFlags = HideFlags.HideAndDontSave };
                    settings.splatMaterial = mat;
                    ownedMaterial = mat;
                }
            }

            if (settings.sortingShader == null)
            {
                settings.sortingShader = Resources.Load<ComputeShader>("GSSorting");
                if (settings.sortingShader == null)
                {
                    settings.sortingShader = (ComputeShader)UnityEngine.Resources.Load("GaussianSplatting/Rendering/GSSorting");
                }
            }

            if (settings.precomputeShader == null)
            {
                settings.precomputeShader = Resources.Load<ComputeShader>("GSPrecompute");
                if (settings.precomputeShader == null)
                {
                    settings.precomputeShader = (ComputeShader)UnityEngine.Resources.Load("GaussianSplatting/Rendering/GSPrecompute");
                }
            }

            renderPass = new ChunkedGSRenderPass(settings);
            renderPass.renderPassEvent = settings.renderPassEvent;
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            if (settings == null || !settings.enabled || settings.splatMaterial == null)
            {
                return;
            }

            renderer.EnqueuePass(renderPass);
        }

        protected override void Dispose(bool disposing)
        {
            // Clean up the Material we created with HideAndDontSave to prevent
            // persistent allocation leaks across domain reloads.
            if (ownedMaterial != null)
            {
                if (Application.isPlaying)
                    Destroy(ownedMaterial);
                else
                    DestroyImmediate(ownedMaterial);
                ownedMaterial = null;
            }
        }

        [Serializable]
        public class ChunkedGSRenderSettings
        {
            public const string DefaultShaderName = "GaussianSplatting/GSShader";

            public bool enabled = true;
            public RenderPassEvent renderPassEvent = RenderPassEvent.BeforeRenderingTransparents;
            public Material splatMaterial;
            public float splatSize = 1.0f;

            [Header("Sorting")]
            public ComputeShader sortingShader;

            [Header("Precompute")]
            public ComputeShader precomputeShader;
        }

        private class ChunkedGSRenderPass : ScriptableRenderPass
        {
            private readonly ChunkedGSRenderSettings settings;
            private const int SortGroupSize = 256;
            private const int RadixBins = 256;

            public ChunkedGSRenderPass(ChunkedGSRenderSettings settings)
            {
                this.settings = settings;
            }

            // Incremental sort state (static because ExecuteSortPass is a static render func)
            private static Vector3 lastFullSortCameraPosition;
            private static Quaternion lastFullSortCameraRotation;
            private static int framesSinceFullSort = 0;

            // Thresholds for choosing full vs incremental sort
            private const float FullSortPositionThreshold = 5.0f;    // meters
            private const float FullSortRotationThreshold = 0.05f;   // ~6 degrees
            private const int MaxFramesBetweenFullSorts = 120;        // Force full sort every N frames
            private const int OddEvenRepairPasses = 3;                // Number of even+odd pass pairs

            // Camera state caching per camera for sort optimization
            private struct CameraPose
            {
                public Vector3 position;
                public Quaternion rotation;
            }

            private readonly Dictionary<int, CameraPose> cameraPoseById = new();
            private const float CameraPositionThreshold = 0.001f;
            private const float CameraPositionThresholdSqr = CameraPositionThreshold * CameraPositionThreshold;
            private const float CameraRotationThreshold = 0.001f;

            private bool IsSortDrivingCamera(Camera camera)
            {
                if (camera == null) return false;

#if UNITY_EDITOR
                if (camera.cameraType == CameraType.Preview || camera.cameraType == CameraType.Reflection)
                    return false;

                if (Application.isPlaying)
                    return camera.cameraType != CameraType.SceneView; // game cameras drive in play mode

                return camera.cameraType == CameraType.SceneView;     // scene camera drives in edit mode
#else
                return true;
#endif
            }

            private bool CameraHasMoved(Camera camera)
            {
                if (!IsSortDrivingCamera(camera))
                    return false;

                int id = camera.GetInstanceID();
                Vector3 currentPos = camera.transform.position;
                Quaternion currentRot = camera.transform.rotation;

                if (!cameraPoseById.TryGetValue(id, out CameraPose last))
                {
                    cameraPoseById[id] = new CameraPose { position = currentPos, rotation = currentRot };
                    return true; // first frame for this camera
                }

                if ((currentPos - last.position).sqrMagnitude > CameraPositionThresholdSqr)
                {
                    cameraPoseById[id] = new CameraPose { position = currentPos, rotation = currentRot };
                    return true;
                }

                float rotDelta = 1.0f - Mathf.Abs(Quaternion.Dot(currentRot, last.rotation));
                if (rotDelta > CameraRotationThreshold)
                {
                    cameraPoseById[id] = new CameraPose { position = currentPos, rotation = currentRot };
                    return true;
                }

                return false;
            }

            private static bool NeedsFullSort(Camera camera, bool chunksDirty)
            {
                // Always full sort if chunks changed (remap invalidated)
                if (chunksDirty) return true;

                // First sort ever
                if (framesSinceFullSort == 0) return true;

                // Periodic full sort to prevent drift
                if (framesSinceFullSort >= MaxFramesBetweenFullSorts) return true;

                // Large camera movement
                float posDelta = Vector3.Distance(camera.transform.position, lastFullSortCameraPosition);
                if (posDelta > FullSortPositionThreshold) return true;

                float rotDot = Quaternion.Dot(camera.transform.rotation, lastFullSortCameraRotation);
                if (1.0f - Mathf.Abs(rotDot) > FullSortRotationThreshold) return true;

                return false;
            }

            private class PassData
            {
                public Material material;
                public float splatSize;
                public Matrix4x4 viewMatrix;
                public Matrix4x4 projMatrix;
                public Vector3 cameraWorldPos;
                public ChunkedGSComponent[] components;
                public bool[] showSplatCenters;
                public float[] centerPointSizes;
            }

            private class SortPassData
            {
                public ComputeShader sortingShader;
                public ComputeShader precomputeShader;
                public Matrix4x4 viewMatrix;
                public Matrix4x4 projMatrix;
                public Vector4 screenParams;
                public float nearPlane;
                public float farPlane;
                public float splatSize;
                public Vector3 cameraWorldPos;
                public ChunkedGSComponent[] components;
                public bool cameraMoved;
                public bool[] chunksDirty;
                public float[] splatScales;
                public int[] colorSpaceModes;
                public bool[] debugPoints;
                public float[] debugPointSizes;
                public Camera camera;
            }

            // Sorting shader kernels Ids
            private static int kernelInitSortBuffers = -1;
            private static int kernelCalcViewData = -1;
            private static int kernelCalcSortKeys = -1;

            private static int kernelInitRadix = -1;
            private static int kernelUpsweep = -1;
            private static int kernelScan = -1;
            private static int kernelDownsweep = -1;

            private static int kernelOddEvenPass = -1;
            private static int kernelUpdateSortKeysInPlace = -1;

            // Precompute shader kernel Id
            private static int kernelCalcSplatViewData = -1;

            private static void EnsureSortKernels(ComputeShader shader)
            {
                if (shader == null) return;
                if (kernelInitSortBuffers >= 0) return;

                kernelInitSortBuffers = shader.FindKernel("InitSortBuffers");
                kernelCalcViewData = shader.FindKernel("CalcViewData");
                kernelCalcSortKeys = shader.FindKernel("CalcSortKeys");
                kernelInitRadix = shader.FindKernel("InitDeviceRadixSort");
                kernelUpsweep = shader.FindKernel("Upsweep");
                kernelScan = shader.FindKernel("Scan");
                kernelDownsweep = shader.FindKernel("Downsweep");
                kernelOddEvenPass = shader.FindKernel("OddEvenPass");
                kernelUpdateSortKeysInPlace = shader.FindKernel("UpdateSortKeysInPlace");
            }

            private static void EnsurePrecomputeKernels(ComputeShader shader)
            {
                if (shader == null) return;
                if (kernelCalcSplatViewData >= 0) return;

                kernelCalcSplatViewData = shader.FindKernel("CalcSplatViewData");
            }

            private static void ExecuteSortPass(SortPassData data, ComputeGraphContext context)
            {
                if (data.sortingShader == null || data.components == null || data.components.Length == 0 || data.precomputeShader == null)
                {
                    return;
                }

                EnsureSortKernels(data.sortingShader);
                EnsurePrecomputeKernels(data.precomputeShader);

                for (int i = 0; i < data.components.Length; i++)
                {
                    ChunkedGSComponent component = data.components[i];
                    if (component == null || !component.HasBuffers)
                    {
                        continue;
                    }

                    // In play mode the game camera drives chunk loading; in edit mode the scene camera does. 
                    // The other view still renders whatever chunks are already loaded but does not trigger I/O.
                    bool driveVisibility = true;
#if UNITY_EDITOR
                    if (Application.isPlaying)
                        driveVisibility = data.camera.cameraType != CameraType.SceneView;
                    else
                        driveVisibility = data.camera.cameraType == CameraType.SceneView;
#endif
                    if (driveVisibility)
                    {
                        component.UpdateVisibilityForCamera(data.camera);
                    }

                    // Skip if no visible splats after visibility update
                    if (component.ActiveSplatCount <= 0)
                    {
                        continue;
                    }

                    component.InitializeSortingResources();
                    var resources = component.SortingResources;
                    if (resources == null || !resources.IsInitialized)
                    {
                        continue;
                    }

                    // Sort is needed when the camera moved OR when chunk data changed
                    bool dirtyAtBuild =
                        data.chunksDirty != null &&
                        i < data.chunksDirty.Length &&
                        data.chunksDirty[i];

                    // Catches load/evict that happened inside UpdateVisibilityForCamera this frame
                    bool dirtyAfterVisibility = component.ConsumeBufferDirtyFlag();

                    bool componentNeedsSort = data.cameraMoved || dirtyAtBuild || dirtyAfterVisibility;

                    int splatCount = component.ActiveSplatCount;
                    int groups = Mathf.CeilToInt((float)splatCount / SortGroupSize);

                    // Common uniforms
                    context.cmd.SetComputeMatrixParam(data.sortingShader, "_MatrixV", data.viewMatrix);
                    context.cmd.SetComputeMatrixParam(data.sortingShader, "_MatrixP", data.projMatrix);
                    context.cmd.SetComputeVectorParam(data.sortingShader, "_ScreenParams", data.screenParams);
                    context.cmd.SetComputeIntParam(data.sortingShader, "_SplatCount", splatCount);
                    context.cmd.SetComputeFloatParam(data.sortingShader, "_NearPlane", data.nearPlane);
                    context.cmd.SetComputeFloatParam(data.sortingShader, "_FarPlane", data.farPlane);

                    // Pool based indirection: enable remap and bind the buffer to
                    // the two kernels that read raw splat data.
                    context.cmd.SetComputeIntParam(data.sortingShader, "_UseRemap", 1);
                    var remapBuf = component.RemapBuffer;
                    if (remapBuf != null)
                    {
                        context.cmd.SetComputeBufferParam(data.sortingShader, kernelCalcViewData, "_SplatRemap", remapBuf);
                        context.cmd.SetComputeBufferParam(data.sortingShader, kernelUpdateSortKeysInPlace, "_SplatRemap", remapBuf);
                        context.cmd.SetComputeBufferParam(data.precomputeShader, kernelCalcSplatViewData, "_SplatRemap", remapBuf);
                    }

                    // Model transform uniforms
                    context.cmd.SetComputeVectorParam(data.sortingShader, "_ModelPosition", component.modelPosition);
                    Quaternion sortRot = component.ModelRotation;
                    context.cmd.SetComputeVectorParam(data.sortingShader, "_ModelRotation", new Vector4(sortRot.x, sortRot.y, sortRot.z, sortRot.w));
                    context.cmd.SetComputeVectorParam(data.sortingShader, "_ModelScale", component.modelScale);

                    // Only run sorting passes if camera moved or chunks changed
                    if (componentNeedsSort)
                    {
                        bool fullSort = NeedsFullSort(data.camera,
                            dirtyAtBuild || dirtyAfterVisibility);

                        if (fullSort)
                        {
                            // Init sort buffers
                            resources.BindSortInputs(context.cmd, data.sortingShader, kernelInitSortBuffers);
                            component.BindToCompute(context.cmd, data.sortingShader, kernelInitSortBuffers);
                            context.cmd.DispatchCompute(data.sortingShader, kernelInitSortBuffers, groups, 1, 1);

                            // View data
                            resources.BindSortInputs(context.cmd, data.sortingShader, kernelCalcViewData);
                            component.BindToCompute(context.cmd, data.sortingShader, kernelCalcViewData);
                            context.cmd.DispatchCompute(data.sortingShader, kernelCalcViewData, groups, 1, 1);

                            // Sort keys
                            resources.BindSortInputs(context.cmd, data.sortingShader, kernelCalcSortKeys);
                            context.cmd.DispatchCompute(data.sortingShader, kernelCalcSortKeys, groups, 1, 1);

                            // Radix sort (4 passes)
                            context.cmd.SetComputeIntParam(data.sortingShader, "e_numKeys", splatCount);
                            context.cmd.SetComputeIntParam(data.sortingShader, "e_threadBlocks", resources.ThreadBlocks);

                            for (int pass = 0; pass < 4; pass++)
                            {
                                bool useAltAsSource = (pass % 2) == 1;
                                resources.BindRadixSortBuffers(context.cmd, data.sortingShader, kernelInitRadix, useAltAsSource);
                                resources.BindRadixSortBuffers(context.cmd, data.sortingShader, kernelUpsweep, useAltAsSource);
                                resources.BindRadixSortBuffers(context.cmd, data.sortingShader, kernelScan, useAltAsSource);
                                resources.BindRadixSortBuffers(context.cmd, data.sortingShader, kernelDownsweep, useAltAsSource);

                                context.cmd.SetComputeIntParam(data.sortingShader, "e_radixShift", pass * 8);
                                context.cmd.DispatchCompute(data.sortingShader, kernelInitRadix, 1, 1, 1);
                                context.cmd.DispatchCompute(data.sortingShader, kernelUpsweep, resources.ThreadBlocks, 1, 1);
                                context.cmd.DispatchCompute(data.sortingShader, kernelScan, RadixBins, 1, 1);
                                context.cmd.DispatchCompute(data.sortingShader, kernelDownsweep, resources.ThreadBlocks, 1, 1);
                            }
                            // Track full sort state
                            lastFullSortCameraPosition = data.camera.transform.position;
                            lastFullSortCameraRotation = data.camera.transform.rotation;
                            framesSinceFullSort = 1;
                        }
                        else    // Incremental repair sort
                        {
                            // Step 1: Update depth keys in-place (preserves existing sort order)
                            resources.BindSortInputs(context.cmd, data.sortingShader, kernelUpdateSortKeysInPlace);
                            component.BindToCompute(context.cmd, data.sortingShader, kernelUpdateSortKeysInPlace);
                            context.cmd.DispatchCompute(data.sortingShader, kernelUpdateSortKeysInPlace, groups, 1, 1);

                            // Step 2: Run N pairs of odd-even passes to fix local inversions
                            int halfGroups = Mathf.CeilToInt((float)splatCount / (SortGroupSize * 2));
                            resources.BindSortInputs(context.cmd, data.sortingShader, kernelOddEvenPass);

                            for (int pass = 0; pass < OddEvenRepairPasses; pass++)
                            {
                                // Even phase: compare-swap (0,1), (2,3), (4,5)...
                                context.cmd.SetComputeIntParam(data.sortingShader, "_OddEvenPhase", 0);
                                context.cmd.DispatchCompute(data.sortingShader, kernelOddEvenPass, halfGroups, 1, 1);

                                // Odd phase: compare-swap (1,2), (3,4), (5,6)...
                                context.cmd.SetComputeIntParam(data.sortingShader, "_OddEvenPhase", 1);
                                context.cmd.DispatchCompute(data.sortingShader, kernelOddEvenPass, halfGroups, 1, 1);
                            }

                            framesSinceFullSort++;
                        }
                    }

                    // Dispatch CalcSplatViewData on the precompute shader
                    var pc = data.precomputeShader;

                    // Shared uniforms (same names as sorting shader for ergonomic C# binding)
                    context.cmd.SetComputeMatrixParam(pc, "_MatrixV", data.viewMatrix);
                    context.cmd.SetComputeMatrixParam(pc, "_MatrixP", data.projMatrix);
                    context.cmd.SetComputeVectorParam(pc, "_VecScreenParams", data.screenParams);
                    context.cmd.SetComputeIntParam(pc, "_SplatCount", splatCount);
                    context.cmd.SetComputeIntParam(pc, "_UseRemap", 1);

                    // Model transform
                    context.cmd.SetComputeVectorParam(pc, "_ModelPosition", component.modelPosition);
                    Quaternion pcRot = component.ModelRotation;
                    context.cmd.SetComputeVectorParam(pc, "_ModelRotation", new Vector4(pcRot.x, pcRot.y, pcRot.z, pcRot.w));
                    context.cmd.SetComputeVectorParam(pc, "_ModelScale", component.modelScale);

                    // Precompute-specific uniforms
                    float effectiveSplatScale = data.splatSize * (data.splatScales != null && i < data.splatScales.Length ? data.splatScales[i] : 1.0f);
                    context.cmd.SetComputeFloatParam(pc, "_SplatScale", effectiveSplatScale);
                    context.cmd.SetComputeVectorParam(pc, "_CameraWorldPos", data.cameraWorldPos);
                    context.cmd.SetComputeIntParam(pc, "_SHBands", component.ShBandsNumber);
                    context.cmd.SetComputeIntParam(pc, "_SHRestCount", component.asset != null ? component.asset.SHRestCount : 0);

                    int csMode = (data.colorSpaceModes != null && i < data.colorSpaceModes.Length) ? data.colorSpaceModes[i] : 0;
                    context.cmd.SetComputeIntParam(pc, "_ColorSpaceMode", csMode);

                    bool debugPts = data.debugPoints != null && i < data.debugPoints.Length && data.debugPoints[i];
                    context.cmd.SetComputeIntParam(pc, "_DebugPoints", debugPts ? 1 : 0);
                    float debugPtSize = data.debugPointSizes != null && i < data.debugPointSizes.Length ? data.debugPointSizes[i] : 3.0f;
                    context.cmd.SetComputeFloatParam(pc, "_DebugPointSize", debugPtSize);

                    Matrix4x4 matrixMV = data.viewMatrix;
                    context.cmd.SetComputeMatrixParam(pc, "_MatrixMV", matrixMV);

                    // Bind splat data buffers (positions, rotations, scales)
                    component.BindToCompute(context.cmd, pc, kernelCalcSplatViewData);
                    if (component.SHBuffer != null)
                        context.cmd.SetComputeBufferParam(pc, kernelCalcSplatViewData, "_SH", component.SHBuffer);
                    if (component.SHRestBuffer != null)
                        context.cmd.SetComputeBufferParam(pc, kernelCalcSplatViewData, "_SHRest", component.SHRestBuffer);

                    resources.BindSplatViewDataOutput(context.cmd, pc, kernelCalcSplatViewData);
                    context.cmd.DispatchCompute(pc, kernelCalcSplatViewData, groups, 1, 1);
                }
            }

            private static void ExecutePass(PassData data, RasterGraphContext context)
            {
                if (data.material == null || data.components == null || data.components.Length == 0)
                {
                    return;
                }

                data.material.SetFloat("_SplatSize", data.splatSize);
                data.material.SetVector("_CameraWorldPos", data.cameraWorldPos);

                var mpb = new MaterialPropertyBlock();

                for (int i = 0; i < data.components.Length; i++)
                {
                    ChunkedGSComponent component = data.components[i];
                    if (component == null || !component.HasBuffers || component.ActiveSplatCount <= 0)
                        continue;

                    var resources = component.SortingResources;
                    if (resources == null || !resources.IsInitialized)
                        continue;

                    mpb.Clear();
                    mpb.SetBuffer("_SplatViewData", resources.GetSplatViewData());
                    mpb.SetBuffer("_OrderBuffer", resources.SortIndices);
                    mpb.SetInteger("_SplatCount", component.ActiveSplatCount);

                    bool showCenters = data.showSplatCenters != null && i < data.showSplatCenters.Length && data.showSplatCenters[i];
                    mpb.SetInteger("_ShowSplatCenters", showCenters ? 1 : 0);
                    float centerSize = data.centerPointSizes != null && i < data.centerPointSizes.Length ? data.centerPointSizes[i] : 0.02f;
                    mpb.SetFloat("_CenterPointSize", centerSize);

                    context.cmd.DrawProcedural(
                        Matrix4x4.identity,
                        data.material,
                        0,
                        MeshTopology.Triangles,
                        6,
                        component.ActiveSplatCount,
                        mpb);
                }
            }

            public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
            {
                if (settings == null || settings.splatMaterial == null)
                {
                    return;
                }

                var components = ChunkedGSManager.Components;
                if (components == null || components.Count == 0)
                {
                    return;
                }

                UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();
                UniversalResourceData resourceData = frameData.Get<UniversalResourceData>();

                const string sortPassName = "Chunked GS Sort";
                const string passName = "Chunked GS Pass";

                Matrix4x4 viewMatrix = cameraData.GetViewMatrix();
                bool renderIntoTexture = !resourceData.isActiveTargetBackBuffer;
                Matrix4x4 projMatrix = GL.GetGPUProjectionMatrix(cameraData.GetProjectionMatrix(), renderIntoTexture);

                // Sorting pass
                if (settings.sortingShader != null)
                {
                    using (var sortBuilder = renderGraph.AddComputePass<SortPassData>(sortPassName, out var sortPassData))
                    {
                        sortBuilder.AllowPassCulling(false);

                        sortPassData.sortingShader = settings.sortingShader;
                        sortPassData.precomputeShader = settings.precomputeShader;
                        sortPassData.viewMatrix = viewMatrix;
                        sortPassData.projMatrix = projMatrix;
                        sortPassData.screenParams = new Vector4(cameraData.camera.pixelWidth, cameraData.camera.pixelHeight,
                            1.0f / Mathf.Max(1, cameraData.camera.pixelWidth), 1.0f / Mathf.Max(1, cameraData.camera.pixelHeight));
                        sortPassData.nearPlane = cameraData.camera.nearClipPlane;
                        sortPassData.farPlane = cameraData.camera.farClipPlane;
                        sortPassData.splatSize = settings.splatSize;
                        sortPassData.cameraWorldPos = cameraData.camera.transform.position;
                        sortPassData.camera = cameraData.camera;

                        sortPassData.cameraMoved = CameraHasMoved(cameraData.camera);

                        int componentCount = components.Count;
                        sortPassData.components = new ChunkedGSComponent[componentCount];
                        sortPassData.chunksDirty = new bool[componentCount];
                        sortPassData.splatScales = new float[componentCount];
                        sortPassData.colorSpaceModes = new int[componentCount];
                        sortPassData.debugPoints = new bool[componentCount];
                        sortPassData.debugPointSizes = new float[componentCount];
                        for (int i = 0; i < componentCount; i++)
                        {
                            sortPassData.components[i] = components[i];
                            sortPassData.chunksDirty[i] = components[i].ConsumeBufferDirtyFlag();
                            sortPassData.splatScales[i] = components[i].splatScale;
                            sortPassData.colorSpaceModes[i] = (int)components[i].colorSpaceMode;
                            sortPassData.debugPoints[i] = components[i].showSplatCenters;
                            sortPassData.debugPointSizes[i] = components[i].centerPointSize;
                        }

                        sortBuilder.SetRenderFunc((SortPassData d, ComputeGraphContext ctx) => ExecuteSortPass(d, ctx));
                    }
                }

                // Raster pass
                using (var builder = renderGraph.AddRasterRenderPass<PassData>(passName, out var passData))
                {
                    builder.AllowPassCulling(false);
                    builder.SetRenderAttachment(resourceData.activeColorTexture, 0);
                    builder.SetRenderAttachmentDepth(resourceData.activeDepthTexture);

                    passData.material = settings.splatMaterial;
                    passData.splatSize = settings.splatSize;
                    passData.viewMatrix = viewMatrix;
                    passData.projMatrix = projMatrix;
                    passData.cameraWorldPos = cameraData.camera.transform.position;

                    int componentCount = components.Count;
                    passData.components = new ChunkedGSComponent[componentCount];
                    passData.showSplatCenters = new bool[componentCount];
                    passData.centerPointSizes = new float[componentCount];
                    for (int i = 0; i < componentCount; i++)
                    {
                        passData.components[i] = components[i];
                        passData.showSplatCenters[i] = components[i].showSplatCenters;
                        passData.centerPointSizes[i] = components[i].centerPointSize;
                    }

                    builder.SetRenderFunc((PassData d, RasterGraphContext ctx) => ExecutePass(d, ctx));
                }
            }
        }
    }
}
