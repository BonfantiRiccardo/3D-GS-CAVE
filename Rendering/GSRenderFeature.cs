using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace GaussianSplatting
{
    /// <summary>
    /// Gaussian Splatting Render Feature for URP. Injects a render pass to render Gaussian splats with optional tile-based sorting.
    /// </summary>
    public class GSRenderFeature : ScriptableRendererFeature
    {
        // Settings for the render feature (example: material, splat size, etc.)
        [SerializeField] private GSRenderFeatureSettings settings = new();
        private GSRenderPass renderPass;        // The render pass that will be injected

        /// <summary>
        /// Called when the render feature is created. Initializes the render pass and settings.
        /// </summary>
        public override void Create()
        {
            if (settings == null)       //Ensure settings are initialized
            {
                settings = new GSRenderFeatureSettings();
            }

            if (settings.splatMaterial == null) //Create default material if none is provided
            {
                Shader shader = Shader.Find(GSRenderFeatureSettings.DefaultShaderName);
                if (shader != null)             //Check if shader is foundn and create material
                {
                    settings.splatMaterial = new Material(shader) { hideFlags = HideFlags.HideAndDontSave };
                }
            }

            // Load sorting compute shader if not already assigned
            if (settings.sortingShader == null)
            {
                settings.sortingShader = Resources.Load<ComputeShader>("GSSorting");
                if (settings.sortingShader == null)
                {
                    // Try to find it by path
                    settings.sortingShader = (ComputeShader)UnityEngine.Resources.Load("GaussianSplatting/Rendering/GSSorting");
                }
            }

            // Create the render pass with the provided settings
            renderPass = new GSRenderPass(settings);
            // Configures where the render pass should be injected.
            renderPass.renderPassEvent = settings.renderPassEvent;

            // You can request URP color texture and depth buffer as inputs by uncommenting the line below,
            // URP will ensure copies of these resources are available for sampling before executing the render pass.
            // Only uncomment it if necessary, it will have a performance impact, especially on mobiles and other TBDR GPUs where it will break render passes.
            //renderPass.ConfigureInput(ScriptableRenderPassInput.Color | ScriptableRenderPassInput.Depth);

            // You can request URP to render to an intermediate texture by uncommenting the line below.
            // Use this option for passes that do not support rendering directly to the backbuffer.
            // Only uncomment it if necessary, it will have a performance impact, especially on mobiles and other TBDR GPUs where it will break render passes.
            //renderPass.requiresIntermediateTexture = true;
        }


        // Here you can inject one or multiple render passes in the renderer.
        // This method is called when setting up the renderer once per-camera.
        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            if (settings == null || !settings.enabled || settings.splatMaterial == null)
            {
                return;     // Skip adding the pass if disabled or settings/material is missing
            }

            renderer.EnqueuePass(renderPass);
        }

        /// <summary>
        /// Settings for the Gaussian Splatting Render Feature.
        /// </summary>
        [Serializable]
        public class GSRenderFeatureSettings
        {
            //Set default shader name for Gaussian Splatting
            public const string DefaultShaderName = "GaussianSplatting/GSShader";

            public bool enabled = true;     //Specifies if the render feature is enabled
            //Event to inject the render pass
            public RenderPassEvent renderPassEvent = RenderPassEvent.BeforeRenderingTransparents; 
            //public RenderPassEvent renderPassEvent = RenderPassEvent.AfterRenderingTransparents;
            public Material splatMaterial;      //Material used for rendering splats
            public float splatSize = 1.0f;    //Global scale multiplier for per-splat size
            
            [Header("Sorting")]
            // Gaussian Splatting requires back-to-front rendering for correct alpha blending (alpha compositing is not commutative)
            public ComputeShader sortingShader; // GSSorting.compute
        }

        /// <summary>
        /// Render pass implementation for rendering Gaussian splats with optional tile-based sorting.
        /// </summary>
        private class GSRenderPass : ScriptableRenderPass
        {
            private readonly GSRenderFeatureSettings settings;  //Settings for the render pass
            private const int SortGroupSize = 256;
            private const int RadixBins = 256;
            
            // Camera state caching for sort optimization
            private Vector3 lastCameraPosition;
            private Quaternion lastCameraRotation;
            private bool hasCachedCamera = false;
            private const float CameraPositionThreshold = 0.001f;  // Minimum movement to trigger re-sort
            private const float CameraRotationThreshold = 0.001f;  // Minimum rotation to trigger re-sort (dot product)

            // Constructor to initialize the render pass with settings
            public GSRenderPass(GSRenderFeatureSettings settings)
            {
                this.settings = settings;
            }

            /// <summary>
            /// Check if camera has moved enough to require re-sorting.
            /// </summary>
            private bool CameraHasMoved(Camera camera)
            {
                if (!hasCachedCamera)
                {
                    return true;  // First frame, always sort
                }

                Vector3 currentPos = camera.transform.position;
                Quaternion currentRot = camera.transform.rotation;

                // Check position change
                float posDelta = Vector3.Distance(currentPos, lastCameraPosition);
                if (posDelta > CameraPositionThreshold)
                {
                    return true;
                }

                // Check rotation change (using dot product for efficiency)
                float rotDot = Quaternion.Dot(currentRot, lastCameraRotation);
                if (1.0f - Mathf.Abs(rotDot) > CameraRotationThreshold)
                {
                    return true;
                }

                return false;
            }

            /// <summary>
            /// Cache the current camera state.
            /// </summary>
            private void CacheCameraState(Camera camera)
            {
                lastCameraPosition = camera.transform.position;
                lastCameraRotation = camera.transform.rotation;
                hasCachedCamera = true;
            }

            /// <summary>
            /// Data structure for the render pass execution. Contains all necessary parameters and resources.
            /// </summary>
            private class PassData
            {
                public Material material;
                public float splatSize;
                public Matrix4x4 viewMatrix;
                public Matrix4x4 projMatrix;
                public Vector3 cameraWorldPos;  // Camera world position for view-dependent SH
                public GSComponent[] components;
                public bool[] showSplatCenters;     // Per-component debug center toggle
                public float[] centerPointSizes;    // Per-component center point size
            }

            /// <summary>
            /// Data structure for the sorting pass execution. Contains all necessary parameters and resources.
            /// </summary>
            private class SortPassData
            {
                public ComputeShader sortingShader;
                public Matrix4x4 viewMatrix;
                public Matrix4x4 projMatrix;
                public Vector4 screenParams;
                public float nearPlane;
                public float farPlane;
                public float splatSize;             // Global splat scale for CalcSplatViewData
                public Vector3 cameraWorldPos;      // Camera position for view-dependent SH
                public GSComponent[] components;
                public bool needsSort;  // Whether sorting is actually needed this frame
                public float[] splatScales;         // Per-component splat scale from GSComponent
                public int[] colorSpaceModes;       // Per-component color space mode
                public bool[] debugPoints;          // Per-component debug point toggle
                public float[] debugPointSizes;     // Per-component debug point size
            }

            private static int kernelInitSortBuffers = -1;
            private static int kernelCalcViewData = -1;
            private static int kernelCalcSortKeys = -1;
            private static int kernelCalcSplatViewData = -1;
            private static int kernelInitRadix = -1;
            private static int kernelUpsweep = -1;
            private static int kernelScan = -1;
            private static int kernelDownsweep = -1;

            /// <summary>
            /// Ensures that the compute shader kernels are initialized.
            /// <param name="shader">The compute shader containing the kernels.</param>
            /// </summary>
            private static void EnsureKernels(ComputeShader shader)
            {
                if (shader == null) return;
                if (kernelInitSortBuffers >= 0) return;

                kernelInitSortBuffers = shader.FindKernel("InitSortBuffers");
                kernelCalcViewData = shader.FindKernel("CalcViewData");
                kernelCalcSortKeys = shader.FindKernel("CalcSortKeys");
                kernelCalcSplatViewData = shader.FindKernel("CalcSplatViewData");
                kernelInitRadix = shader.FindKernel("InitDeviceRadixSort");
                kernelUpsweep = shader.FindKernel("Upsweep");
                kernelScan = shader.FindKernel("Scan");
                kernelDownsweep = shader.FindKernel("Downsweep");
            }

            /// <summary>
            /// This static method is passed as the RenderFunc delegate to the RenderGraph compute pass.
            /// It is used to execute the sorting pass.
            /// </summary>
            /// <param name="data">The data required for the sorting pass.</param>
            /// <param name="context">The compute graph context for dispatching compute shaders.</param>
            private static void ExecuteSortPass(SortPassData data, ComputeGraphContext context)
            {
                if (data.sortingShader == null || data.components == null || data.components.Length == 0)
                {
                    return;
                }

                EnsureKernels(data.sortingShader);

                for (int i = 0; i < data.components.Length; i++)
                {
                    GSComponent component = data.components[i];
                    if (component == null || !component.HasBuffers || component.ActiveSplatCount <= 0)
                    {
                        continue;
                    }

                    component.InitializeSortingResources();
                    var resources = component.SortingResources;
                    if (resources == null || !resources.IsInitialized)
                    {
                        continue;
                    }

                    int splatCount = resources.SplatCount;
                    int groups = Mathf.CeilToInt((float)splatCount / SortGroupSize);

                    // Common uniforms
                    context.cmd.SetComputeMatrixParam(data.sortingShader, "_MatrixV", data.viewMatrix);
                    context.cmd.SetComputeMatrixParam(data.sortingShader, "_MatrixP", data.projMatrix);
                    context.cmd.SetComputeVectorParam(data.sortingShader, "_ScreenParams", data.screenParams);
                    context.cmd.SetComputeIntParam(data.sortingShader, "_SplatCount", splatCount);
                    context.cmd.SetComputeFloatParam(data.sortingShader, "_NearPlane", data.nearPlane);
                    context.cmd.SetComputeFloatParam(data.sortingShader, "_FarPlane", data.farPlane);
                    
                    // Model transform uniforms
                    context.cmd.SetComputeVectorParam(data.sortingShader, "_ModelPosition", component.modelPosition);
                    Quaternion sortRot = component.ModelRotation;
                    context.cmd.SetComputeVectorParam(data.sortingShader, "_ModelRotation", new Vector4(sortRot.x, sortRot.y, sortRot.z, sortRot.w));
                    context.cmd.SetComputeVectorParam(data.sortingShader, "_ModelScale", component.modelScale);

                    // Only run sorting passes if camera has moved
                    if (data.needsSort)
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
                    }
                    
                    // Compute per-splat view data (clip pos, ellipse axes, color)
                    // Also precomputes all expensive per-splat data so vertex shader only does quad expansion
                    context.cmd.SetComputeVectorParam(data.sortingShader, "_VecScreenParams", data.screenParams);
                    // Use per-component splat scale (from GSComponent slider) multiplied by global setting
                    float effectiveSplatScale = data.splatSize * (data.splatScales != null && i < data.splatScales.Length ? data.splatScales[i] : 1.0f);
                    context.cmd.SetComputeFloatParam(data.sortingShader, "_SplatScale", effectiveSplatScale);
                    context.cmd.SetComputeVectorParam(data.sortingShader, "_CameraWorldPos", data.cameraWorldPos);
                    context.cmd.SetComputeIntParam(data.sortingShader, "_SHBands", component.ShBandsNumber);
                    context.cmd.SetComputeIntParam(data.sortingShader, "_SHRestCount", component.gsAsset != null ? component.gsAsset.SHRestCount : 0);
                    // Color space mode from GSComponent
                    int csMode = (data.colorSpaceModes != null && i < data.colorSpaceModes.Length) ? data.colorSpaceModes[i] : 0;
                    context.cmd.SetComputeIntParam(data.sortingShader, "_ColorSpaceMode", csMode);
                    // Debug points mode
                    bool debugPts = data.debugPoints != null && i < data.debugPoints.Length && data.debugPoints[i];
                    context.cmd.SetComputeIntParam(data.sortingShader, "_DebugPoints", debugPts ? 1 : 0);
                    float debugPtSize = data.debugPointSizes != null && i < data.debugPointSizes.Length ? data.debugPointSizes[i] : 3.0f;
                    context.cmd.SetComputeFloatParam(data.sortingShader, "_DebugPointSize", debugPtSize);
                    
                    // Compute combined model-view matrix
                    Matrix4x4 matrixMV = data.viewMatrix; // Model is identity since we apply model transform in shader
                    context.cmd.SetComputeMatrixParam(data.sortingShader, "_MatrixMV", matrixMV);
                    
                    // Bind SH buffers for view-dependent color
                    component.BindToCompute(context.cmd, data.sortingShader, kernelCalcSplatViewData);
                    if (component.SHBuffer != null)
                        context.cmd.SetComputeBufferParam(data.sortingShader, kernelCalcSplatViewData, "_SH", component.SHBuffer);
                    if (component.SHRestBuffer != null)
                        context.cmd.SetComputeBufferParam(data.sortingShader, kernelCalcSplatViewData, "_SHRest", component.SHRestBuffer);
                    
                    resources.BindSplatViewDataOutput(context.cmd, data.sortingShader, kernelCalcSplatViewData);
                    context.cmd.DispatchCompute(data.sortingShader, kernelCalcSplatViewData, groups, 1, 1);
                }
            }

            // This static method is passed as the RenderFunc delegate to the RenderGraph render pass.
            // It is used to execute draw commands.
            private static void ExecutePass(PassData data, RasterGraphContext context)
            {
                if (data.material == null || data.components == null || data.components.Length == 0)
                {
                    return;
                }

                // Set common uniforms (these are still needed for any fallback/debug rendering)
                data.material.SetFloat("_SplatSize", data.splatSize);
                data.material.SetVector("_CameraWorldPos", data.cameraWorldPos);

                var mpb = new MaterialPropertyBlock();


                for (int i = 0; i < data.components.Length; i++)
                {
                    GSComponent component = data.components[i];
                    if (component == null || !component.HasBuffers || component.ActiveSplatCount <= 0)
                        continue;

                    var resources = component.SortingResources;
                    if (resources == null || !resources.IsInitialized)
                        continue;

                    // MaterialPropertyBlock binds per-component buffers without modifying the
                    // shared material, allowing multiple splat objects to coexist.
                    mpb.Clear();
                    mpb.SetBuffer("_SplatViewData", resources.GetSplatViewData());
                    mpb.SetBuffer("_OrderBuffer", resources.SortIndices);
                    mpb.SetInteger("_SplatCount", component.ActiveSplatCount);
                    
                    // Debug: show splat centers as points
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

            // RecordRenderGraph is where the RenderGraph handle can be accessed, through which render passes can be added to the graph.
            // FrameData is a context container through which URP resources can be accessed and managed.
            public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
            {
                if (settings == null || settings.splatMaterial == null)
                {
                    Debug.LogWarning("GSRenderPass: Missing settings or splat material.");
                    return;
                }

                var components = GSManager.Components;
                if (components == null || components.Count == 0)
                {
                    Debug.LogWarning("No GSComponents registered in GSManager.");
                }

                // Use this scope to set the required inputs and outputs of the pass and to
                // setup the passData with the required properties needed at pass execution time.

                // Make use of frameData to access resources and camera data through the dedicated containers.
                // Eg:
                // UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();
                UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();
                UniversalResourceData resourceData = frameData.Get<UniversalResourceData>();

                const string sortPassName = "Gaussian Splat Sort";
                const string passName = "Gaussian Splat Pass";

                // Build GPU projection matrix
                // GetProjectionMatrix() already contains camera jitter in URP.
                // renderIntoTexture must match current active target to get correct Y-flip behavior.
                Matrix4x4 viewMatrix = cameraData.GetViewMatrix();
                bool renderIntoTexture = !resourceData.isActiveTargetBackBuffer;
                Matrix4x4 projMatrix = GL.GetGPUProjectionMatrix(cameraData.GetProjectionMatrix(), renderIntoTexture);

                // Sorting pass - always enabled as GS requires back-to-front rendering
                if (settings.sortingShader != null)
                {
                    using (var sortBuilder = renderGraph.AddComputePass<SortPassData>(sortPassName, out var sortPassData))
                    {
                        sortBuilder.AllowPassCulling(false);

                        sortPassData.sortingShader = settings.sortingShader;
                        sortPassData.viewMatrix = viewMatrix;
                        sortPassData.projMatrix = projMatrix;
                        sortPassData.screenParams = new Vector4(cameraData.camera.pixelWidth, cameraData.camera.pixelHeight,
                            1.0f / Mathf.Max(1, cameraData.camera.pixelWidth), 1.0f / Mathf.Max(1, cameraData.camera.pixelHeight));
                        sortPassData.nearPlane = cameraData.camera.nearClipPlane;
                        sortPassData.farPlane = cameraData.camera.farClipPlane;
                        sortPassData.splatSize = settings.splatSize;
                        sortPassData.cameraWorldPos = cameraData.camera.transform.position;
                        
                        // Check if camera has moved and sorting is needed
                        sortPassData.needsSort = CameraHasMoved(cameraData.camera);
                        if (sortPassData.needsSort)
                        {
                            CacheCameraState(cameraData.camera);
                        }

                        int componentCount = components?.Count ?? 0;
                        sortPassData.components = componentCount > 0 ? new GSComponent[componentCount] : Array.Empty<GSComponent>();
                        sortPassData.splatScales = new float[componentCount];
                        sortPassData.colorSpaceModes = new int[componentCount];
                        sortPassData.debugPoints = new bool[componentCount];
                        sortPassData.debugPointSizes = new float[componentCount];
                        for (int i = 0; i < componentCount; i++)
                        {
                            sortPassData.components[i] = components[i];
                            sortPassData.splatScales[i] = components[i].splatScale;
                            sortPassData.colorSpaceModes[i] = (int)components[i].colorSpaceMode;
                            sortPassData.debugPoints[i] = components[i].showSplatCenters;
                            sortPassData.debugPointSizes[i] = components[i].centerPointSize;
                        }

                        sortBuilder.SetRenderFunc((SortPassData data, ComputeGraphContext context) => ExecuteSortPass(data, context));
                    }
                }

                // This adds a raster render pass to the graph, specifying the name and the data type that will be passed to the ExecutePass function.
                using (var builder = renderGraph.AddRasterRenderPass<PassData>(passName, out var passData))
                {
                    // Setup pass inputs and outputs through the builder interface.
                    // Eg:
                    // builder.UseTexture(sourceTexture);
                    // TextureHandle destination = UniversalRenderer.CreateRenderGraphTexture(renderGraph, cameraData.cameraTargetDescriptor, "Destination Texture", false);
                    // Prevent RenderGraph from culling this pass (splats must always render when components exist)
                    builder.AllowPassCulling(false);

                    // This sets the render target of the pass to the active color texture. Change it to your own render target as needed.
                    builder.SetRenderAttachment(resourceData.activeColorTexture, 0);
                    builder.SetRenderAttachmentDepth(resourceData.activeDepthTexture);

                    passData.material = settings.splatMaterial;
                    passData.splatSize = settings.splatSize;
                    passData.viewMatrix = viewMatrix;
                    passData.projMatrix = projMatrix;
                    passData.cameraWorldPos = cameraData.camera.transform.position;

                    int componentCount = components?.Count ?? 0;
                    passData.components = componentCount > 0 ? new GSComponent[componentCount] : Array.Empty<GSComponent>();
                    passData.showSplatCenters = new bool[componentCount];
                    passData.centerPointSizes = new float[componentCount];
                    for (int i = 0; i < componentCount; i++)
                    {
                        passData.components[i] = components[i];
                        passData.showSplatCenters[i] = components[i].showSplatCenters;
                        passData.centerPointSizes[i] = components[i].centerPointSize;
                    }

                    // Assigns the ExecutePass function to the render pass delegate. This will be called by the render graph when executing the pass.
                    builder.SetRenderFunc((PassData data, RasterGraphContext context) => ExecutePass(data, context));
                }
            }
        }
    }
}
