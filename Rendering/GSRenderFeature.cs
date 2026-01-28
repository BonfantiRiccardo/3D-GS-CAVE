using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace GaussianSplatting
{
    //This clas implements a ScriptableRendererFeature for Gaussian Splatting
    //It is used to set up and inject a render pass into the URP
    public class GSRenderFeature : ScriptableRendererFeature
    {
        // Settings for the render feature (example: material, splat size, etc.)
        [SerializeField] private GSRenderFeatureSettings settings = new();
        // The render pass that will be injected
        private GSRenderPass renderPass;

        /// <inheritdoc/>
        public override void Create()   //Called when the feature is created
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

        // Use this class to pass around settings from the feature to the pass
        [Serializable]
        public class GSRenderFeatureSettings
        {
            //Set default shader name for Gaussian Splatting
            public const string DefaultShaderName = "GaussianSplatting/GSShader";

            public bool enabled = true;     //Specifies if the render feature is enabled
            //Event to inject the render pass
            public RenderPassEvent renderPassEvent = RenderPassEvent.AfterRenderingTransparents;
            public Material splatMaterial;      //Material used for rendering splats
            public float splatSize = 1.0f;    //Global scale multiplier for per-splat size
        }

        // The actual render pass implementation
        private class GSRenderPass : ScriptableRenderPass
        {
            private readonly GSRenderFeatureSettings settings;  //Settings for the render pass

            // Constructor to initialize the render pass with settings
            public GSRenderPass(GSRenderFeatureSettings settings)
            {
                this.settings = settings;
            }

            // This class stores the data needed by the RenderGraph pass.
            // It is passed as a parameter to the delegate function that executes the RenderGraph pass.
            private class PassData
            {
                public Material material;
                public float splatSize;
                public Matrix4x4 viewProj;
                public GSComponent[] components;
            }

            // This static method is passed as the RenderFunc delegate to the RenderGraph render pass.
            // It is used to execute draw commands.
            private static void ExecutePass(PassData data, RasterGraphContext context)
            {
                if (data.material == null || data.components == null || data.components.Length == 0)
                {
                    return;
                }

                data.material.SetMatrix("_ViewProj", data.viewProj);
                data.material.SetFloat("_SplatSize", data.splatSize);

                for (int i = 0; i < data.components.Length; i++)
                {
                    GSComponent component = data.components[i];
                    if (component == null || !component.HasBuffers || component.ActiveSplatCount <= 0)
                    {
                        continue;
                    }

                    component.BindTo(data.material);
                    context.cmd.DrawProcedural(
                        Matrix4x4.identity,
                        data.material,
                        0,
                        MeshTopology.Triangles,
                        component.ActiveSplatCount * 6,
                        1);
                }
            }

            // RecordRenderGraph is where the RenderGraph handle can be accessed, through which render passes can be added to the graph.
            // FrameData is a context container through which URP resources can be accessed and managed.
            public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
            {
                if (settings == null || settings.splatMaterial == null)
                {
                    return;
                }

                var components = GSManager.Components;
                if (components == null || components.Count == 0)
                {
                    return;
                }

                // Use this scope to set the required inputs and outputs of the pass and to
                // setup the passData with the required properties needed at pass execution time.

                // Make use of frameData to access resources and camera data through the dedicated containers.
                // Eg:
                // UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();
                UniversalCameraData cameraData = frameData.Get<UniversalCameraData>();
                UniversalResourceData resourceData = frameData.Get<UniversalResourceData>();

                const string passName = "Gaussian Splat Pass";

                // This adds a raster render pass to the graph, specifying the name and the data type that will be passed to the ExecutePass function.
                using (var builder = renderGraph.AddRasterRenderPass<PassData>(passName, out var passData))
                {
                    // Setup pass inputs and outputs through the builder interface.
                    // Eg:
                    // builder.UseTexture(sourceTexture);
                    // TextureHandle destination = UniversalRenderer.CreateRenderGraphTexture(renderGraph, cameraData.cameraTargetDescriptor, "Destination Texture", false);

                    // This sets the render target of the pass to the active color texture. Change it to your own render target as needed.
                    builder.SetRenderAttachment(resourceData.activeColorTexture, 0);


                    passData.material = settings.splatMaterial;
                    passData.splatSize = settings.splatSize;
                    passData.viewProj = cameraData.camera.projectionMatrix * cameraData.camera.worldToCameraMatrix;

                    passData.components = new GSComponent[components.Count];
                    for (int i = 0; i < components.Count; i++)
                    {
                        passData.components[i] = components[i];
                    }

                    // Assigns the ExecutePass function to the render pass delegate. This will be called by the render graph when executing the pass.
                    builder.SetRenderFunc((PassData data, RasterGraphContext context) => ExecutePass(data, context));
                }
            }
        }
    }
}
