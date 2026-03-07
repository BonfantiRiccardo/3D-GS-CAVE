Shader "GaussianSplatting/GSShader"
{
    Properties
    {
    }

    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }
        
        Pass
        {
            Name "GSShader"
            Tags { "LightMode"="UniversalForward" }

            // Back-to-front blending (reverse compositing order)
            // Splats are rendered far-to-near
            Blend One OneMinusSrcAlpha
            ZWrite Off
            ZTest LEqual
            Cull Off

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma target 4.5
            #pragma require compute

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            // SplatViewData structure - must match compute shader definition
            // Precomputed per-splat rendering data for efficient quad expansion
            struct SplatViewData
            {
                float4 pos;             // Clip-space position
                float2 axis1;           // First ellipse axis (in pixels, includes scale)
                float2 axis2;           // Second ellipse axis (in pixels, includes scale)  
                uint2 color;            // Packed FP16 color: x = (r << 16) | g, y = (b << 16) | a
            };

            // Precomputed splat view data buffer (from compute shader)
            StructuredBuffer<SplatViewData> _SplatViewData;
            
            // Sorted order buffer - maps instance ID to original splat index
            StructuredBuffer<uint> _OrderBuffer;
            
            // Uniforms
            int _SplatCount;            // Total number of splats
            int _ShowSplatCenters;      // 1 = show debug center points, 0 = normal rendering
            float _CenterPointSize;    // Size of center points in UV space

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                half4 color : COLOR0;           // Unpacked color (rgb + alpha)
                float2 uv : TEXCOORD0;          // UV for Gaussian falloff
                nointerpolation uint debugMode : TEXCOORD1; // 1 = debug point, 0 = normal
            };

            // Shared: fetch splat data common to both rendering modes
            void FetchSplatData(uint instanceID, uint vertexID,
                out SplatViewData view, out half4 color, out float2 quadPos, out bool culled)
            {
                culled = false;
                color = (half4)0;
                quadPos = float2(0, 0);
                view = (SplatViewData)0;

                if (instanceID >= (uint)_SplatCount)
                {
                    culled = true;
                    return;
                }

                uint splatIdx = _OrderBuffer[instanceID];
                view = _SplatViewData[splatIdx];

                if (view.pos.w <= 0)
                {
                    culled = true;
                    return;
                }

                // Unpack FP16 color
                color.r = f16tof32(view.color.x >> 16);
                color.g = f16tof32(view.color.x);
                color.b = f16tof32(view.color.y >> 16);
                color.a = f16tof32(view.color.y);

                // Quad corners for 2 triangles (CCW winding)
                static const float2 quadCorners[6] = {
                    float2(-1, -1), float2(1, -1), float2(1, 1),
                    float2(-1, -1), float2(1, 1), float2(-1, 1)
                };
                quadPos = quadCorners[vertexID];
            }

            // Normal splat vertex: elliptical Gaussian quad
            Varyings VertSplat(SplatViewData view, half4 color, float2 quadPos)
            {
                Varyings output = (Varyings)0;
                output.color = color;
                output.debugMode = 0;

                // Scale quad by 2x to cover full Gaussian extent
                quadPos *= 2.0;
                output.uv = quadPos;

                // Compute screen-space offset using precomputed ellipse axes
                float2 deltaScreenPos = (quadPos.x * view.axis1 + quadPos.y * view.axis2) * 2.0 / _ScreenParams.xy;

                output.positionCS = view.pos;
                output.positionCS.xy += deltaScreenPos * view.pos.w;
                return output;
            }

            // Debug point vertex: uniform-size camera-facing square
            Varyings VertDebugPoint(SplatViewData view, half4 color, float2 quadPos)
            {
                Varyings output = (Varyings)0;
                output.color = color;
                output.debugMode = 1;
                output.uv = quadPos; // [-1, 1] range

                // Uniform square billboard using precomputed axes (already set to uniform size in compute)
                float2 deltaScreenPos = (quadPos.x * view.axis1 + quadPos.y * view.axis2) * 2.0 / _ScreenParams.xy;

                output.positionCS = view.pos;
                output.positionCS.xy += deltaScreenPos * view.pos.w;
                return output;
            }

            Varyings Vert(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
            {
                SplatViewData view;
                half4 color;
                float2 quadPos;
                bool culled;
                FetchSplatData(instanceID, vertexID, view, color, quadPos, culled);

                if (culled)
                {
                    Varyings output = (Varyings)0;
                    output.positionCS = asfloat(0x7fc00000);
                    return output;
                }

                if (_ShowSplatCenters)
                    return VertDebugPoint(view, color, quadPos);
                else
                    return VertSplat(view, color, quadPos);
            }

            half4 Frag(Varyings input) : SV_Target
            {   
                // Debug point mode: render as solid colored disc
                if (input.debugMode)
                {
                    float dist = dot(input.uv, input.uv);
                    if (dist > 1.0)
                        discard;
                    return half4(input.color.rgb, 1.0);
                }

                // Gaussian falloff: exp(-r^2) where r^2 = dot(uv, uv)
                // UV range is [-2, 2] so effective sigma coverage is good
                float power = -dot(input.uv, input.uv);
                half alpha = exp(power);
                
                // Apply splat opacity from precomputed color
                alpha = saturate(alpha * input.color.a);
                
                // Discard nearly transparent pixels (reduces overdraw)
                if (alpha < 1.0 / 255.0)
                    discard;

                // Output premultiplied alpha for correct compositing
                return half4(input.color.rgb * alpha, alpha);
            }
            ENDHLSL
        }
    }
}
