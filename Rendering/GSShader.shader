// Gaussian Splatting Shader - Efficient instanced rendering with precomputed view data
// Uses SplatViewData buffer precomputed in compute shader for minimal vertex work
// Implements back-to-front rendering for correct alpha compositing
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

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                half4 color : COLOR0;           // Unpacked color (rgb + alpha)
                float2 uv : TEXCOORD0;          // UV for Gaussian falloff
            };

            Varyings Vert(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
            {
                Varyings output = (Varyings)0;
                
                // instanceID is in render order (0 = first to render = farthest)
                // vertexID is 0-5 for the 6 vertices of the quad (2 triangles)
                
                // Bounds check
                if (instanceID >= (uint)_SplatCount)
                {
                    output.positionCS = asfloat(0x7fc00000);  // NaN discards primitive
                    return output;
                }
                
                // Look up the original splat index from sorted order
                uint splatIdx = _OrderBuffer[instanceID];
                
                // Fetch precomputed view data for this splat
                SplatViewData view = _SplatViewData[splatIdx];
                float4 centerClipPos = view.pos;
                
                // Cull if behind camera or marked as culled (w <= 0)
                if (centerClipPos.w <= 0)
                {
                    output.positionCS = asfloat(0x7fc00000);  // NaN discards primitive
                    return output;
                }
                
                // Unpack FP16 color
                output.color.r = f16tof32(view.color.x >> 16);
                output.color.g = f16tof32(view.color.x);
                output.color.b = f16tof32(view.color.y >> 16);
                output.color.a = f16tof32(view.color.y);
                
                // Quad corners for 2 triangles (CCW winding)
                // Triangle 1: 0,1,2  Triangle 2: 3,4,5
                static const float2 quadCorners[6] = {
                    float2(-1, -1), float2(1, -1), float2(1, 1),   // First triangle
                    float2(-1, -1), float2(1, 1), float2(-1, 1)    // Second triangle
                };
                
                float2 quadPos = quadCorners[vertexID];
                
                // Scale quad by 2x to cover full Gaussian extent
                // (axis already includes sqrt(2*lambda))
                quadPos *= 2.0;
                
                // Store UV for fragment shader Gaussian evaluation
                output.uv = quadPos;
                
                // Compute screen-space offset using precomputed ellipse axes
                // Axes are in pixels, convert to normalized device coordinates
                float2 deltaScreenPos = (quadPos.x * view.axis1 + quadPos.y * view.axis2) * 2.0 / _ScreenParams.xy;
                
                // Final clip position = center + offset * w (perspective-correct)
                output.positionCS = centerClipPos;
                output.positionCS.xy += deltaScreenPos * centerClipPos.w;
                
                return output;
            }

            half4 Frag(Varyings input) : SV_Target
            {   
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

                // Debug: red color by sorted index
                //return half4(input.splatIndex/(float(_SplatCount)),0,0,1);

                //Debug: only draw the splat centers
                //if (length(input.uv) > 0.02)
                //    discard;

                //return half4(1,0,0,1);

                /*
                // Debug
                float alpha = 1.0; // Force opaque
                return half4(input.color.rgb, alpha);*/
            }
            ENDHLSL
        }
    }
}
