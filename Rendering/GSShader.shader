Shader "GaussianSplatting/GSShader"
{
    Properties
    {
        _SplatSize ("Splat Size", Float) = 0.02
    }
    // Gaussian Splatting Shader
    SubShader
    {
        // Render as transparent
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }
        Pass
        {
            Name "GSShader"
            Tags { "LightMode"="UniversalForward" }     // For URP compatibility

            Blend SrcAlpha OneMinusSrcAlpha             // Alpha blending
            ZWrite Off
            ZTest LEqual
            Cull Off

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma target 4.5                  // Ensure Shader Model 4.5 for StructuredBuffer support

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            StructuredBuffer<float3> _Positions;        // World space positions
            StructuredBuffer<float3> _SH;               // SH Direct Colors (f_dc_0..2)
            StructuredBuffer<float4> _Rotations;        // Rotations as quaternions
            StructuredBuffer<float3> _Scales;           // Scale factors

            float4x4 _ViewProj;
            float _SplatSize;                  // Global scale multiplier for per-splat size    
            int _SplatCount;

            struct Attributes                   // Vertex input structure
            {
                uint vertexID : SV_VertexID;
            };

            struct Varyings          // Vertex to fragment structure
            {
                float4 positionCS : SV_POSITION;
                float4 color : COLOR;
                float2 quadUV : TEXCOORD0;
            };

            // Rotate vector v by quaternion q
            float3 RotateByQuaternion(float3 v, float4 q)
            {
                float3 t = 2.0 * cross(q.xyz, v);
                return v + q.w * t + cross(q.xyz, t);
            }

            //--------------------------------------------------------------------- Vertex shader
            // Simple implementation generating quads for each splat (bounding box approach)
            Varyings Vert(Attributes input)
            {
                Varyings output;

                // Determine which splat and which vertex of the quad
                uint splatIndex = input.vertexID / 6;
                uint triVertex = input.vertexID % 6;

                float2 corners[6] = {       // Initialize quad corners for two triangles
                    float2(-1.0, -1.0),
                    float2( 1.0, -1.0),
                    float2( 1.0,  1.0),
                    float2(-1.0, -1.0),
                    float2( 1.0,  1.0),
                    float2(-1.0,  1.0)
                };

                // Fetch splat data
                float3 positionWS = _Positions[splatIndex];
                float2 corner = corners[triVertex];

                float4 rotation = _Rotations[splatIndex];
                float3 scale = _Scales[splatIndex] * _SplatSize;

                // Compute quad vertex position in world space
                float3 axisX = RotateByQuaternion(float3(1.0, 0.0, 0.0), rotation) * scale.x;
                float3 axisY = RotateByQuaternion(float3(0.0, 1.0, 0.0), rotation) * scale.y;

                // Offset from center
                float3 offsetWS = axisX * corner.x + axisY * corner.y;

                // Transform to clip space
                float4 worldPos = float4(positionWS + offsetWS, 1.0);
                output.positionCS = mul(_ViewProj, worldPos);       //multiply by view-projection matrix
                output.color = float4(_SH[splatIndex], 1.0);
                output.quadUV = corner;                             // Pass quad UV for fragment shader

                return output;
            }

            // --------------------------------------------------------------------------------- Fragment shader
            half4 Frag(Varyings input) : SV_Target
            {
                // Compute Gaussian alpha based on distance from center
                float radius2 = dot(input.quadUV, input.quadUV);     // Squared radius
                float alpha = exp(-0.5 * radius2);                   // Gaussian falloff
                half4 col = input.color;                            // FIX NEEDED: this is not alpha blending, as we are overwriting the color of the pixel every time a splat is drawn  
                col.a *= alpha;                                      // Modulate color based on distance from center
                return col;
            }
            ENDHLSL
        }
    }
}
