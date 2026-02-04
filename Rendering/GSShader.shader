// Gaussian Splatting Shader - Depth-sorted rendering with proper 2D covariance projection
// Implements standard 3DGS projection: 3D covariance -> 2D screen-space ellipse
// Requires sorted indices for correct alpha blending (back-to-front)
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

            // Premultiplied alpha blending for correct compositing
            // Output = src.rgb * 1 + dst.rgb * (1 - src.a)
            // This requires shader to output (color.rgb * alpha, alpha)
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
            #include "GSUtils.hlsl"

            // Buffers
            StructuredBuffer<float3> _Positions;        // Local/world space positions
            StructuredBuffer<float4> _SH;               // SH0 colors (rgb) + opacity (a)
            StructuredBuffer<float> _SHRest;            // SH rest coefficients (float array)
            StructuredBuffer<float4> _Rotations;        // Rotations as quaternions (x,y,z,w) - Unity format
            StructuredBuffer<float3> _Scales;           // Scale factors (already exp() applied in import)
            StructuredBuffer<uint> _SortedIndices;      // Sorted splat indices (from GPU sort)
            
            // Uniforms
            float3 _ModelPosition;          // Model position offset
            float4 _ModelRotation;          // Model rotation quaternion (x,y,z,w)
            float3 _ModelScale;             // Model scale
            float _SplatSize;               // Global scale multiplier
            int _SplatCount;                // Total number of splats
            int _UseSortedIndices;          // Whether to use sorted indices
            int _SHRestCount;               // Number of SH rest floats per splat (3 * (bands^2 - 1))
            int _SHBands;                   // Number of SH bands (0-3)
            float3 _CameraWorldPos;         // Camera world position for view-dependent SH

            struct Attributes
            {
                uint vertexID : SV_VertexID;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float4 color : COLOR;           // rgb = SH DC color or view-dependent color, a = opacity
                float2 uv : TEXCOORD0;          // UV for Gaussian falloff (-1 to 1 in ellipse space)
                nointerpolation float3 worldPos : TEXCOORD1;    // Splat world position for view-dependent SH
                nointerpolation uint splatIndex : TEXCOORD2;    // Splat index for SH lookup
                nointerpolation uint sortedIndex : TEXCOORD3;   // Sorted index for debug coloring
            };



            Varyings Vert(Attributes input)
            {
                Varyings output;
                output.positionCS = float4(0, 0, 2, 1);  // Default: behind far plane (culled)
                output.color = float4(0, 0, 0, 0);
                output.uv = float2(0, 0);
                output.worldPos = float3(0, 0, 0);
                output.splatIndex = 0;
                output.sortedIndex = 0;

                // Decode vertex: 6 vertices per splat (2 triangles forming a quad)
                uint sortedIndex = input.vertexID / 6;
                uint cornerIndex = input.vertexID % 6;

                // Bounds check
                if (sortedIndex >= (uint)_SplatCount)
                    return output;

                // Get actual splat index (sorted or unsorted)
                uint splatIndex = sortedIndex;
                
                if (_UseSortedIndices != 0)
                {
                    splatIndex = _SortedIndices[sortedIndex];
                }

                // Quad corners for 2 triangles (CCW winding)
                static const float2 quadCorners[6] = {
                    float2(-1, -1), float2(1, -1), float2(1, 1),
                    float2(-1, -1), float2(1, 1), float2(-1, 1)
                };

                // Fetch splat data
                float3 localPos = _Positions[splatIndex];
                float4 localQuat = _Rotations[splatIndex];
                float3 localScale = _Scales[splatIndex];
                float4 shColor = _SH[splatIndex];

                // Clip quad size by alpha (reduces foggy tails from low-opacity splats)
                float alpha0 = shColor.a;
                if (alpha0 < 1.0 / 255.0)
                    return output;

                // Match reference: clip = min(1, sqrt(-log(1/255/alpha)) / 2)
                float clip = min(1.0, sqrt(-log((1.0 / 255.0) / max(alpha0, 1e-6))) * 0.5);
                float2 corner = quadCorners[cornerIndex] * clip;

                // Apply model transform
                float3 worldPos = TransformPosition(localPos, _ModelPosition, _ModelRotation, _ModelScale);
                float4 worldQuat = TransformRotation(localQuat, _ModelRotation);
                float3 worldScale = localScale * _ModelScale * _SplatSize;

                // Transform to view space
                float4 viewPos4 = mul(UNITY_MATRIX_V, float4(worldPos, 1.0));
                float3 viewPos = viewPos4.xyz;
                
                // Cull if behind camera (view space z should be negative)
                //if (viewPos.z >= -0.1)
                 //   return output;

                // Compute 3D covariance in world space
                float3 covA, covB;
                ComputeCovariance3D(worldQuat, worldScale, covA, covB);

                // Get focal length from projection matrix
                float focal = UNITY_MATRIX_P[0][0] * _ScreenParams.x * 0.5;

                // Extract view rotation
                float3x3 viewRotation = (float3x3)UNITY_MATRIX_V;

                // Project to 2D covariance
                float3 cov2D = ProjectCovariance(covA, covB, viewPos, focal, viewRotation);

                // Compute ellipse axes
                float4 ellipse = ComputeEllipseAxes(cov2D);
                float2 axis1 = ellipse.xy;
                float2 axis2 = float2(-axis1.y, axis1.x);
                float radius1 = ellipse.z;
                float radius2 = ellipse.w;

                // Skip if too small (< 1 pixel)
                if (radius1 < 1.0 || radius2 < 1.0)
                    return output;
                    
                // Limit maximum size
                float maxRadius = min(2048.0, min(_ScreenParams.x, _ScreenParams.y));
                radius1 = min(radius1, maxRadius);
                radius2 = min(radius2, maxRadius);

                // Compute screen-space offset
                float2 screenOffset = corner.x * axis1 * radius1 + corner.y * axis2 * radius2;

                // Project center to clip space
                float4 centerClip = mul(UNITY_MATRIX_P, viewPos4);
                
                // Convert offset from pixels to clip space
                float2 clipOffset = screenOffset * 2.0 * centerClip.w / _ScreenParams.xy;

                // Final clip position
                output.positionCS = centerClip + float4(clipOffset, 0, 0);

                // UV for fragment shader
                output.uv = corner;
                
                // Pass world position and splat index for fragment shader SH evaluation
                output.worldPos = worldPos;
                output.splatIndex = splatIndex;
                output.sortedIndex = sortedIndex;

                // Compute view direction in world space (from splat to camera)
                float3 viewDir = normalize(_CameraWorldPos - worldPos);
                
                // Evaluate spherical harmonics for view-dependent color
                // Uses SH bands based on available data
                float3 rgb;
                if (_SHBands > 0 && _SHRestCount > 0)
                {
                    // Full SH evaluation with higher bands
                    rgb = EvaluateSH(shColor.rgb, _SHRest, _SHRestCount, splatIndex, viewDir, _SHBands);
                }
                else
                {
                    // DC only (band 0)
                    rgb = SHToColor(shColor.rgb);
                }
                output.color = float4(rgb, shColor.a);

                return output;
            }

            half4 Frag(Varyings input) : SV_Target
            {   
                // Compute squared distance from center in ellipse space
                // UV spans [-1, 1] representing 3 standard deviations
                float r2 = dot(input.uv, input.uv);
                
                // Discard pixels outside the ellipse (beyond 3 sigma)
                if (r2 > 1.0)
                    discard;

                // Reference Gaussian falloff used by gsplat: exp(-A * r^2) with A=4.0
                float gaussian = exp(- 4.0 * r2);
                float alpha = saturate(gaussian * input.color.a);

                // Discard nearly transparent pixels
                if (alpha < 1.0 / 255.0)
                    discard;

                // Output premultiplied alpha
                return half4(input.color.rgb * alpha, alpha);

                // Debug: red color by sorted index
                //return half4(input.splatIndex/(float(_SplatCount)),0,0,1);

                /*
                // Debug
                float alpha = 1.0; // Force opaque
                return half4(input.color.rgb, alpha);*/
            }
            ENDHLSL
        }
    }
}
