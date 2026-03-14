// Utility functions for Gaussian Splatting rendering
// Contains: quaternion math, covariance computation, projection utilities

// Partially inspired by: 
// Gaussian Splatting helper functions & structs
// most of these are from https://github.com/playcanvas/engine/tree/main/src/scene/shader-lib/glsl/chunks/gsplat
// Copyright (c) 2011-2024 PlayCanvas Ltd
// SPDX-License-Identifier: MIT

// SH coefficient computation inspired by aras-p/UnityGaussianSplatting
// https://github.com/aras-p/UnityGaussianSplatting
// SPDX-License-Identifier: MIT

#ifndef GS_UTILS_INCLUDED
#define GS_UTILS_INCLUDED

// Spherical Harmonics coefficients for view-dependent color computation
// Spherical Harmonics coefficient for DC term (band 0)
// SH_C0 = 1 / (2 * sqrt(pi)) ≈ 0.28209479177387814
static const float SH_C0 = 0.28209479177387814;

// higher bands
// SH_C1 = sqrt(3) / (2 * sqrt(pi)) ≈ 0.4886025
static const float SH_C1 = 0.4886025119029199;

// Band 2 coefficients
static const float SH_C2_0 = 1.0925484305920792;   // sqrt(15) / (2 * sqrt(pi))
static const float SH_C2_1 = -1.0925484305920792;  // -sqrt(15) / (2 * sqrt(pi))
static const float SH_C2_2 = 0.31539156525252005;   // sqrt(5) / (4 * sqrt(pi))
static const float SH_C2_3 = -1.0925484305920792;  // -sqrt(15) / (2 * sqrt(pi))
static const float SH_C2_4 = 0.5462742152960396;   // sqrt(15) / (4 * sqrt(pi))

// Band 3 coefficients
static const float SH_C3_0 = -0.5900435899266435;
static const float SH_C3_1 = 2.890611442640554;
static const float SH_C3_2 = -0.4570457994644658;
static const float SH_C3_3 = 0.3731763325901154;
static const float SH_C3_4 = -0.4570457994644658;
static const float SH_C3_5 = 1.445305721320277;
static const float SH_C3_6 = -0.5900435899266435;



// Convert quaternion (x,y,z,w) to 3x3 rotation matrix
// Unity/C# quaternion format: (x, y, z, w)
float3x3 QuatToRotationMatrix(float4 q)
{
    float x = q.x, y = q.y, z = q.z, w = q.w;

    return float3x3(
        1.0 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1.0 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1.0 - 2*(x*x + y*y)
    );
}

// Compute 3D covariance matrix from rotation quaternion and scale
// Covariance: Σ = R * S * S^T * R^T = M * M^T where M = R * S
// Output: covA = (Σ00, Σ01, Σ02), covB = (Σ11, Σ12, Σ22)
void ComputeCovariance3D(float4 quat, float3 scale, out float3 covA, out float3 covB)
{
    float3x3 R = QuatToRotationMatrix(quat);

    float3x3 S = float3x3(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );
    
    // M = R * S (scale matrix is diagonal, so multiply columns by scale)
    float3x3 M = mul(R, S);

    float3x3 sigma = mul(M, transpose(M));
    
    // Cov = M * M^T (symmetric matrix, only 6 unique values)
    covA = float3(sigma._m00,  // Σ[0][0]
                  sigma._m01,  // Σ[0][1]
                  sigma._m02); // Σ[0][2]
    covB = float3(sigma._m11,  // Σ[1][1]
                  sigma._m12,  // Σ[1][2]
                  sigma._m22); // Σ[2][2]
}

// Project 3D covariance to 2D screen space using EWA (Elliptical Weighted Average)
// Uses the Jacobian of perspective projection for accurate ellipse computation
// from "EWA Splatting" (Zwicker et al 2002) eq. 31
float3 CalcCovariance2D(float3 worldPos, float3 cov3d0, float3 cov3d1, float4x4 matrixV, float4x4 matrixP, float4 screenParams)
{
    float4x4 viewMatrix = matrixV;
    float3 viewPos = mul(viewMatrix, float4(worldPos, 1)).xyz;

    // this is needed in order for splats that are visible in view but clipped "quite a lot" to work
    float aspect = matrixP._m00 / matrixP._m11;
    float tanFovX = rcp(matrixP._m00);
    float tanFovY = rcp(matrixP._m11 * aspect);
    float limX = 1.3 * tanFovX;
    float limY = 1.3 * tanFovY;
    viewPos.x = clamp(viewPos.x / viewPos.z, -limX, limX) * viewPos.z;
    viewPos.y = clamp(viewPos.y / viewPos.z, -limY, limY) * viewPos.z;

    float focal = screenParams.x * matrixP._m00 / 2;

    float3x3 J = float3x3(
        focal / viewPos.z, 0, -(focal * viewPos.x) / (viewPos.z * viewPos.z),
        0, focal / viewPos.z, -(focal * viewPos.y) / (viewPos.z * viewPos.z),
        0, 0, 0
    );
    float3x3 W = (float3x3)viewMatrix;
    float3x3 T = mul(J, W);
    float3x3 V = float3x3(
        cov3d0.x, cov3d0.y, cov3d0.z,
        cov3d0.y, cov3d1.x, cov3d1.y,
        cov3d0.z, cov3d1.y, cov3d1.z
    );
    float3x3 cov = mul(T, mul(V, transpose(T)));

    // Low pass filter to make each splat at least 1px size.
    cov._m00 += 0.3;
    cov._m11 += 0.3;
    return float3(cov._m00, cov._m01, cov._m11);
}


// Convert SH DC coefficients to RGB color
// Standard formula: color = sh_dc * SH_C0 + 0.5
// The +0.5 shifts from [-0.5, 0.5] to [0, 1] for typical SH coefficient ranges
float3 SHToColor(float3 sh_dc)
{
    float3 rgb = sh_dc * SH_C0 + 0.5;
    return max(rgb, 0);
}

// Convert a single channel from sRGB (gamma ~2.2) to linear
// Uses the exact sRGB transfer function
float GammaToLinearChannel(float c)
{
    // sRGB transfer function:
    // if (c <= 0.04045) linear = c / 12.92
    // else linear = pow((c + 0.055) / 1.055, 2.4)
    return (c <= 0.04045) ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

// Convert RGB color from sRGB (gamma) to linear color space
float3 GammaToLinear(float3 color)
{
    return float3(
        GammaToLinearChannel(color.r),
        GammaToLinearChannel(color.g),
        GammaToLinearChannel(color.b)
    );
}


// Spherical Harmonics view-dependent color computation
// Based on aras-p/UnityGaussianSplatting implementation - https://github.com/aras-p/UnityGaussianSplatting
// SPDX-License-Identifier: MIT
//
// Multi-bank layout: SH rest data is split into three separate buffers (one per band) so that each stays under the 2 GB structured-buffer limit.
float3 EvaluateSH(
    float3 sh_dc,
    StructuredBuffer<float> shRest0,
    StructuredBuffer<float> shRest1,
    StructuredBuffer<float> shRest2,
    uint splatIndex,
    float3 viewDir,
    int shOrder)
{
    // Start with DC term (band 0): col = sh0 * SH_C0 + 0.5
    float3 result = sh_dc * SH_C0 + 0.5;

    if (shOrder < 1)
        return saturate(result);

    // View direction components (negated as per 3DGS convention)
    float3 dir = -viewDir;
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    // Band 1 (9 floats in shRest0)
    //Base offset for this splat's SH rest coefficients in the band 1 buffer
    uint b1 = splatIndex * 9u;
    float3 sh1 = float3(shRest0[b1 + 0], shRest0[b1 + 1], shRest0[b1 + 2]);
    float3 sh2 = float3(shRest0[b1 + 3], shRest0[b1 + 4], shRest0[b1 + 5]);
    float3 sh3 = float3(shRest0[b1 + 6], shRest0[b1 + 7], shRest0[b1 + 8]);
    result += SH_C1 * (-sh1 * y + sh2 * z - sh3 * x);

    // Band 2 (15 floats in shRest1)
    if (shOrder >= 2)
    {
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, yz = y * z, xz = x * z;

        uint b2 = splatIndex * 15u;
        float3 sh4 = float3(shRest1[b2 +  0], shRest1[b2 +  1], shRest1[b2 +  2]);
        float3 sh5 = float3(shRest1[b2 +  3], shRest1[b2 +  4], shRest1[b2 +  5]);
        float3 sh6 = float3(shRest1[b2 +  6], shRest1[b2 +  7], shRest1[b2 +  8]);
        float3 sh7 = float3(shRest1[b2 +  9], shRest1[b2 + 10], shRest1[b2 + 11]);
        float3 sh8 = float3(shRest1[b2 + 12], shRest1[b2 + 13], shRest1[b2 + 14]);

        result += (SH_C2_0 * xy) * sh4 +
                  (SH_C2_1 * yz) * sh5 +
                  (SH_C2_2 * (2.0 * zz - xx - yy)) * sh6 +
                  (SH_C2_3 * xz) * sh7 +
                  (SH_C2_4 * (xx - yy)) * sh8;

        // Band 3 (21 floats in shRest2)
        if (shOrder >= 3)
        {
            uint b3 = splatIndex * 21u;
            float3 sh9  = float3(shRest2[b3 +  0], shRest2[b3 +  1], shRest2[b3 +  2]);
            float3 sh10 = float3(shRest2[b3 +  3], shRest2[b3 +  4], shRest2[b3 +  5]);
            float3 sh11 = float3(shRest2[b3 +  6], shRest2[b3 +  7], shRest2[b3 +  8]);
            float3 sh12 = float3(shRest2[b3 +  9], shRest2[b3 + 10], shRest2[b3 + 11]);
            float3 sh13 = float3(shRest2[b3 + 12], shRest2[b3 + 13], shRest2[b3 + 14]);
            float3 sh14 = float3(shRest2[b3 + 15], shRest2[b3 + 16], shRest2[b3 + 17]);
            float3 sh15 = float3(shRest2[b3 + 18], shRest2[b3 + 19], shRest2[b3 + 20]);

            result += (SH_C3_0 * y * (3.0 * xx - yy)) * sh9 +
                      (SH_C3_1 * xy * z) * sh10 +
                      (SH_C3_2 * y * (4.0 * zz - xx - yy)) * sh11 +
                      (SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)) * sh12 +
                      (SH_C3_4 * x * (4.0 * zz - xx - yy)) * sh13 +
                      (SH_C3_5 * z * (xx - yy)) * sh14 +
                      (SH_C3_6 * x * (xx - 3.0 * yy)) * sh15;
        }
    }

    return max(result, 0);
}

// Alternative: direct color pass-through for PLY files with pre-computed colors
float3 DirectColor(float3 color)
{
    return saturate(color);
}

// Apply model transform to position (scale -> rotate -> translate)
float3 TransformPosition(float3 localPos, float3 modelPosition, float4 modelRotation, float3 modelScale)
{
    // Apply scale
    float3 scaled = localPos * modelScale;
    
    // Apply rotation
    float3x3 modelRot = QuatToRotationMatrix(modelRotation);
    float3 rotated = mul(modelRot, scaled);
    
    // Apply translation
    return rotated + modelPosition;
}

// Apply model rotation to splat rotation via quaternion multiplication
// Result = modelRotation * localQuat
float4 TransformRotation(float4 localQuat, float4 modelRotation)
{
    float4 q1 = modelRotation;
    float4 q2 = localQuat;
    
    return float4(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

#endif // GS_UTILS_INCLUDED
