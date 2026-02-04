// Utility functions for Gaussian Splatting rendering
// Contains: quaternion math, covariance computation, projection utilities

// Partially inspired by: 
// Gaussian Splatting helper functions & structs
// most of these are from https://github.com/playcanvas/engine/tree/main/src/scene/shader-lib/glsl/chunks/gsplat
// Copyright (c) 2011-2024 PlayCanvas Ltd
// Copyright (c) 2025 Yize Wu
// SPDX-License-Identifier: MIT

#ifndef GS_UTILS_INCLUDED
#define GS_UTILS_INCLUDED

// ============================================================================
// Spherical Harmonics coefficients for view-dependent color computation
// SH coefficient computation inspired by aras-p/UnityGaussianSplatting
// https://github.com/aras-p/UnityGaussianSplatting
// SPDX-License-Identifier: MIT
// ============================================================================

// Spherical Harmonics coefficient for DC term (band 0)
// SH_C0 = 1 / (2 * sqrt(pi)) ≈ 0.28209479177387814
static const float SH_C0 = 0.28209479177387814;

// Spherical Harmonics coefficients for higher bands
// SH_C1 = sqrt(3) / (2 * sqrt(pi)) ≈ 0.4886025
static const float SH_C1 = 0.4886025;

// Band 2 coefficients
static const float SH_C2_0 = 1.0925484;   // sqrt(15) / (2 * sqrt(pi))
static const float SH_C2_1 = -1.0925484;  // -sqrt(15) / (2 * sqrt(pi))
static const float SH_C2_2 = 0.3153916;   // sqrt(5) / (4 * sqrt(pi))
static const float SH_C2_3 = -1.0925484;  // -sqrt(15) / (2 * sqrt(pi))
static const float SH_C2_4 = 0.5462742;   // sqrt(15) / (4 * sqrt(pi))

// Band 3 coefficients
static const float SH_C3_0 = -0.5900436;
static const float SH_C3_1 = 2.8906114;
static const float SH_C3_2 = -0.4570458;
static const float SH_C3_3 = 0.3731763;
static const float SH_C3_4 = -0.4570458;
static const float SH_C3_5 = 1.4453057;
static const float SH_C3_6 = -0.5900436;



// Convert quaternion (x,y,z,w) to 3x3 rotation matrix
// Unity/C# quaternion format: (x, y, z, w)
float3x3 QuatToRotationMatrix(float4 q)
{
    float x = q.x, y = q.y, z = q.z, w = q.w;
    
    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;

    return float3x3(
        1.0 - (yy + zz), xy - wz, xz + wy,
        xy + wz, 1.0 - (xx + zz), yz - wx,
        xz - wy, yz + wx, 1.0 - (xx + yy)
    );
}

// Compute 3D covariance matrix from rotation quaternion and scale
// Covariance: Σ = R * S * S^T * R^T = M * M^T where M = R * S
// Output: covA = (Σ00, Σ01, Σ02), covB = (Σ11, Σ12, Σ22)
void ComputeCovariance3D(float4 quat, float3 scale, out float3 covA, out float3 covB)
{
    float3x3 R = QuatToRotationMatrix(quat);
    
    // M = R * S (scale matrix is diagonal, so multiply columns by scale)
    float3x3 M = float3x3(
        R[0] * scale.x,
        R[1] * scale.y,
        R[2] * scale.z
    );
    
    // Cov = M * M^T (symmetric matrix, only 6 unique values)
    // For row-major M, (M * M^T)[i][j] = dot(row_i, row_j) = dot(M[i], M[j])
    covA = float3(
        dot(M[0], M[0]),  // Σ[0][0]
        dot(M[0], M[1]),  // Σ[0][1]
        dot(M[0], M[2])   // Σ[0][2]
    );
    covB = float3(
        dot(M[1], M[1]),  // Σ[1][1]
        dot(M[1], M[2]),  // Σ[1][2]
        dot(M[2], M[2])   // Σ[2][2]
    );
}

// Project 3D covariance to 2D screen space using EWA (Elliptical Weighted Average)
// Uses the Jacobian of perspective projection for accurate ellipse computation
// viewMatrix: world-to-view transformation matrix
// Returns: (cov2D[0][0], cov2D[0][1], cov2D[1][1])
float3 ProjectCovariance(float3 covA, float3 covB, float3 viewPos, float focal, float3x3 viewRotation)
{
    // Build symmetric 3D covariance matrix
    float3x3 cov3D = float3x3(
        covA.x, covA.y, covA.z,
        covA.y, covB.x, covB.y,
        covA.z, covB.y, covB.z
    );
    
    // Jacobian of perspective projection
    // J = | fx/z    0    -fx*x/z² |
    //     |   0   fy/z   -fy*y/z² |
    // Assumes fx = fy = focal
    float z = viewPos.z;
    float invZ = 1.0 / z;
    float invZ2 = invZ * invZ;
    
    float3x3 J = float3x3(
        focal * invZ, 0.0, -focal * viewPos.x * invZ2,
        0.0, focal * invZ, -focal * viewPos.y * invZ2,
        0.0, 0.0, 0.0
    );
    
    // Transform covariance: cov2D = J * W * cov3D * W^T * J^T
    // where W is the view rotation matrix
    float3x3 T = mul(J, viewRotation);
    float3x3 cov2D_full = mul(mul(T, cov3D), transpose(T));
    
    // Add low-pass filter for numerical stability (prevents aliasing)
    cov2D_full[0][0] += 0.3;
    cov2D_full[1][1] += 0.3;
    
    return float3(cov2D_full[0][0], cov2D_full[0][1], cov2D_full[1][1]);
}

// Compute eigenvalues and eigenvectors of 2D covariance for ellipse rendering
// Input: cov2D = (a, b, c) representing matrix | a  b |
//                                               | b  c |
// Returns: xy = major axis direction (normalized), zw = (major_radius, minor_radius) in pixels
float4 ComputeEllipseAxes(float3 cov2D)
{
    float a = cov2D.x;  // cov[0][0]
    float b = cov2D.y;  // cov[0][1]
    float c = cov2D.z;  // cov[1][1]
    
    // Eigenvalues of 2x2 symmetric matrix: λ = (trace ± sqrt(trace² - 4*det)) / 2
    float trace = a + c;
    float det = a * c - b * b;
    float discriminant = sqrt(max(0.25 * trace * trace - det, 0.0));
    
    float lambda1 = 0.5 * trace + discriminant;  // Larger eigenvalue
    float lambda2 = 0.5 * trace - discriminant;  // Smaller eigenvalue
    
    // Clamp to avoid numerical issues with very small splats
    lambda1 = max(lambda1, 0.01);
    lambda2 = max(lambda2, 0.01);
    
    // Eigenvector for larger eigenvalue
    float2 v1;
    if (abs(b) > 1e-6)
    {
        v1 = normalize(float2(lambda1 - c, b));
    }
    else
    {
        v1 = (a >= c) ? float2(1, 0) : float2(0, 1);
    }
    
    // Radii = sqrt(eigenvalues) * 3 to cover 3-sigma (99.7% of Gaussian mass)
    float r1 = 3.0 * sqrt(lambda1);
    float r2 = 3.0 * sqrt(lambda2);
    
    return float4(v1.x, v1.y, r1, r2);
}


// Convert SH DC coefficients to RGB color
// Standard formula: color = sh_dc * SH_C0 + 0.5
// The +0.5 shifts from [-0.5, 0.5] to [0, 1] for typical SH coefficient ranges
float3 SHToColor(float3 sh_dc)
{
    float3 rgb = sh_dc * SH_C0 + 0.5;
    return saturate(rgb);
}

// ============================================================================
// Spherical Harmonics view-dependent color computation
// Computes view-dependent color using SH bands 0-3
// Based on aras-p/UnityGaussianSplatting implementation
// https://github.com/aras-p/UnityGaussianSplatting
// SPDX-License-Identifier: MIT
// ============================================================================

// Evaluate spherical harmonics for view-dependent color
// Parameters:
//   sh_dc: DC term (band 0) coefficients (RGB)
//   shRest: Higher band coefficients (bands 1-3), stored as float array
//   shRestCount: Number of floats per splat in shRest (3 * (bands^2 - 1))
//   splatIndex: Index of the current splat
//   viewDir: World-space view direction from splat to camera (normalized)
//   shOrder: Number of SH bands to evaluate (0-3)
// Returns: Final RGB color
float3 EvaluateSH(
    float3 sh_dc,
    StructuredBuffer<float> shRest,
    int shRestCount,
    uint splatIndex,
    float3 viewDir,
    int shOrder)
{
    // Start with DC term (band 0): col = sh0 * SH_C0 + 0.5
    float3 result = sh_dc * SH_C0 + 0.5;
    
    // If no higher bands, return DC only
    if (shOrder < 1 || shRestCount <= 0)
    {
        return saturate(result);
    }
    
    // View direction components (negated as per convention)
    // The negation matches the 3DGS convention where we evaluate at -viewDir
    float3 dir = -viewDir;
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    
    // Base offset into shRest for this splat
    uint baseIdx = splatIndex * (uint)shRestCount;
    
    // Band 1 (3 coefficients per color channel = 9 floats total)
    // SH basis functions: Y_1^{-1} = y, Y_1^0 = z, Y_1^1 = x
    if (shOrder >= 1 && shRestCount >= 9)
    {
        // sh1, sh2, sh3 for RGB
        float3 sh1 = float3(shRest[baseIdx + 0], shRest[baseIdx + 1], shRest[baseIdx + 2]);
        float3 sh2 = float3(shRest[baseIdx + 3], shRest[baseIdx + 4], shRest[baseIdx + 5]);
        float3 sh3 = float3(shRest[baseIdx + 6], shRest[baseIdx + 7], shRest[baseIdx + 8]);
        
        result += SH_C1 * (-sh1 * y + sh2 * z - sh3 * x);

        
        // Band 2 (5 coefficients per color channel = 15 floats, total offset: 9)
        if (shOrder >= 2 && shRestCount >= 24)
        {
            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float xy = x * y;
            float yz = y * z;
            float xz = x * z;
            
            float3 sh4 = float3(shRest[baseIdx + 9], shRest[baseIdx + 10], shRest[baseIdx + 11]);
            float3 sh5 = float3(shRest[baseIdx + 12], shRest[baseIdx + 13], shRest[baseIdx + 14]);
            float3 sh6 = float3(shRest[baseIdx + 15], shRest[baseIdx + 16], shRest[baseIdx + 17]);
            float3 sh7 = float3(shRest[baseIdx + 18], shRest[baseIdx + 19], shRest[baseIdx + 20]);
            float3 sh8 = float3(shRest[baseIdx + 21], shRest[baseIdx + 22], shRest[baseIdx + 23]);
            
            result += (SH_C2_0 * xy) * sh4 +
                    (SH_C2_1 * yz) * sh5 +
                    (SH_C2_2 * (2.0 * zz - xx - yy)) * sh6 +
                    (SH_C2_3 * xz) * sh7 +
                    (SH_C2_4 * (xx - yy)) * sh8;

             
            // Band 3 (7 coefficients per color channel = 21 floats, total offset: 24)
            if (shOrder >= 3 && shRestCount >= 45)
            {
                float xx = x * x;
                float yy = y * y;
                float zz = z * z;
                
                float3 sh9  = float3(shRest[baseIdx + 24], shRest[baseIdx + 25], shRest[baseIdx + 26]);
                float3 sh10 = float3(shRest[baseIdx + 27], shRest[baseIdx + 28], shRest[baseIdx + 29]);
                float3 sh11 = float3(shRest[baseIdx + 30], shRest[baseIdx + 31], shRest[baseIdx + 32]);
                float3 sh12 = float3(shRest[baseIdx + 33], shRest[baseIdx + 34], shRest[baseIdx + 35]);
                float3 sh13 = float3(shRest[baseIdx + 36], shRest[baseIdx + 37], shRest[baseIdx + 38]);
                float3 sh14 = float3(shRest[baseIdx + 39], shRest[baseIdx + 40], shRest[baseIdx + 41]);
                float3 sh15 = float3(shRest[baseIdx + 42], shRest[baseIdx + 43], shRest[baseIdx + 44]);
                
                result += (SH_C3_0 * y * (3.0 * xx - yy)) * sh9 +
                        (SH_C3_1 * xy * z) * sh10 +
                        (SH_C3_2 * y * (4.0 * zz - xx - yy)) * sh11 +
                        (SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)) * sh12 +
                        (SH_C3_4 * x * (4.0 * zz - xx - yy)) * sh13 +
                        (SH_C3_5 * z * (xx - yy)) * sh14 +
                        (SH_C3_6 * x * (xx - 3.0 * yy)) * sh15;
            }
        }
    }
    
    return saturate(max(result, 0.0));
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
