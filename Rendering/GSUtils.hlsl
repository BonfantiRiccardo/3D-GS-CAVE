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

// Spherical Harmonics coefficient for DC term (band 0)
// SH_C0 = 1 / (2 * sqrt(pi)) ≈ 0.28209479177387814
static const float SH_C0 = 0.28209479177387814;



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
// Covariance: Σ = R * S * S^T * R^T = R * S² * R^T
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
