using System;
using UnityEngine;

namespace GaussianSplatting
{
    /// <summary>
    /// ChunkInfo contains metadata for a single spatial chunk of Gaussian splats.
    /// Chunks are used for frustum culling and streaming, allowing the renderer to load only visible portions of very large splat scenes.
    /// </summary>
    [Serializable]
    public struct ChunkInfo
    {
        /// <summary>
        /// Axis aligned bounding box for this chunk in local model space. Used for frustum culling tests.
        /// </summary>
        public Bounds bounds;

        /// <summary> Number of splats contained in this chunk. </summary>
        public int splatCount;

        /// <summary>
        /// Starting index of this chunk's splats in the global sorted splat array.
        /// Splats for this chunk span indices [startIndex, startIndex + splatCount).
        /// </summary>
        public int startIndex;

        /// <summary>
        /// Byte offset into each data file where this chunk's data begins.
        /// For positions.bytes: offset = startIndex * 12
        /// For rotations.bytes: offset = startIndex * 16 ...
        /// </summary>
        public long dataOffset;

        /// <summary>
        /// Checks if this chunk intersects with a set of frustum planes.
        /// </summary>
        /// <param name="frustumPlanes">The six frustum planes from the camera.</param>
        /// <param name="modelMatrix">The model transformation matrix.</param>
        /// <returns>True if the chunk is potentially visible.</returns>
        public bool IsVisibleInFrustum(Plane[] frustumPlanes, Matrix4x4 modelMatrix)
        {
            // Transform the bounds to world space
            Bounds worldBounds = TransformBounds(bounds, modelMatrix);
            return GeometryUtility.TestPlanesAABB(frustumPlanes, worldBounds);
        }

        /// <summary> Static version of bounds transformation for use in contexts where instance methods are not available. </summary>
        public static Bounds TransformBoundsStatic(Bounds localBounds, Matrix4x4 matrix)
            => TransformBounds(localBounds, matrix);

        /// <summary>
        /// Transforms an AABB by a matrix, computing a new AABB that encloses
        /// the transformed original.
        /// </summary>
        private static Bounds TransformBounds(Bounds localBounds, Matrix4x4 matrix)
        {
            Vector3 center = matrix.MultiplyPoint3x4(localBounds.center);
            Vector3 extents = localBounds.extents;

            // Transform the extents by the absolute values of the matrix axes
            // This computes the tightest AABB around the rotated/scaled box
            Vector3 axisX = matrix.GetColumn(0);
            Vector3 axisY = matrix.GetColumn(1);
            Vector3 axisZ = matrix.GetColumn(2);

            Vector3 newExtents = new Vector3(
                Mathf.Abs(axisX.x) * extents.x + Mathf.Abs(axisY.x) * extents.y + Mathf.Abs(axisZ.x) * extents.z,
                Mathf.Abs(axisX.y) * extents.x + Mathf.Abs(axisY.y) * extents.y + Mathf.Abs(axisZ.y) * extents.z,
                Mathf.Abs(axisX.z) * extents.x + Mathf.Abs(axisY.z) * extents.y + Mathf.Abs(axisZ.z) * extents.z
            );

            return new Bounds(center, newExtents * 2f);
        }
    }
}
