using System;
using System.Collections.Generic;
using UnityEngine;

namespace GaussianSplatting
{
    /// <summary>
    /// A serializable, flat-array octree built over chunk AABBs at import time. Each internal node stores a 
    /// tight AABB enclosing all descendant chunks. Leaf nodes reference a contiguous range of chunk indices.
    /// Nodes are stored breadth-first in a flat array.
    /// Construction: top-down recursive subdivision. At each level, chunks whose AABB centre falls inside each 
    /// octant are assigned to that child.
    /// </summary>
    public static class ChunkOctree
    {
        /// <summary>Maximum chunks in a leaf before it is subdivided.</summary>
        public const int MaxLeafChunks = 16;

        /// <summary>Maximum recursion depth (8^10 theoretical nodes, but the tree is sparse).</summary>
        public const int MaxDepth = 10;

        /// <summary>
        /// Builds a flat-array octree from chunk metadata. Returns the serialisable node array.
        /// </summary>
        public static ChunkOctreeNode[] Build(ChunkInfo[] chunks, Bounds globalBounds)
        {
            if (chunks == null || chunks.Length == 0)
                return Array.Empty<ChunkOctreeNode>();

            // Collect all chunk indices
            var allIndices = new List<int>(chunks.Length);
            for (int i = 0; i < chunks.Length; i++)
                allIndices.Add(i);

            // Build tree recursively, collecting nodes in a list
            var nodes = new List<ChunkOctreeNode>();
            BuildNode(chunks, allIndices, globalBounds, 0, nodes);

            return nodes.ToArray();
        }

        /// <summary>
        /// Recursive top-down octree construction. Returns the index of the node it just created in <paramref name="nodes"/>.
        /// </summary>
        private static int BuildNode( ChunkInfo[] chunks, List<int> chunkIndices, Bounds bounds, int depth, List<ChunkOctreeNode> nodes)
        {
            // Reserve a slot for this node (we'll fill in children later)
            int thisIndex = nodes.Count;
            nodes.Add(default);

            // Compute tight bounds over the actual chunks (not the spatial subdivision cell)
            Bounds tight = ComputeTightBounds(chunks, chunkIndices);

            // Leaf condition: few chunks or max depth
            if (chunkIndices.Count <= MaxLeafChunks || depth >= MaxDepth)
            {
                nodes[thisIndex] = new ChunkOctreeNode
                {
                    bounds = tight,
                    firstChild = -1,            // leaf marker
                    chunkStart = -1,            // filled below
                    chunkCount = chunkIndices.Count,
                    leafChunkIndices = chunkIndices.ToArray()
                };
                return thisIndex;
            }

            // Subdivide into 8 octants
            Vector3 center = bounds.center;
            Vector3 halfSize = bounds.size * 0.5f;

            // Group chunks into octants by their AABB centre
            var childBins = new List<int>[8];
            for (int o = 0; o < 8; o++)
                childBins[o] = new List<int>();

            for (int i = 0; i < chunkIndices.Count; i++)
            {
                int ci = chunkIndices[i];
                Vector3 cc = chunks[ci].bounds.center;

                int octant = 0;
                if (cc.x >= center.x) octant |= 1;
                if (cc.y >= center.y) octant |= 2;
                if (cc.z >= center.z) octant |= 4;

                childBins[octant].Add(ci);
            }

            // If all chunks fall into a single octant (degenerate case), make a leaf
            int nonEmpty = 0;
            int singleOctant = -1;
            for (int o = 0; o < 8; o++)
            {
                if (childBins[o].Count > 0)
                {
                    nonEmpty++;
                    singleOctant = o;
                }
            }
            if (nonEmpty <= 1)
            {
                nodes[thisIndex] = new ChunkOctreeNode
                {
                    bounds = tight,
                    firstChild = -1,
                    chunkStart = -1,
                    chunkCount = chunkIndices.Count,
                    leafChunkIndices = chunkIndices.ToArray()
                };
                return thisIndex;
            }

            // Create children for non-empty octants
            int firstChildIdx = -1;
            var childIndices = new int[8];
            for (int o = 0; o < 8; o++)
                childIndices[o] = -1;

            for (int o = 0; o < 8; o++)
            {
                if (childBins[o].Count == 0)
                    continue;

                // Compute child spatial bounds
                Bounds childBounds = OctantBounds(center, halfSize, o);

                int childNodeIdx = BuildNode(chunks, childBins[o], childBounds, depth + 1, nodes);
                childIndices[o] = childNodeIdx;
                if (firstChildIdx < 0) firstChildIdx = childNodeIdx;
            }

            nodes[thisIndex] = new ChunkOctreeNode
            {
                bounds = tight,
                firstChild = firstChildIdx,
                chunkStart = -1,
                chunkCount = 0,
                childNodeIndices = childIndices,
                leafChunkIndices = null
            };

            return thisIndex;
        }

        /// <summary>
        /// Computes spatial bounds for an octant given the parent's centre and half-size.
        /// Octant numbering: bit 0 = (x >= centre), bit 1 = (y >= centre), bit 2 = (z >= centre).
        /// </summary>
        private static Bounds OctantBounds(Vector3 parentCenter, Vector3 halfSize, int octant)
        {
            Vector3 quarterSize = halfSize * 0.5f;
            Vector3 childCenter = parentCenter;
            childCenter.x += ((octant & 1) != 0) ? quarterSize.x : -quarterSize.x;
            childCenter.y += ((octant & 2) != 0) ? quarterSize.y : -quarterSize.y;
            childCenter.z += ((octant & 4) != 0) ? quarterSize.z : -quarterSize.z;
            return new Bounds(childCenter, halfSize); // half of parent size
        }

        /// <summary> Compute tight AABB enclosing all specified chunks. </summary>
        private static Bounds ComputeTightBounds(ChunkInfo[] chunks, List<int> indices)
        {
            if (indices.Count == 0)
                return new Bounds(Vector3.zero, Vector3.zero);

            Bounds b = chunks[indices[0]].bounds;
            for (int i = 1; i < indices.Count; i++)
                b.Encapsulate(chunks[indices[i]].bounds);
            return b;
        }

        // -----------------------------------------------------------------
        //  Runtime traversal

        /// <summary>
        /// Traverses the octree against frustum planes and collects indices of chunks whose node AABB intersects the 
        /// frustum. Uses a stack-based iterative traversal to avoid GC from recursion closures.
        /// </summary>
        /// <param name="nodes">Flat octree node array from the asset.</param>
        /// <param name="planes">6 frustum planes (expanded by desired margin).</param>
        /// <param name="modelMatrix">Model-to-world transform for the splat object.</param>
        /// <param name="result">Output list; cleared then filled with visible chunk indices.</param>
        public static void QueryFrustum(
            ChunkOctreeNode[] nodes,
            Plane[] planes,
            Matrix4x4 modelMatrix,
            List<int> result)
        {
            result.Clear();
            if (nodes == null || nodes.Length == 0) return;

            // Stack-based traversal
            var stack = new Stack<int>(64);
            stack.Push(0); // root

            while (stack.Count > 0)
            {
                int ni = stack.Pop();
                if (ni < 0 || ni >= nodes.Length) continue;

                ref ChunkOctreeNode node = ref nodes[ni];

                // Test node AABB against frustum
                Bounds worldBounds = ChunkInfo.TransformBoundsStatic(node.bounds, modelMatrix);
                if (!GeometryUtility.TestPlanesAABB(planes, worldBounds))
                    continue; // entire sub-tree culled

                // Leaf node: add chunk indices
                if (node.firstChild < 0)
                {
                    if (node.leafChunkIndices != null)
                    {
                        for (int i = 0; i < node.leafChunkIndices.Length; i++)
                            result.Add(node.leafChunkIndices[i]);
                    }
                    continue;
                }

                // Internal node: push children
                if (node.childNodeIndices != null)
                {
                    for (int o = 0; o < 8; o++)
                    {
                        if (node.childNodeIndices[o] >= 0)
                            stack.Push(node.childNodeIndices[o]);
                    }
                }
            }
        }
    }

    /// <summary> A single node in the flat-array octree. Serialised as part of ChunkedGSAsset.
    /// Internal nodes have childNodeIndices; leaf nodes have leafChunkIndices. </summary>
    [Serializable]
    public struct ChunkOctreeNode
    {
        /// <summary>Tight AABB enclosing all descendant chunks (local model space).</summary>
        public Bounds bounds;

        /// <summary>Index of the first child in the flat node array (-1 for leaf nodes).</summary>
        public int firstChild;

        /// <summary>Start index of chunks in this node (for leaf nodes).</summary>
        public int chunkStart;

        /// <summary>Number of chunks in a leaf node (0 for internal nodes).</summary>
        public int chunkCount;

        /// <summary>For internal nodes: indices into the node array for each of the 8 octants.
        /// -1 means that octant is empty. Null for leaf nodes.</summary>
        public int[] childNodeIndices;

        /// <summary>For leaf nodes: the chunk indices contained in this leaf. Null for internal nodes.</summary>
        public int[] leafChunkIndices;
    }
}
