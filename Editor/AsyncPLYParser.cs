using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    /// <summary>
    /// Async PLY parser that produces GSImportData. Uses chunked I/O and background threading to avoid blocking Unity's 
    /// main thread, enabling import of very large PLY files.
    /// Performance optimizations:
    /// 1. Property lookup table: all property name resolution happens once during header parsing.
    /// 2. Multi-threaded batch parsing: after sequential I/O fills the buffer, vertex parsing is split across available CPU cores using Parallel.For.
    /// 3. Buffered sequential I/O: 16 MB read buffer with SequentialScan hint.
    ///
    /// Flexibility:
    /// 1. Multiple property name conventions (f_dc_0/red/r, rot_0/qw, scale_0/scale_x, ...)
    /// 2. Type-aware conversion (SH colors can be stored as uchars or floats)
    /// 3. Graceful handling of missing properties (identity rotation, unit scale, neutral color, full opacity)
    /// 4. SH import quality levels: Full, Reduced (Band 1), None (DC only)
    ///
    /// SH quality level concept inspired by aras-p/UnityGaussianSplatting.
    /// </summary>
    public static class AsyncPLYParser
    {
        // 16 MB I/O buffer for sequential reads
        private const int IO_BUFFER_SIZE = 16 * 1024 * 1024;

        // Progress report interval: every N vertices
        private const int PROGRESS_INTERVAL = 100_000;

        // Minimum batch size to warrant multi-threaded parsing
        private const int PARALLEL_THRESHOLD = 10_000;

        // SH Band 0 constant for color ↔ SH DC conversion
        private const float SH_C0 = 0.2820948f;

        #region Internal Types

        private enum PLYFormat { ASCII, BinaryLittleEndian, BinaryBigEndian }

        /// <summary>
        /// Internal PLY data types used during parsing.
        /// </summary>
        private enum PLYType { Int8, UInt8, Int16, UInt16, Int32, UInt32, Float32, Float64 }

        /// <summary>
        /// Represents a property in the PLY file header.
        /// </summary>
        private struct PLYProperty
        {
            public string Name;
            public PLYType Type;
            public int ByteSize => Type switch
            {
                PLYType.Int8    => 1, PLYType.UInt8   => 1,
                PLYType.Int16   => 2, PLYType.UInt16  => 2,
                PLYType.Int32   => 4, PLYType.UInt32  => 4,
                PLYType.Float32 => 4, PLYType.Float64 => 8,
                _ => 0
            };
        }

        /// <summary>
        /// Describes a single field's byte offset and data type within a vertex record.
        /// Built once during header parse, read directly in the inner loop.
        /// </summary>
        private struct FieldInfo
        {
            public int Offset;   // Byte offset within vertex record (-1 = not present)
            public PLYType Type;

            public bool IsPresent => Offset >= 0;

            public static FieldInfo Missing => new FieldInfo { Offset = -1, Type = PLYType.Float32 };
        }

        /// <summary>
        /// Precomputed layout of all interesting fields in the vertex record.
        /// Built once during header parse from property name resolution.
        /// </summary>
        private sealed class PropertyLayout
        {
            // Position
            public FieldInfo PosX, PosY, PosZ;

            // Rotation: two conventions supported
            public FieldInfo Rot0, Rot1, Rot2, Rot3;     // WXYZ (3DGS standard: rot_0=W)
            public FieldInfo QuatX, QuatY, QuatZ, QuatW;  // XYZW convention

            // Scale
            public FieldInfo ScaleX, ScaleY, ScaleZ;

            // Opacity
            public FieldInfo Opacity;

            // Color (SH DC coefficients or direct RGB)
            public FieldInfo ColorR, ColorG, ColorB;

            // SH Rest coefficients (indexed by output rest index)
            public FieldInfo[] RestFields;
            public int RestCount;

            // Data presence flags
            public bool HasRotation;
            public bool HasScale;
            public bool HasColor;
            public bool HasOpacity;
            public bool HasSHRest;

            // Conversion flags (determined from property names and types)
            public bool HasDirectColors;    // Colors are RGB (not SH DC), need conversion
            public bool OpacityNeedsSigmoid; // Standard 3DGS: float opacity needs sigmoid activation
            public bool ScaleNeedsExp;       // Standard 3DGS: float scale needs exp activation
        }

        /// <summary> Represents the parsed PLY header information needed for binary parsing. </summary>
        private struct PLYHeader
        {
            public int VertexCount;
            public PLYFormat Format;
            public long DataStartOffset;
            public int BytesPerVertex;
            public PropertyLayout Layout;
        }

        /// <summary>Thread-local bounds accumulator for multi-threaded parsing.</summary>
        private sealed class ThreadBounds
        {
            public Vector3 Min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            public Vector3 Max = new Vector3(float.MinValue, float.MinValue, float.MinValue);
        }

        #endregion

        #region Public API

        /// <summary>
        /// Parses a PLY file asynchronously on a background thread.
        /// </summary>
        public static Task<GSImportData> ParseAsync(
            string path,
            IProgress<float> progress = null,
            CancellationToken cancellationToken = default,
            SHImportQuality shQuality = SHImportQuality.Full,
            CoordinateFlip flip = CoordinateFlip.None)
        {
            return Task.Run(() => ParseInternal(path, progress, cancellationToken, shQuality, flip), cancellationToken);
        }

        /// <summary>
        /// Synchronous parse entry point (for use in ScriptedImporter).
        /// </summary>
        public static GSImportData Parse(
            string path,
            IProgress<float> progress = null,
            SHImportQuality shQuality = SHImportQuality.Full,
            CoordinateFlip flip = CoordinateFlip.None)
        {
            return ParseInternal(path, progress, CancellationToken.None, shQuality, flip);
        }

        #endregion

        #region Core Implementation

        /// <summary> Core PLY parsing implementation. Called by both async and sync entry points. </summary>
        private static GSImportData ParseInternal(
            string path,
            IProgress<float> progress,
            CancellationToken ct,
            SHImportQuality shQuality,
            CoordinateFlip flip)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"PLY file not found: {path}");

            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read,
                FileShare.Read, IO_BUFFER_SIZE, FileOptions.SequentialScan);

            // Step 1: Parse header
            progress?.Report(0f);

            PLYHeader header = ReadHeader(stream, shQuality);

            if (header.Format != PLYFormat.BinaryLittleEndian)
                throw new NotSupportedException("Only binary little-endian PLY is supported.");

            // Step 2: Parse binary vertex data using buffered reading
            return ParseBinaryBuffered(stream, header, progress, ct, shQuality, flip);
        }

        #endregion

        #region Header Parsing

        /// <summary>
        /// Reads the PLY header from the stream.
        /// </summary>
        private static PLYHeader ReadHeader(Stream stream, SHImportQuality shQuality)
        {
            int vertexCount = 0;
            bool readingVertexProps = false;
            var properties = new List<PLYProperty>();
            PLYFormat format = PLYFormat.BinaryLittleEndian;

            int headerBytes = 0;
            var lineBuffer = new List<byte>(256);

            while (true)
            {
                int b = stream.ReadByte();
                if (b < 0)
                    throw new InvalidDataException("PLY header not found (unexpected end of file).");

                headerBytes++;
                if (b == '\n')
                {
                    string line = System.Text.Encoding.ASCII.GetString(lineBuffer.ToArray()).TrimEnd('\r');
                    lineBuffer.Clear();

                    if (headerBytes <= 4 && !line.Equals("ply", StringComparison.OrdinalIgnoreCase))
                    {
                        if (line != "ply")
                            throw new InvalidDataException("File is not a valid PLY file (missing 'ply').");
                    }

                    if (line.StartsWith("format ", StringComparison.OrdinalIgnoreCase))
                    {
                        if (line.IndexOf("binary_little_endian", StringComparison.OrdinalIgnoreCase) >= 0)
                            format = PLYFormat.BinaryLittleEndian;
                        else if (line.IndexOf("binary_big_endian", StringComparison.OrdinalIgnoreCase) >= 0)
                            throw new NotSupportedException("Binary big-endian PLY is not supported.");
                        else if (line.IndexOf("ascii", StringComparison.OrdinalIgnoreCase) >= 0)
                            throw new NotSupportedException("ASCII PLY is not supported. Use binary little-endian.");
                        else
                            throw new NotSupportedException($"Unknown PLY format: {line}");
                    }
                    else if (line.StartsWith("element ", StringComparison.OrdinalIgnoreCase))
                    {
                        readingVertexProps = false;
                        string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length >= 3 && parts[1].Equals("vertex", StringComparison.OrdinalIgnoreCase))
                        {
                            int.TryParse(parts[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out vertexCount);
                            readingVertexProps = true;
                        }
                    }
                    else if (readingVertexProps && line.StartsWith("property ", StringComparison.OrdinalIgnoreCase))
                    {
                        string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length >= 3)
                        {
                            if (parts[1].Equals("list", StringComparison.OrdinalIgnoreCase))
                                throw new NotSupportedException("PLY list properties are not supported.");

                            properties.Add(new PLYProperty
                            {
                                Type = ParseType(parts[1]),
                                Name = parts[2]
                            });
                        }
                    }
                    else if (line.Equals("end_header", StringComparison.OrdinalIgnoreCase))
                    {
                        break;
                    }
                }
                else
                {
                    lineBuffer.Add((byte)b);
                }
            }

            // Calculate bytes per vertex
            int bytesPerVertex = 0;
            foreach (var prop in properties)
                bytesPerVertex += prop.ByteSize;

            // Build property layout: this resolves ALL property names to byte offsets.
            // After this point, no string operations occur during vertex parsing.
            PropertyLayout layout = BuildPropertyLayout(properties, shQuality);

            return new PLYHeader
            {
                VertexCount = vertexCount,
                Format = format,
                DataStartOffset = headerBytes,
                BytesPerVertex = bytesPerVertex,
                Layout = layout
            };
        }

        #endregion

        #region Property Layout Builder

        /// <summary>
        /// Resolves property names to byte offsets and determines conversion flags.
        /// Allows flexible property naming conventions and graceful handling of missing data.
        /// </summary>
        private static PropertyLayout BuildPropertyLayout(List<PLYProperty> properties, SHImportQuality shQuality)
        {
            var layout = new PropertyLayout
            {
                PosX = FieldInfo.Missing, PosY = FieldInfo.Missing, PosZ = FieldInfo.Missing,
                Rot0 = FieldInfo.Missing, Rot1 = FieldInfo.Missing, Rot2 = FieldInfo.Missing, Rot3 = FieldInfo.Missing,
                QuatX = FieldInfo.Missing, QuatY = FieldInfo.Missing, QuatZ = FieldInfo.Missing, QuatW = FieldInfo.Missing,
                ScaleX = FieldInfo.Missing, ScaleY = FieldInfo.Missing, ScaleZ = FieldInfo.Missing,
                Opacity = FieldInfo.Missing,
                ColorR = FieldInfo.Missing, ColorG = FieldInfo.Missing, ColorB = FieldInfo.Missing,
            };

            bool foundDirectColor = false;
            bool foundSHDC = false;

            // Pass 1: compute byte offsets and discover SH rest indices
            int offset = 0;
            var restIndexMap = new Dictionary<int, FieldInfo>();
            int maxRestIndex = -1;

            for (int i = 0; i < properties.Count; i++)
            {
                var prop = properties[i];
                var field = new FieldInfo { Offset = offset, Type = prop.Type };
                string name = prop.Name.ToLowerInvariant();

                switch (name)
                {
                    // Position
                    case "x": layout.PosX = field; break;
                    case "y": layout.PosY = field; break;
                    case "z": layout.PosZ = field; break;

                    // Rotation WXYZ (3DGS standard: rot_0 = W)
                    case "rot_0": layout.Rot0 = field; break;
                    case "rot_1": layout.Rot1 = field; break;
                    case "rot_2": layout.Rot2 = field; break;
                    case "rot_3": layout.Rot3 = field; break;

                    // Rotation XYZW (alternative convention)
                    case "qx": layout.QuatX = field; break;
                    case "qy": layout.QuatY = field; break;
                    case "qz": layout.QuatZ = field; break;
                    case "qw": layout.QuatW = field; break;

                    // Scale
                    case "scale_0": case "scale_x": layout.ScaleX = field; break;
                    case "scale_1": case "scale_y": layout.ScaleY = field; break;
                    case "scale_2": case "scale_z": layout.ScaleZ = field; break;

                    // Opacity
                    case "opacity": case "alpha": layout.Opacity = field; break;

                    // Color: SH DC takes priority over direct RGB
                    case "f_dc_0":
                        layout.ColorR = field; foundSHDC = true; break;
                    case "f_dc_1":
                        layout.ColorG = field; foundSHDC = true; break;
                    case "f_dc_2":
                        layout.ColorB = field; foundSHDC = true; break;

                    // Direct color: only used if no f_dc_* was found
                    case "red": case "r":
                        if (!foundSHDC) { layout.ColorR = field; foundDirectColor = true; }
                        break;
                    case "green": case "g":
                        if (!foundSHDC) { layout.ColorG = field; foundDirectColor = true; }
                        break;
                    case "blue": case "b":
                        if (!foundSHDC) { layout.ColorB = field; foundDirectColor = true; }
                        break;

                    default:
                        // Check for SH rest coefficients: f_rest_N
                        if (name.StartsWith("f_rest_"))
                        {
                            string suffix = name.Substring("f_rest_".Length);
                            if (int.TryParse(suffix, NumberStyles.Integer, CultureInfo.InvariantCulture, out int restIdx) && restIdx >= 0)
                            {
                                restIndexMap[restIdx] = field;
                                if (restIdx > maxRestIndex)
                                    maxRestIndex = restIdx;
                            }
                        }
                        break;
                }

                offset += prop.ByteSize;
            }

            // Data presence flags
            layout.HasRotation = layout.Rot0.IsPresent || layout.QuatX.IsPresent;
            layout.HasScale = layout.ScaleX.IsPresent && layout.ScaleY.IsPresent && layout.ScaleZ.IsPresent;
            layout.HasColor = layout.ColorR.IsPresent || layout.ColorG.IsPresent || layout.ColorB.IsPresent;
            layout.HasOpacity = layout.Opacity.IsPresent;
            layout.HasDirectColors = foundDirectColor && !foundSHDC;

            // Determine activation functions based on property types:
            // Standard 3DGS stores opacity as pre-sigmoid float and scale as log-encoded float.
            // Non-3DGS files (e.g., plain point clouds) may have integer types that are already linear.
            layout.OpacityNeedsSigmoid = layout.HasOpacity && IsFloatType(layout.Opacity.Type);
            layout.ScaleNeedsExp = layout.HasScale && IsFloatType(layout.ScaleX.Type);

            // Build SH rest field mapping based on requested quality level
            BuildRestLayout(layout, restIndexMap, maxRestIndex, shQuality);

            return layout;
        }

        /// <summary>
        /// Builds the SH rest coefficient mapping based on the requested quality level.
        /// In standard 3DGS PLY files, SH rest coefficients are stored grouped by channel:
        /// </summary>
        private static void BuildRestLayout(
            PropertyLayout layout,
            Dictionary<int, FieldInfo> restIndexMap,
            int maxRestIndex,
            SHImportQuality shQuality)
        {
            int fullRestCount = maxRestIndex + 1;

            if (shQuality == SHImportQuality.None || fullRestCount <= 0)
            {
                layout.RestFields = Array.Empty<FieldInfo>();
                layout.RestCount = 0;
                layout.HasSHRest = false;
                return;
            }

            if (shQuality == SHImportQuality.Full)
            {
                // Import all rest coefficients
                layout.RestFields = new FieldInfo[fullRestCount];
                for (int r = 0; r < fullRestCount; r++)
                    layout.RestFields[r] = restIndexMap.ContainsKey(r) ? restIndexMap[r] : FieldInfo.Missing;
                layout.RestCount = fullRestCount;
                layout.HasSHRest = true;
                return;
            }

            // Reduced: Band 1 only (3 coefficients per channel)
            int coeffsPerChannel = fullRestCount >= 3 ? fullRestCount / 3 : fullRestCount;
            int band1PerChannel = Math.Min(3, coeffsPerChannel);
            int effectiveCount = coeffsPerChannel >= 1 ? band1PerChannel * 3 : 0;

            layout.RestFields = new FieldInfo[effectiveCount];
            int outIdx = 0;
            for (int ch = 0; ch < 3 && outIdx < effectiveCount; ch++)
            {
                for (int c = 0; c < band1PerChannel && outIdx < effectiveCount; c++)
                {
                    int plyRestIdx = ch * coeffsPerChannel + c;
                    layout.RestFields[outIdx] = restIndexMap.ContainsKey(plyRestIdx)
                        ? restIndexMap[plyRestIdx]
                        : FieldInfo.Missing;
                    outIdx++;
                }
            }

            layout.RestCount = effectiveCount;
            layout.HasSHRest = effectiveCount > 0;
        }

        private static bool IsFloatType(PLYType type)
        {
            return type == PLYType.Float32 || type == PLYType.Float64;
        }

        #endregion

        #region Binary Data Parsing (Buffered + Multi-threaded)

        /// <summary>
        /// Parses binary vertex data using buffered I/O and multi-threaded batch processing.
        ///
        /// Strategy:
        /// 1. Read a batch of raw bytes from disk (sequential I/O, 16 MB buffer)
        /// 2. If batch is large enough, split vertex parsing across CPU cores via Parallel.For
        /// 3. Each thread accumulates local bounds; merged after each batch
        /// 4. Vertex parsing uses precomputed byte offsets
        /// </summary>
        private static GSImportData ParseBinaryBuffered(
            Stream stream,
            PLYHeader header,
            IProgress<float> progress,
            CancellationToken ct,
            SHImportQuality shQuality,
            CoordinateFlip flip)
        {
            int vertexCount = header.VertexCount;
            int bytesPerVertex = header.BytesPerVertex;
            PropertyLayout layout = header.Layout;

            if (stream.Position != header.DataStartOffset)
                stream.Position = header.DataStartOffset;

            // Pre-allocate output arrays
            var positions = new Vector3[vertexCount];
            var rotations = new quaternion[vertexCount];
            var scales = new Vector3[vertexCount];
            var sh = new Vector4[vertexCount];
            float[] shRest = layout.RestCount > 0
                ? new float[vertexCount * layout.RestCount]
                : Array.Empty<float>();

            Vector3 globalMin = new(float.MaxValue, float.MaxValue, float.MaxValue);
            Vector3 globalMax = new(float.MinValue, float.MinValue, float.MinValue);

            // I/O buffer: read as many complete vertices as fit
            int verticesPerBuffer = Math.Max(1, IO_BUFFER_SIZE / bytesPerVertex);
            byte[] buffer = new byte[verticesPerBuffer * bytesPerVertex];

            int vertexIndex = 0;
            int nextProgressVertex = PROGRESS_INTERVAL;
            object boundsLock = new object();

            var parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = Environment.ProcessorCount,
                CancellationToken = ct
            };

            while (vertexIndex < vertexCount)
            {
                ct.ThrowIfCancellationRequested();

                // How many vertices to read in this batch?
                int batchCount = Math.Min(verticesPerBuffer, vertexCount - vertexIndex);
                int batchBytes = batchCount * bytesPerVertex;

                // Read the full batch from disk (sequential I/O)
                int bytesRead = ReadExact(stream, buffer, batchBytes);
                if (bytesRead < batchBytes)
                    throw new IOException($"Unexpected end of PLY data at vertex {vertexIndex}.");

                int batchStart = vertexIndex;

                if (batchCount >= PARALLEL_THRESHOLD)
                {
                    // Multi-threaded vertex parsing
                    Parallel.For<ThreadBounds>(
                        0, batchCount, parallelOptions,
                        () => new ThreadBounds(),
                        (b, _, localBounds) =>
                        {
                            int vi = batchStart + b;
                            ParseVertex(buffer, b, vi, bytesPerVertex, layout, flip,
                                positions, rotations, scales, sh, shRest);
                            localBounds.Min = Vector3.Min(localBounds.Min, positions[vi]);
                            localBounds.Max = Vector3.Max(localBounds.Max, positions[vi]);
                            return localBounds;
                        },
                        localBounds =>
                        {
                            lock (boundsLock)
                            {
                                globalMin = Vector3.Min(globalMin, localBounds.Min);
                                globalMax = Vector3.Max(globalMax, localBounds.Max);
                            }
                        }
                    );
                }
                else
                {
                    // Single-threaded for small batches
                    for (int b = 0; b < batchCount; b++)
                    {
                        int vi = batchStart + b;
                        ParseVertex(buffer, b, vi, bytesPerVertex, layout, flip,
                            positions, rotations, scales, sh, shRest);
                        globalMin = Vector3.Min(globalMin, positions[vi]);
                        globalMax = Vector3.Max(globalMax, positions[vi]);
                    }
                }

                vertexIndex += batchCount;

                // Report progress at fixed intervals
                if (progress != null && vertexIndex >= nextProgressVertex)
                {
                    progress.Report((float)vertexIndex / vertexCount);
                    nextProgressVertex = vertexIndex + PROGRESS_INTERVAL;
                }
            }

            progress?.Report(1f);

            //Reorder SH rest coefficients
            // Based on aras-p/UnityGaussianSplatting ReorderSHs().
            if (shQuality == SHImportQuality.Full && layout.RestCount > 0 && layout.RestCount % 3 == 0)
            {
                ReorderSHRestToInterleaved(shRest, vertexCount, layout.RestCount);
            }

            return new GSImportData
            {
                Positions = positions,
                Rotations = rotations,
                Scales = scales,
                SH = sh,
                SHRest = shRest,
                SHRestCount = layout.RestCount,
                Bounds = new Bounds((globalMin + globalMax) * 0.5f, globalMax - globalMin),
                HasRotations = layout.HasRotation,
                HasScales = layout.HasScale,
                HasColors = layout.HasColor,
                HasOpacity = layout.HasOpacity,
                HasSHRest = layout.HasSHRest,
                ImportedSHQuality = layout.RestCount > 0 ? shQuality : SHImportQuality.None,
                AppliedFlip = flip
            };
        }

        /// <summary>
        /// Parses a single vertex from the buffer using precomputed field layout.
        /// Safe to call from multiple threads (writes only to unique array indices).
        /// </summary>
        private static void ParseVertex(
            byte[] buffer, int bufferIndex, int vertexIndex,
            int bytesPerVertex, PropertyLayout layout, CoordinateFlip flip,
            Vector3[] positions, quaternion[] rotations, Vector3[] scales,
            Vector4[] sh, float[] shRest)
        {
            int baseOff = bufferIndex * bytesPerVertex;

            // Position
            float x = layout.PosX.IsPresent ? ReadFloat(buffer, baseOff + layout.PosX.Offset, layout.PosX.Type) : 0f;
            float y = layout.PosY.IsPresent ? ReadFloat(buffer, baseOff + layout.PosY.Offset, layout.PosY.Type) : 0f;
            float z = layout.PosZ.IsPresent ? ReadFloat(buffer, baseOff + layout.PosZ.Offset, layout.PosZ.Type) : 0f;

            // Apply coordinate flip for handedness conversion
            if (flip == CoordinateFlip.FlipZ) z = -z;
            else if (flip == CoordinateFlip.FlipX) x = -x;

            positions[vertexIndex] = new Vector3(x, y, z);

            // Rotation
            float qx, qy, qz, qw;
            if (layout.Rot0.IsPresent)
            {
                // WXYZ convention (3DGS standard): rot_0=W, rot_1=X, rot_2=Y, rot_3=Z
                qw = ReadFloat(buffer, baseOff + layout.Rot0.Offset, layout.Rot0.Type);
                qx = layout.Rot1.IsPresent ? ReadFloat(buffer, baseOff + layout.Rot1.Offset, layout.Rot1.Type) : 0f;
                qy = layout.Rot2.IsPresent ? ReadFloat(buffer, baseOff + layout.Rot2.Offset, layout.Rot2.Type) : 0f;
                qz = layout.Rot3.IsPresent ? ReadFloat(buffer, baseOff + layout.Rot3.Offset, layout.Rot3.Type) : 0f;
            }
            else if (layout.QuatX.IsPresent)
            {
                qx = ReadFloat(buffer, baseOff + layout.QuatX.Offset, layout.QuatX.Type);
                qy = layout.QuatY.IsPresent ? ReadFloat(buffer, baseOff + layout.QuatY.Offset, layout.QuatY.Type) : 0f;
                qz = layout.QuatZ.IsPresent ? ReadFloat(buffer, baseOff + layout.QuatZ.Offset, layout.QuatZ.Type) : 0f;
                qw = layout.QuatW.IsPresent ? ReadFloat(buffer, baseOff + layout.QuatW.Offset, layout.QuatW.Type) : 1f;
            }
            else
            {
                qx = 0f; qy = 0f; qz = 0f; qw = 1f;
            }

            // When flipping an axis, a reflection is applied to the rotation.
            // Negating one position axis converts rotation q to a reflected quaternion by negating the 
            // two q-components.     FlipZ: negate qx, qy (keep qz, qw)     FlipX: negate qy, qz (keep qx, qw)
            if (flip == CoordinateFlip.FlipZ) { qx = -qx; qy = -qy; }
            else if (flip == CoordinateFlip.FlipX) { qy = -qy; qz = -qz; }

            rotations[vertexIndex] = math.normalize(new quaternion(qx, qy, qz, qw));

            // Scale
            if (layout.HasScale)
            {
                float sx = ReadFloat(buffer, baseOff + layout.ScaleX.Offset, layout.ScaleX.Type);
                float sy = ReadFloat(buffer, baseOff + layout.ScaleY.Offset, layout.ScaleY.Type);
                float sz = ReadFloat(buffer, baseOff + layout.ScaleZ.Offset, layout.ScaleZ.Type);
                if (layout.ScaleNeedsExp)
                    scales[vertexIndex] = new Vector3(Mathf.Exp(sx), Mathf.Exp(sy), Mathf.Exp(sz));
                else
                    scales[vertexIndex] = new Vector3(sx, sy, sz);
            }
            else
            {
                scales[vertexIndex] = Vector3.one;
            }

            // Color (SH DC or direct RGB)
            float sh0 = 0f, sh1 = 0f, sh2 = 0f;
            if (layout.HasColor)
            {
                sh0 = layout.ColorR.IsPresent ? ReadFloat(buffer, baseOff + layout.ColorR.Offset, layout.ColorR.Type) : 0f;
                sh1 = layout.ColorG.IsPresent ? ReadFloat(buffer, baseOff + layout.ColorG.Offset, layout.ColorG.Type) : 0f;
                sh2 = layout.ColorB.IsPresent ? ReadFloat(buffer, baseOff + layout.ColorB.Offset, layout.ColorB.Type) : 0f;

                if (layout.HasDirectColors)
                {
                    // Normalize uchar colors (0-255) to 0-1 range
                    if (layout.ColorR.Type == PLYType.UInt8) sh0 /= 255f;
                    if (layout.ColorG.Type == PLYType.UInt8) sh1 /= 255f;
                    if (layout.ColorB.Type == PLYType.UInt8) sh2 /= 255f;

                    // Convert from linear color space to SH DC representation
                    // so the downstream rendering pipeline receives consistent data.
                    // SH0ToColor: color = dc0 * C0 + 0.5  →  dc0 = (color - 0.5) / C0
                    sh0 = (sh0 - 0.5f) / SH_C0;
                    sh1 = (sh1 - 0.5f) / SH_C0;
                    sh2 = (sh2 - 0.5f) / SH_C0;
                }
            }

            // Opacity
            float alpha;
            if (layout.HasOpacity)
            {
                float rawOpacity = ReadFloat(buffer, baseOff + layout.Opacity.Offset, layout.Opacity.Type);
                if (layout.OpacityNeedsSigmoid)
                {
                    // Standard 3DGS: opacity is pre-sigmoid (logit) value
                    alpha = 1f / (1f + Mathf.Exp(-rawOpacity));
                }
                else
                {
                    // Non-3DGS: normalize integer types (e.g., uchar 0-255) and use directly
                    if (layout.Opacity.Type == PLYType.UInt8) rawOpacity /= 255f;
                    alpha = Mathf.Clamp01(rawOpacity);
                }
            }
            else
            {
                alpha = 1f; // Default: fully opaque
            }

            sh[vertexIndex] = new Vector4(sh0, sh1, sh2, alpha);

            // ── SH Rest coefficients ──
            if (layout.RestCount > 0)
            {
                int restBase = vertexIndex * layout.RestCount;
                for (int r = 0; r < layout.RestCount; r++)
                {
                    if (layout.RestFields[r].IsPresent)
                        shRest[restBase + r] = ReadFloat(buffer, baseOff + layout.RestFields[r].Offset, layout.RestFields[r].Type);
                }
            }
        }

        /// <summary>
        /// Reorders SH rest coefficients from PLY grouped-by-channel layout to
        /// interleaved-per-coefficient layout expected by the shader.
        ///
        /// Based on aras-p/UnityGaussianSplatting ReorderSHs().
        /// SPDX-License-Identifier: MIT
        /// </summary>
        private static void ReorderSHRestToInterleaved(float[] shRest, int vertexCount, int restCount)
        {
            int coeffsPerChannel = restCount / 3;
            float[] tmp = new float[restCount];

            for (int i = 0; i < vertexCount; i++)
            {
                int baseIdx = i * restCount;

                // Read grouped-by-channel into temp buffer, write back interleaved
                for (int j = 0; j < coeffsPerChannel; j++)
                {
                    tmp[j * 3 + 0] = shRest[baseIdx + j];                       // Red
                    tmp[j * 3 + 1] = shRest[baseIdx + j + coeffsPerChannel];     // Green
                    tmp[j * 3 + 2] = shRest[baseIdx + j + coeffsPerChannel * 2]; // Blue
                }

                Array.Copy(tmp, 0, shRest, baseIdx, restCount);
            }
        }

        #endregion

        #region Buffer Read Helpers

        /// <summary>
        /// Read a float value from a byte buffer at the given offset.
        /// Interprets bytes as little-endian for the given PLY type.
        /// </summary>
        private static float ReadFloat(byte[] buf, int offset, PLYType type)
        {
            switch (type)
            {
                case PLYType.Float32: return BitConverter.ToSingle(buf, offset);
                case PLYType.Float64: return (float)BitConverter.ToDouble(buf, offset);
                case PLYType.Int8:    return (sbyte)buf[offset];
                case PLYType.UInt8:   return buf[offset];
                case PLYType.Int16:   return BitConverter.ToInt16(buf, offset);
                case PLYType.UInt16:  return BitConverter.ToUInt16(buf, offset);
                case PLYType.Int32:   return BitConverter.ToInt32(buf, offset);
                case PLYType.UInt32:  return BitConverter.ToUInt32(buf, offset);
                default: return 0f;
            }
        }

        /// <summary>
        /// Reads exactly the requested number of bytes from the stream.
        /// Handles partial reads (where Stream.Read returns fewer bytes than requested).
        /// </summary>
        private static int ReadExact(Stream stream, byte[] buffer, int count)
        {
            int totalRead = 0;
            while (totalRead < count)
            {
                int read = stream.Read(buffer, totalRead, count - totalRead);
                if (read == 0) break;
                totalRead += read;
            }
            return totalRead;
        }

        #endregion

        #region PLY Type Parsing

        private static PLYType ParseType(string type)
        {
            return type.ToLowerInvariant() switch
            {
                "char" or "int8" => PLYType.Int8,
                "uchar" or "uint8" => PLYType.UInt8,
                "short" or "int16" => PLYType.Int16,
                "ushort" or "uint16" => PLYType.UInt16,
                "int" or "int32" => PLYType.Int32,
                "uint" or "uint32" => PLYType.UInt32,
                "float" or "float32" => PLYType.Float32,
                "double" or "float64" => PLYType.Float64,
                _ => throw new NotSupportedException($"Unsupported PLY property type: {type}")
            };
        }

        #endregion
    }
}
