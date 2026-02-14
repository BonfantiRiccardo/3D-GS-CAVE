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
    /// 
    /// Design decisions:
    /// - Supports cancellation via CancellationToken.
    /// - Reports progress for editor progress bars.
    /// - Large-file strategy: reads data in buffered chunks (16 MB I/O buffer), but still allocates final arrays up front.
    /// </summary>
    public static class AsyncPLYParser
    {
        // 16 MB I/O buffer for sequential reads
        private const int IO_BUFFER_SIZE = 16 * 1024 * 1024;

        // Progress report interval: every N vertices (keeps overhead low)
        private const int PROGRESS_INTERVAL = 50000;

        /// <summary>
        /// Internal PLY format representation used during parsing.
        /// </summary>
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
                PLYType.Int8    => 1,     PLYType.UInt8   => 1,
                PLYType.Int16   => 2,     PLYType.UInt16  => 2,
                PLYType.Int32   => 4,     PLYType.UInt32  => 4,
                PLYType.Float32 => 4,     PLYType.Float64 => 8,
                _ => 0
            };
        }

        /// <summary>
        /// Parsed PLY header metadata.
        /// </summary>
        private struct PLYHeader
        {
            public int VertexCount;
            public List<PLYProperty> Properties;
            public PLYFormat Format;
            public long DataStartOffset;
            public int BytesPerVertex;
            public int[] RestIndexMap;
            public int RestCount;
        }

        /// <summary>
        /// Parses a PLY file asynchronously on a background thread.
        /// </summary>
        public static Task<GSImportData> ParseAsync(
            string path,
            IProgress<float> progress = null,
            CancellationToken cancellationToken = default)
        {
            return Task.Run(() => ParseInternal(path, progress, cancellationToken), cancellationToken);
        }

        /// <summary>
        /// Synchronous parse entry point (for use in ScriptedImporter where we need synchronous results).
        /// Internally uses the same optimized path but runs on the calling thread.
        /// </summary>
        public static GSImportData Parse(string path, IProgress<float> progress = null)
        {
            return ParseInternal(path, progress, cancellationToken: default);
        }

        /// <summary>
        /// Internal parse implementation — runs on whatever thread calls it.
        /// </summary>
        private static GSImportData ParseInternal(
            string path,
            IProgress<float> progress,
            CancellationToken cancellationToken)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"PLY file not found: {path}");

            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read,
                FileShare.Read, IO_BUFFER_SIZE, FileOptions.SequentialScan);

            // Step 1: Parse header
            progress?.Report(0f);
            PLYHeader header = ReadHeader(stream);

            if (header.Format != PLYFormat.BinaryLittleEndian)
                throw new NotSupportedException("Only binary little-endian PLY is supported.");

            // Step 2: Parse binary vertex data using buffered reading
            return ParseBinaryBuffered(stream, header, progress, cancellationToken);
        }


        /// <summary>
        /// Reads the PLY header from the stream.
        /// </summary>
        private static PLYHeader ReadHeader(Stream stream)
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
                        // First line must be "ply"
                        if (lineBuffer.Count == 0 && line != "ply")
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

            // Build SH rest index map
            var restIndexMap = BuildRestIndexMap(properties, out int restCount);

            return new PLYHeader
            {
                VertexCount = vertexCount,
                Properties = properties,
                Format = format,
                DataStartOffset = headerBytes,
                BytesPerVertex = bytesPerVertex,
                RestIndexMap = restIndexMap,
                RestCount = restCount
            };
        }


        // Binary Data Parsing — Buffered

        /// <summary>
        /// Parses binary vertex data using a large I/O buffer for sequential reads.
        /// This avoids per-property BinaryReader.ReadBytes allocations (the bottleneck 
        /// in the original synchronous PLYParser) by reading large blocks and parsing 
        /// from a byte buffer with computed offsets.
        /// </summary>
        private static GSImportData ParseBinaryBuffered(
            Stream stream,
            PLYHeader header,
            IProgress<float> progress,
            CancellationToken cancellationToken)
        {
            int vertexCount = header.VertexCount;
            int bytesPerVertex = header.BytesPerVertex;

            // Ensure stream is at data start
            if (stream.Position != header.DataStartOffset)
                stream.Position = header.DataStartOffset;

            // Pre-allocate output arrays
            var positions = new Vector3[vertexCount];
            var rotations = new quaternion[vertexCount];
            var scales = new Vector3[vertexCount];
            var sh = new Vector4[vertexCount];
            float[] shRest = header.RestCount > 0
                ? new float[vertexCount * header.RestCount]
                : Array.Empty<float>();

            Vector3 min = new(float.MaxValue, float.MaxValue, float.MaxValue);
            Vector3 max = new(float.MinValue, float.MinValue, float.MinValue);

            // Pre-compute property byte offsets within each vertex record
            var propOffsets = new int[header.Properties.Count];
            {
                int offset = 0;
                for (int p = 0; p < header.Properties.Count; p++)
                {
                    propOffsets[p] = offset;
                    offset += header.Properties[p].ByteSize;
                }
            }

            // I/O buffer: read as many complete vertices as fit
            int verticesPerBuffer = Math.Max(1, IO_BUFFER_SIZE / bytesPerVertex);
            byte[] buffer = new byte[verticesPerBuffer * bytesPerVertex];

            int vertexIndex = 0;
            int nextProgressVertex = PROGRESS_INTERVAL;

            while (vertexIndex < vertexCount)
            {
                cancellationToken.ThrowIfCancellationRequested();

                // How many vertices to read in this batch?
                int batchCount = Math.Min(verticesPerBuffer, vertexCount - vertexIndex);
                int batchBytes = batchCount * bytesPerVertex;

                // Read the full batch from disk
                int bytesRead = ReadExact(stream, buffer, batchBytes);
                if (bytesRead < batchBytes)
                    throw new IOException($"Unexpected end of PLY data at vertex {vertexIndex}.");

                // Parse vertices from the buffer
                for (int b = 0; b < batchCount; b++)
                {
                    int baseOffset = b * bytesPerVertex;
                    int vi = vertexIndex + b;

                    // Read property values
                    float x = 0f, y = 0f, z = 0f;
                    float opacity = 1f;
                    float rot0 = float.NaN, rot1 = float.NaN, rot2 = float.NaN, rot3 = float.NaN;
                    float qx = 0f, qy = 0f, qz = 0f, qw = 1f;
                    float s0 = float.NaN, s1 = float.NaN, s2 = float.NaN;
                    float sh0 = 0f, sh1 = 0f, sh2 = 0f;

                    for (int p = 0; p < header.Properties.Count; p++)
                    {
                        PLYProperty prop = header.Properties[p];
                        int pOffset = baseOffset + propOffsets[p];
                        float value = ReadFloatFromBuffer(buffer, pOffset, prop.Type);

                        switch (prop.Name.ToLowerInvariant())
                        {
                            case "x": x = value; break;
                            case "y": y = value; break;
                            case "z": z = value; break;
                            case "alpha" or "opacity": opacity = value; break;
                            case "rot_0": rot0 = value; break;
                            case "rot_1": rot1 = value; break;
                            case "rot_2": rot2 = value; break;
                            case "rot_3": rot3 = value; break;
                            case "qx": qx = value; break;
                            case "qy": qy = value; break;
                            case "qz": qz = value; break;
                            case "qw": qw = value; break;
                            case "scale_0" or "scale_x": s0 = value; break;
                            case "scale_1" or "scale_y": s1 = value; break;
                            case "scale_2" or "scale_z": s2 = value; break;
                            case "f_dc_0" or "r" or "red": sh0 = value; break;
                            case "f_dc_1" or "g" or "green": sh1 = value; break;
                            case "f_dc_2" or "b" or "blue": sh2 = value; break;
                        }

                        // Always capture SH rest coefficients
                        if (header.RestCount > 0)
                        {
                            int restIndex = header.RestIndexMap[p];
                            if (restIndex >= 0)
                                shRest[vi * header.RestCount + restIndex] = value;
                        }
                    }

                    // Position
                    positions[vi] = new Vector3(x, y, z);
                    min = Vector3.Min(min, positions[vi]);
                    max = Vector3.Max(max, positions[vi]);

                    // Rotation — always WXYZ (standard 3DGS: rot_0 = W)
                    quaternion q;
                    if (!float.IsNaN(rot0))
                        q = new quaternion(rot1, rot2, rot3, rot0); // WXYZ: rot_0=W, rot_1=X, rot_2=Y, rot_3=Z
                    else
                        q = new quaternion(qx, qy, qz, qw);
                    rotations[vi] = math.normalize(q);

                    // Scale — always apply exp transform
                    if (float.IsNaN(s0) || float.IsNaN(s1) || float.IsNaN(s2))
                        scales[vi] = Vector3.one;
                    else
                        scales[vi] = new Vector3(Mathf.Exp(s0), Mathf.Exp(s1), Mathf.Exp(s2));

                    // Opacity (sigmoid) + SH DC (raw)
                    float alpha = 1.0f / (1.0f + Mathf.Exp(-opacity));
                    sh[vi] = new Vector4(sh0, sh1, sh2, alpha);
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

            return new GSImportData
            {
                Positions = positions,
                Rotations = rotations,
                Scales = scales,
                SH = sh,
                SHRest = shRest,
                SHRestCount = header.RestCount,
                Bounds = new Bounds((min + max) * 0.5f, max - min)
            };
        }


        // Buffer Read Helpers

        /// <summary>
        /// Read a float value from a byte buffer at the given offset, interpreting 
        /// bytes as little-endian for the given PLY type.
        /// </summary>
        private static float ReadFloatFromBuffer(byte[] buf, int offset, PLYType type)
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

        // PLY Type Parsing
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


        /// <summary>
        /// Builds a mapping from property indices to SH rest coefficient indices.
        /// </summary>
        private static int[] BuildRestIndexMap(List<PLYProperty> properties, out int restCount)
        {
            int[] map = new int[properties.Count];
            for (int i = 0; i < map.Length; i++)
                map[i] = -1;

            int maxIndex = -1;
            for (int i = 0; i < properties.Count; i++)
            {
                string name = properties[i].Name;
                if (!name.StartsWith("f_rest_", StringComparison.OrdinalIgnoreCase))
                    continue;

                string suffix = name.Substring("f_rest_".Length);
                if (int.TryParse(suffix, NumberStyles.Integer, CultureInfo.InvariantCulture, out int idx) && idx >= 0)
                {
                    map[i] = idx;
                    if (idx > maxIndex)
                        maxIndex = idx;
                }
            }

            restCount = maxIndex + 1;
            return map;
        }
    }
}
