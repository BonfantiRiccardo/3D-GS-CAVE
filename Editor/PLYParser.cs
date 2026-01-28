using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    // This class is responsible for parsing PLY files and extracting gaussian data
    public static class PLYParser
    {
        // Only binary little-endian PLY is supported
        private enum PLYFormat
        {
            BinaryLittleEndian,
            BinaryBigEndian
        }

        // Supported PLY property types
        private enum PLYType
        {
            Int8,
            UInt8,
            Int16,
            UInt16,
            Int32,
            UInt32,
            Float32,
            Float64
        }

        // Represents a property in the PLY file
        private struct PLYProperty
        {
            public string Name;
            public PLYType Type;
        }

        // Represents the PLY file header, where we store metadata about the file structure
        private struct PLYHeader
        {
            public int VertexCount;
            public List<PLYProperty> Properties;
            public PLYFormat Format;        //SPecifies the format of the PLY file
            public long DataStartOffset;    //Byte offset where the actual data starts
            public int[] RestIndexMap;       //Maps property index -> rest coeff index
            public int RestCount;            //Number of f_rest_* coefficients
        }

        // Main method to parse a PLY file and return the imported data
        public static GSImportData Parse(string path, PLYImportOptions options)
        {       // Open the file stream for reading
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);

            PLYHeader header = ReadHeader(stream);
            return ParseBinary(stream, header, options);
        }

        // Reads the PLY header and extracts metadata
        private static PLYHeader ReadHeader(Stream stream)
        {
            int vertexCount = 0;
            bool readingVertexProps = false;
            var properties = new List<PLYProperty>();
            PLYFormat format = PLYFormat.BinaryLittleEndian;

            int headerBytes = 0;
            var lineBuffer = new List<byte>(256);
            var headerLines = new List<string>();

            while (true)
            {
                int b = stream.ReadByte();
                if (b < 0)
                {
                    throw new InvalidDataException("PLY header not found.");
                }

                headerBytes++;
                if (b == '\n')
                {
                    string line = System.Text.Encoding.ASCII.GetString(lineBuffer.ToArray()).TrimEnd('\r');
                    lineBuffer.Clear();
                    headerLines.Add(line);

                    if (line.StartsWith("format ", StringComparison.OrdinalIgnoreCase))
                    {
                        if (line.IndexOf("binary_little_endian", StringComparison.OrdinalIgnoreCase) >= 0)
                        {
                            format = PLYFormat.BinaryLittleEndian;
                        }
                        else if (line.IndexOf("binary_big_endian", StringComparison.OrdinalIgnoreCase) >= 0)
                        {
                            throw new NotSupportedException("Binary big-endian PLY is not supported.");
                        }
                        else
                        {
                            throw new NotSupportedException("Only binary little-endian PLY is supported.");
                        }
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
                            {
                                throw new NotSupportedException("PLY list properties are not supported.");
                            }

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

            var restIndexMap = BuildRestIndexMap(properties, out int restCount);
            LogHeader(format, vertexCount, properties, restCount, headerLines);

            return new PLYHeader
            {
                VertexCount = vertexCount,
                Properties = properties,
                Format = format,
                DataStartOffset = headerBytes,
                RestIndexMap = restIndexMap,
                RestCount = restCount
            };
        }

        private static GSImportData ParseBinary(Stream stream, PLYHeader header, PLYImportOptions options)
        {   //If the format is binary, we read the data using a BinaryReader
            bool littleEndian = header.Format == PLYFormat.BinaryLittleEndian;

            // Ensure stream is positioned at the start of vertex data
            if (stream.Position != header.DataStartOffset)
            {
                stream.Position = header.DataStartOffset;
            }

            using var reader = new BinaryReader(stream, System.Text.Encoding.ASCII, leaveOpen: false);

            var positions = new Vector3[header.VertexCount];
            var rotations = new quaternion[header.VertexCount];
            var scales = new Vector3[header.VertexCount];
            var sh = new Vector3[header.VertexCount];
            float[] shRest = header.RestCount > 0 && options.ImportSH
                ? new float[header.VertexCount * header.RestCount]
                : Array.Empty<float>();

            Vector3 min = new(float.MaxValue, float.MaxValue, float.MaxValue);
            Vector3 max = new(float.MinValue, float.MinValue, float.MinValue);

            for (int i = 0; i < header.VertexCount; i++)    // Read each vertex
            {
                float x = 0f, y = 0f, z = 0f;
                float r = 1f, g = 1f, b = 1f;
                float rot0 = float.NaN, rot1 = float.NaN, rot2 = float.NaN, rot3 = float.NaN;
                float qx = 0f, qy = 0f, qz = 0f, qw = 1f;
                float s0 = float.NaN, s1 = float.NaN, s2 = float.NaN;
                float sh0 = 0f, sh1 = 0f, sh2 = 0f;

                for (int p = 0; p < header.Properties.Count; p++)
                {
                    PLYProperty prop = header.Properties[p];
                    float value = ReadAsFloat(reader, prop.Type, littleEndian);

                    switch (prop.Name.ToLowerInvariant())
                    {
                        case "x": x = value; break;
                        case "y": y = value; break;
                        case "z": z = value; break;
                        case "red": r = value; break;
                        case "green": g = value; break;
                        case "blue": b = value; break;
                        case "r": r = value; break;
                        case "g": g = value; break;
                        case "b": b = value; break;
                        case "alpha": break;
                        case "opacity": break;
                        case "rot_0": rot0 = value; break;
                        case "rot_1": rot1 = value; break;
                        case "rot_2": rot2 = value; break;
                        case "rot_3": rot3 = value; break;
                        case "qx": qx = value; break;
                        case "qy": qy = value; break;
                        case "qz": qz = value; break;
                        case "qw": qw = value; break;
                        case "scale_0": s0 = value; break;
                        case "scale_1": s1 = value; break;
                        case "scale_2": s2 = value; break;
                        case "scale_x": s0 = value; break;
                        case "scale_y": s1 = value; break;
                        case "scale_z": s2 = value; break;
                        case "f_dc_0": sh0 = value; break;
                        case "f_dc_1": sh1 = value; break;
                        case "f_dc_2": sh2 = value; break;
                    }

                    if (options.ImportSH && header.RestCount > 0)
                    {
                        int restIndex = header.RestIndexMap[p];
                        if (restIndex >= 0)
                        {
                            shRest[i * header.RestCount + restIndex] = value;
                        }
                    }
                }

                positions[i] = new Vector3(x, y, z);
                min = Vector3.Min(min, positions[i]);
                max = Vector3.Max(max, positions[i]);

                if (options.ImportRotations)
                {
                    if (!float.IsNaN(rot0))
                    {
                        rotations[i] = options.RotationOrder == RotationOrder.WXYZ
                            ? new quaternion(rot1, rot2, rot3, rot0)
                            : new quaternion(rot0, rot1, rot2, rot3);
                    }
                    else
                    {
                        rotations[i] = new quaternion(qx, qy, qz, qw);
                    }
                }
                else
                {
                    rotations[i] = quaternion.identity;
                }

                if (options.ImportScales)
                {
                    if (float.IsNaN(s0) || float.IsNaN(s1) || float.IsNaN(s2))
                    {
                        scales[i] = Vector3.one;
                    }
                    else
                    {
                        scales[i] = new Vector3(s0, s1, s2);
                    }
                }
                else
                {
                    scales[i] = Vector3.one;
                }

                sh[i] = new Vector3(sh0, sh1, sh2);
            }

            return new GSImportData
            {
                Positions = positions,
                Rotations = rotations,
                Scales = scales,
                SH = sh,
                SHRest = options.ImportSH ? shRest : Array.Empty<float>(),
                SHRestCount = options.ImportSH ? header.RestCount : 0,
                Bounds = new Bounds((min + max) * 0.5f, max - min)
            };
        }

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

        private static float ReadAsFloat(BinaryReader reader, PLYType type, bool littleEndian)
        {
            return type switch
            {
                PLYType.Int8 => (sbyte)ReadInteger(reader, 1, littleEndian, signed: true),
                PLYType.UInt8 => (byte)ReadInteger(reader, 1, littleEndian, signed: false),
                PLYType.Int16 => (short)ReadInteger(reader, 2, littleEndian, signed: true),
                PLYType.UInt16 => (ushort)ReadInteger(reader, 2, littleEndian, signed: false),
                PLYType.Int32 => (int)ReadInteger(reader, 4, littleEndian, signed: true),
                PLYType.UInt32 => (uint)ReadInteger(reader, 4, littleEndian, signed: false),
                PLYType.Float32 => ReadFloat(reader, littleEndian),
                PLYType.Float64 => (float)ReadDouble(reader, littleEndian),
                _ => 0f
            };
        }

        private static long ReadInteger(BinaryReader reader, int size, bool littleEndian, bool signed)
        {
            byte[] bytes = reader.ReadBytes(size);
            if (!littleEndian)
            {
                Array.Reverse(bytes);
            }

            return size switch
            {
                1 => signed ? (sbyte)bytes[0] : bytes[0],
                2 => signed ? BitConverter.ToInt16(bytes, 0) : BitConverter.ToUInt16(bytes, 0),
                4 => signed ? BitConverter.ToInt32(bytes, 0) : BitConverter.ToUInt32(bytes, 0),
                _ => 0
            };
        }

        private static float ReadFloat(BinaryReader reader, bool littleEndian)
        {
            byte[] bytes = reader.ReadBytes(4);
            if (!littleEndian)
            {
                Array.Reverse(bytes);
            }
            return BitConverter.ToSingle(bytes, 0);
        }

        private static double ReadDouble(BinaryReader reader, bool littleEndian)
        {
            byte[] bytes = reader.ReadBytes(8);
            if (!littleEndian)
            {
                Array.Reverse(bytes);
            }
            return BitConverter.ToDouble(bytes, 0);
        }

        private static int[] BuildRestIndexMap(List<PLYProperty> properties, out int restCount)
        {
            int[] map = new int[properties.Count];
            for (int i = 0; i < map.Length; i++)
            {
                map[i] = -1;
            }

            int maxIndex = -1;
            for (int i = 0; i < properties.Count; i++)
            {
                string name = properties[i].Name;
                if (!name.StartsWith("f_rest_", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                string suffix = name.Substring("f_rest_".Length);
                if (int.TryParse(suffix, NumberStyles.Integer, CultureInfo.InvariantCulture, out int idx) && idx >= 0)
                {
                    map[i] = idx;
                    if (idx > maxIndex)
                    {
                        maxIndex = idx;
                    }
                }
            }

            restCount = maxIndex + 1;
            return map;
        }

        private static void LogHeader(PLYFormat format, int vertexCount, List<PLYProperty> properties, int restCount, List<string> headerLines)
        {
            string propertyList = properties.Count > 0
                ? string.Join(", ", properties.ConvertAll(p => p.Name))
                : "(none)";

            Debug.Log(
                "PLY header parsed:\n" +
                $"Format: {format}\n" +
                $"Vertex Count: {vertexCount}\n" +
                $"Property Count: {properties.Count}\n" +
                $"SH Rest Count: {restCount}\n" +
                $"Properties: {propertyList}\n" +
                string.Join("\n", headerLines));
        }
    }
}
