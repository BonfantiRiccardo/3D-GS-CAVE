using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting.Editor
{
    /// <summary>
    /// PLYParser is responsible for parsing PLY files and extracting gaussian data
    /// </summary>
    public static class PLYParser
    {
        /// <summary>
        /// Specifies the format of the PLY file (ASCII, Binary Little Endian, Binary Big Endian)
        /// </summary>
        private enum PLYFormat
        {
            ASCII,
            BinaryLittleEndian,             //As for now only binary little-endian PLY is supported
            BinaryBigEndian
        }

        /// <summary>
        /// Specifies the data type of a PLY property
        /// </summary>
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

        /// <summary>
        /// Represents a property in the PLY file
        /// </summary>
        private struct PLYProperty
        {
            public string Name;
            public PLYType Type;
        }

        /// <summary>
        /// Represents the PLY file header, where we store metadata about the file structure
        /// </summary>
        private struct PLYHeader
        {
            public int VertexCount;
            public List<PLYProperty> Properties;
            public PLYFormat Format;        //SPecifies the format of the PLY file
            public long DataStartOffset;    //Byte offset where the actual data starts
            public int[] RestIndexMap;       //Maps property index -> rest coeff index
            public int RestCount;            //Number of f_rest_* coefficients
        }

        /// <summary>
        /// Parses a PLY file at the given path. Always imports all data (rotations, scales, SH).
        /// </summary>
        /// <param name="path">The file path of the PLY file to parse.</param>
        /// <param name="options">The import options to use during parsing.</param>
        /// <returns>A GSImportData object containing the parsed data.</returns>
        public static GSImportData Parse(string path)
        {       // Open the file stream for reading
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);

            PLYHeader header = ReadHeader(stream);
            return ParseBinary(stream, header);
        }

        /// <summary>
        /// Reads the PLY header (ASCII) and extracts metadata
        /// </summary>
        /// <param name="stream">The stream to read the header from.</param>
        /// <returns>A PLYHeader struct containing the parsed header information.</returns>
        private static PLYHeader ReadHeader(Stream stream)
        {
            int vertexCount = 0;
            bool readingVertexProps = false;
            var properties = new List<PLYProperty>();
            PLYFormat format = PLYFormat.BinaryLittleEndian;

            int headerBytes = 0;
            var lineBuffer = new List<byte>(256);
            var headerLines = new List<string>();

            while (true)                        // Until we reach "end_header"
            {
                int b = stream.ReadByte();      //Read byte by byte
                if (b < 0)
                {
                    throw new InvalidDataException("PLY header not found.");
                }

                headerBytes++;
                if (b == '\n')                    // When we reach a newline, process the line and store it
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
                    lineBuffer.Add((byte)b);      // Accumulate bytes for the current line
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

        /// <summary>
        /// Parses binary PLY data from the stream based on the provided header and import options.
        /// </summary>
        /// <param name="stream">The stream containing the binary PLY data.</param>
        /// <param name="header">The PLYHeader containing metadata about the file structure.</param>
        /// <returns>A GSImportData object containing the parsed data.</returns>
        private static GSImportData ParseBinary(Stream stream, PLYHeader header)
        {   //If the format is binary, we read the data using a BinaryReader
            bool littleEndian = header.Format == PLYFormat.BinaryLittleEndian;

            // Ensure stream is positioned at the start of vertex data
            if (stream.Position != header.DataStartOffset)
            {
                stream.Position = header.DataStartOffset;
            }

            using var reader = new BinaryReader(stream, System.Text.Encoding.ASCII, leaveOpen: false);

            var positions = new Vector3[header.VertexCount];        //Initialize arrays to hold the data
            var rotations = new quaternion[header.VertexCount];
            var scales = new Vector3[header.VertexCount];
            var sh = new Vector4[header.VertexCount];  // Use Vector4 to preserve raw SH coefficients without clamping
            float[] shRest = header.RestCount > 0
                ? new float[header.VertexCount * header.RestCount]
                : Array.Empty<float>();

            Vector3 min = new(float.MaxValue, float.MaxValue, float.MaxValue);      //Keep track of bounds
            Vector3 max = new(float.MinValue, float.MinValue, float.MinValue);

            for (int i = 0; i < header.VertexCount; i++)    // Read each vertex
            {
                float x = 0f, y = 0f, z = 0f;               //Initialize default values for all properties
                float opacity = 1f;
                float rot0 = float.NaN, rot1 = float.NaN, rot2 = float.NaN, rot3 = float.NaN;
                float qx = 0f, qy = 0f, qz = 0f, qw = 1f;
                float s0 = float.NaN, s1 = float.NaN, s2 = float.NaN;
                float sh0 = 0f, sh1 = 0f, sh2 = 0f;

                for (int p = 0; p < header.Properties.Count; p++)           // Read each property for the vertex
                {
                    PLYProperty prop = header.Properties[p];
                    float value = ReadAsFloat(reader, prop.Type, littleEndian);

                    switch (prop.Name.ToLowerInvariant())           // Map property names to data fields
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
                        {
                            shRest[i * header.RestCount + restIndex] = value;
                        }
                    }
                }

                positions[i] = new Vector3(x, y, z);        //Assign read values to arrays
                min = Vector3.Min(min, positions[i]);
                max = Vector3.Max(max, positions[i]);

                // Rotation — always WXYZ (standard 3DGS: rot_0 = W)
                quaternion q;
                if (!float.IsNaN(rot0))
                {
                    q = new quaternion(rot1, rot2, rot3, rot0); // WXYZ: rot_0=W, rot_1=X, rot_2=Y, rot_3=Z
                }
                else
                {
                    q = new quaternion(qx, qy, qz, qw);
                }
                // Normalize quaternion to ensure valid rotation
                rotations[i] = math.normalize(q);

                // Scale — always apply exp transform
                if (float.IsNaN(s0) || float.IsNaN(s1) || float.IsNaN(s2))
                {
                    scales[i] = Vector3.one;
                }
                else
                {
                    scales[i] = new Vector3(
                        Mathf.Exp(s0),
                        Mathf.Exp(s1),
                        Mathf.Exp(s2)
                    );
                }

                // Standard 3DGS PLY files store opacity as logit (inverse sigmoid)
                // Apply sigmoid to convert to [0, 1] range: 1 / (1 + exp(-x))
                float alpha = 1.0f / (1.0f + Mathf.Exp(-opacity));
                
                // Store as Vector4 to preserve raw SH coefficients (can be negative or > 1)
                sh[i] = new Vector4(sh0, sh1, sh2, alpha);
            }

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

        /// <summary>
        /// Parses a PLY property type string into a PLYType enum.
        /// </summary>
        /// <param name="type">The property type string.</param>
        /// <returns>The corresponding PLYType enum value.</returns>
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
        /// Reads a value from the BinaryReader as a float based on the specified PLYType
        /// </summary>
        /// <param name="reader">The BinaryReader to read from.</param>
        /// <param name="type">The PLYType of the value to read.</param>
        /// <param name="littleEndian">Whether the data is in little-endian format.</param>
        /// <returns>The read value as a float.</returns>
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

        /// <summary>
        /// Reads an integer value from the BinaryReader based on the specified size, endianness, and signedness.
        /// </summary>
        /// <param name="reader">The BinaryReader to read from.</param>
        /// <param name="size">The size of the integer in bytes.</param>
        /// <param name="littleEndian">Whether the data is in little-endian format.</param>
        /// <param name="signed">Whether the integer is signed.</param>
        /// <returns>The read integer value as a long.</returns>
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

        /// <summary>
        /// Reads a float value from the BinaryReader with the specified endianness.
        /// </summary>
        /// <param name="reader">The BinaryReader to read from.</param>
        /// <param name="littleEndian">Whether the data is in little-endian format.</param>
        /// <returns>The read float value.</returns>
        private static float ReadFloat(BinaryReader reader, bool littleEndian)
        {
            byte[] bytes = reader.ReadBytes(4);
            if (!littleEndian)
            {
                Array.Reverse(bytes);
            }
            return BitConverter.ToSingle(bytes, 0);
        }

        /// <summary>
        /// Reads a double value from the BinaryReader with the specified endianness.
        /// </summary>
        /// <param name="reader">The BinaryReader to read from.</param>
        /// <param name="littleEndian">Whether the data is in little-endian format.</param>
        /// <returns>The read double value.</returns>
        private static double ReadDouble(BinaryReader reader, bool littleEndian)
        {
            byte[] bytes = reader.ReadBytes(8);
            if (!littleEndian)
            {
                Array.Reverse(bytes);
            }
            return BitConverter.ToDouble(bytes, 0);
        }

        /// <summary>
        /// Builds a mapping from property indices to SH rest coefficient indices.
        /// </summary>
        /// <param name="properties">The list of PLY properties.</param>
        /// <param name="restCount">The output count of SH rest coefficients.</param>
        /// <returns>An array mapping property indices to SH rest coefficient indices.</returns>
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

        /// <summary>
        /// Logs the parsed PLY header information for debugging purposes.
        /// </summary>
        /// <param name="format">The PLY file format.</param>
        /// <param name="vertexCount">The number of vertices.</param>
        /// <param name="properties">The list of PLY properties.</param>
        /// <param name="restCount">The number of SH rest coefficients.</param>
        /// <param name="headerLines">The raw header lines.</param>
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
