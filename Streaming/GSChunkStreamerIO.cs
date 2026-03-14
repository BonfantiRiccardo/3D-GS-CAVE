using System;
using System.Collections.Concurrent;
using System.IO;
using System.Threading;
using UnityEngine;

namespace GaussianSplatting
{
    /// <summary> Partial class: background I/O thread, pooled reads, and SH de-interleaving. </summary>
    public partial class GSChunkStreamer
    {
        // ------------------------------------------------------------------
        //  Read buffer pool

        /// <summary>
        /// Thread-safe buffer pool keyed by byte count. All chunks share the same splat count, so each (attribute × 
        /// chunk) combination produces identically sized arrays, perfect for bucketed pooling with zero waste.
        /// </summary>
        private readonly ConcurrentDictionary<int, ConcurrentBag<byte[]>> readBufferPool
            = new ConcurrentDictionary<int, ConcurrentBag<byte[]>>();    //ConcurrentBag: thread-safe unordered collection

        private byte[] RentBuffer(int byteCount)
        {
            var bag = readBufferPool.GetOrAdd(byteCount, _ => new ConcurrentBag<byte[]>());
            return bag.TryTake(out byte[] buf) ? buf : new byte[byteCount];
        }

        private void ReturnBuffer(byte[] buf)
        {
            if (buf == null) return;
            var bag = readBufferPool.GetOrAdd(buf.Length, _ => new ConcurrentBag<byte[]>());
            bag.Add(buf);
        }

        // ------------------------------------------------------------------
        //  I/O thread lifecycle
        private void StartIOThread()
        {
            ioThreadRunning = true;
            ioThread = new Thread(IOThreadMain)
            {
                Name = "GSChunkIO",
                IsBackground = true,
                Priority = System.Threading.ThreadPriority.BelowNormal
            };
            ioThread.Start();
        }


        // ------------------------------------------------------------------
        //  Background I/O thread

        /// <summary>
        /// Entry point for the background I/O thread. Opens its own set of file handles and reads chunks on demand. 
        /// SH rest data is de-interleaved into per-band arrays here.
        /// </summary>
        private void IOThreadMain()
        {
            FileStream posFs = OpenFileHandle(posFilePath);
            FileStream rotFs = OpenFileHandle(rotFilePath);
            FileStream scaleFs = OpenFileHandle(scaleFilePath);
            FileStream shFs = OpenFileHandle(shFilePath);
            FileStream shRestFs = !string.IsNullOrEmpty(shRestFilePath)
                ? OpenFileHandle(shRestFilePath)
                : null;

            try
            {
                while (ioThreadRunning)
                {
                    while (requestQueue.TryDequeue(out ReadRequest req))
                    {
                        if (!ioThreadRunning) return;
                        ReadResult result = ExecuteChunkRead(req, posFs, rotFs, scaleFs, shFs, shRestFs);
                        completionQueue.Enqueue(result);
                    }

                    ioWakeEvent.Wait(50);
                    ioWakeEvent.Reset();
                }
            }
            finally
            {
                posFs?.Dispose();
                rotFs?.Dispose();
                scaleFs?.Dispose();
                shFs?.Dispose();
                shRestFs?.Dispose();
            }
        }

        private static FileStream OpenFileHandle(string path)
        {
            if (string.IsNullOrEmpty(path) || !File.Exists(path)) return null;
            return new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 65536);
        }

        /// <summary>
        /// Reads and builds a ReadResult for one chunk. Uses pooled buffers for all attribute arrays. 
        /// SH rest data is de-interleaved into per-band arrays on this thread.
        /// </summary>
        private ReadResult ExecuteChunkRead(
            ReadRequest req,
            FileStream posFs, FileStream rotFs,
            FileStream scaleFs, FileStream shFs, FileStream shRestFs)
        {
            ChunkInfo meta = asset.Chunks[req.chunkIndex];
            int count = meta.splatCount;
            int start = meta.startIndex;

            ReadResult r;
            r.chunkIndex = req.chunkIndex;
            r.slotIndex  = req.slotIndex;
            r.positions  = ReadRegionPooled(posFs,   start, count, ChunkedGSAsset.PositionStride);
            r.rotations  = ReadRegionPooled(rotFs,   start, count, ChunkedGSAsset.RotationStride);
            r.scales     = ReadRegionPooled(scaleFs, start, count, ChunkedGSAsset.ScaleStride);
            r.sh         = ReadRegionPooled(shFs,    start, count, ChunkedGSAsset.SHStride);

            // De-interleave SH rest on the IO thread
            r.shBand1 = null;
            r.shBand2 = null;
            r.shBand3 = null;

            if (shRestFs != null && shRestStride > 0)
            {
                byte[] raw = ReadRegionPooled(shRestFs, start, count, shRestStride);
                if (raw != null)
                {
                    SplitSHRestOnIOThread(raw, count, out r.shBand1, out r.shBand2, out r.shBand3);
                    ReturnBuffer(raw); // raw interleaved buffer no longer needed
                }
            }

            bool coreOk = r.positions != null && r.rotations != null &&
                          r.scales    != null && r.sh        != null;
            bool shRestOk = shRestStride <= 0 || shBands <= 0 ||
                            (shBands < 1 || r.shBand1 != null) &&
                            (shBands < 2 || r.shBand2 != null) &&
                            (shBands < 3 || r.shBand3 != null);

            r.ok    = coreOk && shRestOk;
            r.error = r.ok ? null : "missing or truncated chunk payload";
            return r;
        }

        /// <summary>
        /// Reads a contiguous region of a data file into a pooled byte array. Handles partial OS reads with a loop. 
        /// Returns the rented buffer on success, or returns it to the pool and yields null on failure.
        /// </summary>
        private byte[] ReadRegionPooled(FileStream fs, int startSplat, int splatCount, int stride)
        {
            if (fs == null) return null;

            long byteOffset = startSplat * (long)stride;
            int byteCount = splatCount * stride;
            byte[] data = RentBuffer(byteCount);

            fs.Seek(byteOffset, SeekOrigin.Begin);
            int totalRead = 0;
            while (totalRead < byteCount)
            {
                int read = fs.Read(data, totalRead, byteCount - totalRead);
                if (read <= 0) break;
                totalRead += read;
            }

            if (totalRead != byteCount)
            {
                ReturnBuffer(data); // don't leak the rented buffer on failure
                Debug.LogWarning($"GSChunkStreamer: incomplete read for chunk {startSplat} splats, expected {byteCount} bytes but got {totalRead}");
                return null;
            }
            return data;
        }

        /// <summary>
        /// Runs on the IO thread. Splits interleaved shRest into per-band arrays rented from the pool. 
        /// Caller returns these buffers after main-thread use.
        /// </summary>
        private void SplitSHRestOnIOThread(byte[] raw, int splatCount,
            out byte[] band1, out byte[] band2, out byte[] band3)
        {
            int b1Stride = ChunkedGSAsset.SHBand1Stride;
            int b2Stride = ChunkedGSAsset.SHBand2Stride;
            int b3Stride = ChunkedGSAsset.SHBand3Stride;

            band1 = shBands >= 1 ? RentBuffer(splatCount * b1Stride) : null;
            band2 = shBands >= 2 ? RentBuffer(splatCount * b2Stride) : null;
            band3 = shBands >= 3 ? RentBuffer(splatCount * b3Stride) : null;

            for (int s = 0; s < splatCount; s++)
            {
                int src = s * shRestStride;
                if (band1 != null) { Buffer.BlockCopy(raw, src, band1, s * b1Stride, b1Stride); src += b1Stride; }
                if (band2 != null) { Buffer.BlockCopy(raw, src, band2, s * b2Stride, b2Stride); src += b2Stride; }
                if (band3 != null)   Buffer.BlockCopy(raw, src, band3, s * b3Stride, b3Stride);
            }
        }
    }
}
