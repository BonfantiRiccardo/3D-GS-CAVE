# [Work in Progress] Large Scenes Gaussian Splatting Renderer (Unity URP)

This folder contains a Unity URP Gaussian Splatting implementation with two runtime paths:

- `GSComponent` + `GSAsset`: full dataset loaded into GPU buffers.
- `ChunkedGSComponent` + `ChunkedGSAsset` + `GSChunkStreamer`: chunked streaming for very large scenes.

The system imports `.ply` point cloud splat data, stores heavy payloads as external binary files, sorts splats on GPU every frame (or on demand), and renders with procedural transparent splats.

## Requirements

- Unity `6000.3`
- `com.unity.render-pipelines.universal` `17.3.0`
- Compute shader capable GPU with wave/subgroup support

## Folder Structure

- `Editor/`
  - PLY import pipeline (`PLYImporter`, `AsyncPLYParser`, `SpatialSorter`, asset builders)
- `Runtime/`
  - Runtime assets/components/managers/streaming (`GSAsset`, `ChunkedGSAsset`, `GSComponent`, `ChunkedGSComponent`, `GSChunkStreamer`)
- `Rendering/`
  - URP renderer features, shaders, compute sorting
- `UI/`
  - Runtime performance overlay (`GSPerformanceOverlay`)
- `ThirdParty/`
  - GPU radix sort implementation (`DeviceRadixSort.hlsl`, `SortCommon.hlsl`)
- `Settings/`
  - Sample URP pipeline and renderer assets with render features enabled

## End-to-End Pipeline

1. Import `.ply` via `PLYImporter` (ScriptedImporter).
2. Parse binary little-endian vertex data with `AsyncPLYParser`.
3. Convert/normalize properties (position, rotation, scale, opacity, SH coefficients).
4. Sort splats spatially by Morton code and partition into chunks (`SpatialSorter`).
5. Write external payload files into `Assets/StreamingAssets/<plyName>/`.
6. Build a `ChunkedGSAsset` (chunk metadata + global metadata) as the imported Unity object.
7. At runtime:
   - Non-chunked path: upload all splats once.
   - Chunked path: stream visible chunks into a fixed GPU pool, build remap indirection, and evict by hysteresis + delay.
8. In render features:
   - Run compute passes for culling/sort key generation/radix sort/view-data precompute.
   - Draw procedural quads in sorted back-to-front order.

## Import Pipeline Details

### `PLYImporter` (`Editor/PLYImporter.cs`)

- Registered as `[ScriptedImporter(5, "ply")]`.
- Quality options:
  - `Full`: import all SH rest coefficients.
  - `Reduced`: band 1 only.
  - `None`: DC + opacity only.
- Coordinate conversion options:
  - `None`, `FlipZ`, `FlipX`.
- Chunk size option (default `4096`).
- Uses `AsyncPLYParser` first; falls back to legacy `PLYParser` if needed.
- Current default output is chunked (`ChunkedGSAsset`).

### `AsyncPLYParser` (`Editor/AsyncPLYParser.cs`)

- Reads binary little-endian PLY using buffered sequential I/O.
- Uses precomputed property layout for fast parsing.
- Supports multiple common naming conventions:
  - Position: `x,y,z`
  - Rotation: `rot_0..rot_3` (WXYZ) or `qx,qy,qz,qw`
  - Scale: `scale_0..2` or `scale_x..z`
  - Color: `f_dc_*` or `r/g/b`, `red/green/blue`
  - SH rest: `f_rest_N`
  - Opacity: `opacity` or `alpha`
- Applies transforms when needed:
  - Opacity sigmoid for 3DGS logit-style opacity
  - Scale exponential for log-scale inputs
  - RGB-to-SH-DC conversion when direct color is provided
- Reorders full SH rest layout to shader-expected interleaved format.

### Spatial Sorting and Chunking

- `SpatialSorter` computes 30-bit Morton codes (`MortonCode.Encode`) from positions.
- Arrays are reordered by spatial locality.
- Data is partitioned into chunks with per-chunk AABB metadata:
  - `bounds`, `splatCount`, `startIndex`, `dataOffset`
- Chunk metadata is serialized in `ChunkedGSAsset`.

## Runtime Data Model

### External Payload Files

For both asset paths, heavy data lives outside the ScriptableObject in:

`Assets/StreamingAssets/<assetFolderName>/`

Files:

- `positions.bytes` (`float3`, 12 bytes/splat)
- `rotations.bytes` (`float4`, 16 bytes/splat)
- `scales.bytes` (`float3`, 12 bytes/splat)
- `sh.bytes` (`float4`, DC RGB + opacity, 16 bytes/splat)
- `shrest.bytes` (`float[SHRestCount]`, `4 * SHRestCount` bytes/splat)

### `ChunkedGSAsset` (chunked)

- Stores global metadata + `ChunkInfo[]`.
- Reads chunk regions from external files using `startIndex * stride` offsets.
- Supports single-chunk and multi-chunk reads.

## Runtime Components


### `ChunkedGSComponent`

- Registers with `ChunkedGSManager`.
- Creates `GSChunkStreamer` with streaming parameters:
  - `maxVisibleSplats`
  - `frustumMargin` (load margin)
  - `unloadMargin` (evict margin)
  - `evictionDelayFrames`
- Exposes active/loaded stats and estimated GPU memory.
- Exposes remap buffer for pool indirection.

### `GSChunkStreamer` internals

- Fixed-size GPU pool split into chunk slots.
- Free-slot stack for slot allocation/reuse.
- Background I/O thread with dedicated file handles.
- CPU staging arrays mirror full GPU pool.
- Chunk read completions are copied into staging arrays (CPU only).
- Batched GPU flush: at most 6 `SetData` calls/frame:
  - up to 5 attribute uploads + 1 remap upload.
- Remap (`uint`) maps logical active splat index -> physical pool index.
- Frustum hysteresis:
  - inner planes for loading
  - outer planes for eviction
- Eviction delay prevents load/unload thrash.

## Rendering Architecture

### URP Features
`ChunkedGSRenderFeature`: renders `ChunkedGSComponent` instances and drives per-camera visibility updates:
- enqueue compute sort pass + raster draw pass via RenderGraph.
- use `RenderPassEvent.BeforeRenderingTransparents` by default.
- can skip expensive sort passes when camera has not moved.

### Compute Sorting (`Rendering/GSSorting.compute`)

Kernels:

- `InitSortBuffers`
- `CalcViewData`
- `CalcSortKeys`
- `CalcSplatViewData`
- `InitDeviceRadixSort`, `Upsweep`, `Scan`, `Downsweep` (from third-party radix sort)

Process:

1. Initialize identity payload indices.
2. Compute view-space data and basic visibility.
3. Compute depth sort keys (back-to-front using ascending `viewZ`).
4. Run 4-pass LSD radix sort over float keys.
5. Precompute per-splat clip position, ellipse axes, and packed FP16 color into `SplatViewData`.

### Shader Draw (`Rendering/GSShader.shader`)

- Draws `6` vertices per splat instance (`DrawProcedural`, triangle list).
- Uses sorted order buffer + precomputed `SplatViewData`.
- Transparent blend setup:
  - `Blend One OneMinusSrcAlpha`
  - `ZWrite Off`
  - `ZTest LEqual`
- Gaussian alpha falloff in fragment shader, premultiplied output.
- Optional debug mode for splat centers.

## Spherical Harmonics and Color

- `sh` buffer stores DC RGB + opacity.
- Optional `shrest` stores higher-order coefficients.
- `GSUtils.hlsl` evaluates SH view-dependent color (up to higher bands when available).
- Color-space control per component:
  - `Auto`
  - `ForceLinear`
  - `ForceGamma`

## Setup Guide

1. Use URP and ensure a renderer includes Gaussian splat render features.
   - Sample assets: `Settings/GSPass.asset` and `Settings/GSPass_Renderer.asset`.
2. Import a `.ply` file into Unity (`Assets/GaussianSplatting/Models` already contains examples).
3. For large scenes, add `ChunkedGSComponent` to a GameObject and assign the imported chunked asset.
4. For full-resident path, use `GSComponent` + `GSAsset` (builder/API path still exists).
5. Tune component settings:
   - `maxVisibleSplats`
   - `frustumMargin` / `unloadMargin`
   - `evictionDelayFrames`
   - `splatScale`
   - SH quality at import time

## Diagnostics and Debugging

- `GSPerformanceOverlay` (auto-bootstrap at runtime) shows:
  - FPS/frame time
  - active/total splats
  - loaded/total chunks
  - load/evict/pending counts
  - estimated GPU allocation
- `ChunkedGSComponent.showChunkBounds` draws loaded/unloaded chunk gizmos.
- `showSplatCenters` and `centerPointSize` help verify ordering and coverage.

## How to Use
Clone the repository, open the project in Unity6 or later and create a Models folder under Assets/GaussianSplatting. Place your .ply files there and they will be automatically imported. You can then create a GameObject, add the ChunkedGSComponent, and assign the imported asset to it.

To actually see the assets, open the project settings and assign the GSPass Universal Pipeline Asset in the Quality and Graphics settings. This will ensure the custom render features are active. Then, the Settings folder and assign the ChunkedGSRenderFeature, GSMaterial and GSSorting shader to your URP renderer.

## Credits

This implementation includes or is inspired by:

- `aras-p/UnityGaussianSplatting` (data/layout and SH workflow inspiration)
- PlayCanvas GS helper math ideas (`GSUtils.hlsl` references)
- `b0nes164/GPUSorting` (`DeviceRadixSort.hlsl`, MIT)
- `wuyize25/gsplat-unity` (renderer inspiration)
