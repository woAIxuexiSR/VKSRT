# Scene Configuration

Config files define a pipeline of render passes executed in order. Each pass receives the output image of the previous pass.

```json
{
    "passes": [
        { "type": "path_tracing", "params": { "maxDepth": 8, "scene": { "type": "cornell_box" } } },
        { "type": "taa" },
        { "type": "tonemap" },
        { "type": "blit" }
    ]
}
```

Run with: `main.exe --config ../../scenes/path_tracing.json`

## Render Passes

### `ray_tracing` — Debug Visualization

Visualizes scene attributes (material, position, normal, UV). Shading mode is selectable at runtime via ImGui.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `scene` | object | cornell_box | Scene specification |

### `path_tracing` — Megakernel Path Tracing

Standard path tracer with NEE and MIS. Uses RT pipeline (raygen/miss/closesthit).

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `maxDepth` | int | 8 | Maximum bounce depth |
| `rrDepth` | int | 3 | Russian Roulette starts at this depth |
| `scene` | object | cornell_box | Scene specification |

### `light_tracing` — Light Tracing

Photon emission from light sources with camera splatting.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `maxDepth` | int | 8 | Maximum light path depth |
| `rrDepth` | int | 3 | Russian Roulette starts at this depth |
| `scene` | object | cornell_box | Scene specification |

### `wavefront_pt` — Wavefront Path Tracing

Compute-based path tracer with separate kernels per stage and indirect dispatch.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `maxDepth` | int | 8 | Maximum bounce depth |
| `rrDepth` | int | 3 | Russian Roulette starts at this depth |
| `scene` | object | cornell_box | Scene specification |

### `branch_pt` — Branching Path Tracing

Tree-based branching path tracer for stylized rendering. Supports nonlinear stylization g(integral L) with debiasing.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `maxDepth` | int | 8 | Maximum tree depth |
| `rrDepth` | int | 3 | Russian Roulette starts at this depth |
| `innerSamples` | int[] | [4, 2, 1, 1] | Branching factor per rough-bounce depth (up to 4 entries, last value repeats) |
| `useDebiasing` | int | 1 | 0=Off, 1=Jackknife, 2=Telescoping |
| `debiasR` | float | 0.5 | Geometric distribution parameter r for debiasing |
| `vramBudgetMB` | int | 4096 | VRAM budget (MB) for vertex buffer tiling |
| `scene` | object | cornell_box | Scene specification |

### `taa` — Temporal Anti-Aliasing

Frame accumulation with reprojection and neighborhood clamping.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `blendFactor` | float | 0.1 | Temporal blend weight (lower = more accumulation) |

### `bilateral` — Bilateral Denoise

Edge-preserving denoising using G-buffer (normal + position).

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `sigmaS` | float | 3.0 | Spatial sigma (pixels) |
| `sigmaN` | float | 0.1 | Normal edge-stopping threshold |
| `sigmaP` | float | 0.1 | Position edge-stopping threshold |
| `kernelRadius` | int | 5 | Filter radius (pixels) |

### `tonemap` — Tone Mapping

ACES filmic tone mapping.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `exposure` | float | 1.0 | Exposure multiplier |

### `blit` — Blit to Swapchain

Copies the pipeline result to the swapchain for display. No parameters. Should be the last pass.

## Scene Specification

Passes that load a scene accept a `scene` object in `params`:

```json
"scene": {
    "type": "cornell_box"
}
```

```json
"scene": {
    "type": "model",
    "path": "../../assets/bunny.obj",
    "scale": 0.01,
    "offset": [0, 0, 0]
}
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | "cornell_box" | `"cornell_box"` for built-in scene, `"model"` to load a file |
| `path` | string | — | Model file path (required when type=`"model"`) |
| `scale` | float | 1.0 | Uniform scale factor |
| `offset` | float[3] | [0,0,0] | Translation offset |

## Example Configs

| Config | Pipeline |
|--------|----------|
| `ray_tracing.json` | ray_tracing -> tonemap -> blit |
| `path_tracing.json` | path_tracing -> taa -> bilateral -> tonemap -> blit |
| `light_tracing.json` | light_tracing -> taa -> tonemap -> blit |
| `wavefront_pt.json` | wavefront_pt -> taa -> bilateral -> tonemap -> blit |
| `branch_pt.json` | branch_pt -> taa -> tonemap -> blit |
