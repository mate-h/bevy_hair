# bevy_hair

Deferred software rasterization for strand hair, following Lipp, Jarabo, Wimmer, and Bode, *Deferred Software Rasterization for Efficient Real-time Hair Rendering* (2026).

The renderer is a Bevy 0.19 plugin. It bakes explicit strands into layered hair-mesh bundles, generates strands in a subgroup compute shader, and rasterizes them into a 64-bit atomic G-buffer. Shading is Chiang et al. 2016 (R, TT, TRT) plus a Kajiya–Kay diffuse lobe, a 4-layer deep opacity map, a GGX prefiltered probe, irradiance spherical harmonics, baked ambient occlusion, and the paper’s reconnection filter.

Native wgpu is required. The example requests subgroup operations and 64-bit atomic min/max (`SHADER_INT64`, `SHADER_INT64_ATOMIC_MIN_MAX`). That combination is available on Metal (Apple9, or Apple8 with Mac2), Vulkan, and DX12. Browser WebGPU is out of scope.

## Run the example

```text
cargo run --example groom
```

The first launch bakes three grooms on the CPU before the window opens.

| Key | Action |
| --- | --- |
| drag / scroll | orbit and zoom |
| 1 / 2 / 3 | straight, wavy, curly |
| L | level of detail |
| F | reconnection filter |
| O | ambient occlusion |
| D | deep opacity map |
| `[` / `]` | lambda (default 3) |

## Hair models

The example uses Cem Yuksel’s woman grooms and the matching head mesh by Murat Afshar:

<https://www.cemyuksel.com/research/hairmodels>

Those files live in `assets/hair/`. Public material that shows them should link that page.
