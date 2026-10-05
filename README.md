# bevy_hair

Real-time strand hair with deferred software rasterization on hair meshes.

The renderer is a Bevy 0.19 plugin. It bakes explicit strands into layered hair-mesh bundles, generates strands in a subgroup compute shader, and rasterizes them into a 64-bit atomic G-buffer. Shading evaluates Chiang's R, TT, and TRT lobes plus a Kajiya–Kay diffuse term for every clustered light, with Bevy shadow maps and a 4-layer deep opacity map on the key directional. The camera environment map supplies one prefiltered specular lookup and an irradiance lookup. A reconnection filter reconstructs coverage.

Native wgpu is required. The example requests subgroup operations and 64-bit atomic min/max (`SHADER_INT64`, `SHADER_INT64_ATOMIC_MIN_MAX`). That combination is available on Metal (Apple9, or Apple8 with Mac2), Vulkan, and DX12. Browser WebGPU is out of scope.

## Run the example

```text
cargo run --example groom
```

`cargo run --example bake` writes the three grooms to `target/groom-cache/` without opening a window. The viewer bakes any missing groom on first launch, then loads the cache until a hair file or the baker changes.

| Key | Action |
| --- | --- |
| right-drag | look around |
| WASD / Q E | move, and rise or fall |
| scroll | change fly speed |
| shift | move faster |
| 1 / 2 / 3 | straight, wavy, curly |
| L | level of detail |
| F | reconnection filter |
| O | ambient occlusion |
| M | deep opacity map |
| `[` / `]` | lambda (default 3) |

## Hair models

The example uses Cem Yuksel’s woman grooms and the matching head mesh by Murat Afshar:

<https://www.cemyuksel.com/research/hairmodels>

Those files live in `assets/hair/`. Public material that shows them should link that page.

## Reference

Lipp, Jarabo, Wimmer, and Bode (2026). *Deferred Software Rasterization for Efficient Real-time Hair Rendering.* Proceedings of the ACM on Computer Graphics and Interactive Techniques, 9(4).

<https://arxiv.org/abs/2607.04230>

<https://doi.org/10.1145/3820015>
