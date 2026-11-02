// Fragment shader for `HairShadowMaterial`. A standard PBR surface whose lights
// on material shadow channels are attenuated by the hair's deep opacity map from
// that light.

import bevy_pbr::render::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
};

// Mirrors `ShadowLight` in `pass.rs`.
struct ShadowLight {
    light_clip_from_world: mat4x4<f32>,
    light_eye: vec4<f32>,
    light_forward: vec4<f32>,
    // x near, y far, z extinction per strand fragment, w 1 once a map exists.
    depth: vec4<f32>,
}

// Keep in sync with `MAX_MATERIAL_SHADOW_CHANNELS` in Bevy.
const CHANNELS: u32 = 4u;

// One map of `DOM_WORDS` per channel.
@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<storage, read> hair_dom: array<u32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var<storage, read> hair_lights: array<ShadowLight, CHANNELS>;

// Keep in sync with `DOM_SIZE` and `DOM_LAYERS` in `pass.rs`.
const DOM_SIZE: u32 = 512u;
const DOM_LAYERS: u32 = 16u;
const DOM_EMPTY: u32 = 0xffffffffu;
const DOM_WORDS: u32 = DOM_SIZE * DOM_SIZE * (1u + DOM_LAYERS);

// Same slice integration as `dom_texel_visibility` in `shade.wesl`, without the
// hair-mesh layer offset: a surface sits behind every strand in front of it.
fn texel_visibility(channel: u32, x: i32, y: i32, dist: f32) -> f32 {
    if x < 0 || y < 0 || x >= i32(DOM_SIZE) || y >= i32(DOM_SIZE) {
        return 1.0;
    }
    let light = hair_lights[channel];
    let map = channel * DOM_WORDS;
    let pix = u32(y) * DOM_SIZE + u32(x);
    let front_q = hair_dom[map + pix];
    if front_q == DOM_EMPTY {
        return 1.0;
    }
    let near = light.depth.x;
    let far = light.depth.y;
    let z_front = mix(near, far, f32(front_q) / 16777214.0);
    let thickness = max(far - near, 1e-3) / f32(DOM_LAYERS);
    let t = max(dist - z_front, 0.0) / thickness;
    let base = map + DOM_SIZE * DOM_SIZE + pix * DOM_LAYERS;
    var hits = 0.0;
    for (var slice = 0u; slice < DOM_LAYERS; slice++) {
        hits += f32(hair_dom[base + slice]) * clamp(t - f32(slice), 0.0, 1.0);
    }
    return exp(-hits * light.depth.z);
}

fn hair_visibility(channel: u32, world: vec3<f32>) -> f32 {
    let light = hair_lights[channel];
    if light.depth.w == 0.0 {
        return 1.0;
    }
    let clip = light.light_clip_from_world * vec4<f32>(world, 1.0);
    if clip.w <= 0.0 {
        return 1.0;
    }
    let ndc = clip.xy / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return 1.0;
    }
    let dist = dot(world - light.light_eye.xyz, light.light_forward.xyz);
    let f = uv * f32(DOM_SIZE) - 0.5;
    let x0 = i32(floor(f.x));
    let y0 = i32(floor(f.y));
    let w = fract(f);
    let v00 = texel_visibility(channel, x0, y0, dist);
    let v10 = texel_visibility(channel, x0 + 1, y0, dist);
    let v01 = texel_visibility(channel, x0, y0 + 1, dist);
    let v11 = texel_visibility(channel, x0 + 1, y0 + 1, dist);
    return mix(mix(v00, v10, w.x), mix(v01, v11, w.x), w.y);
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color,
    );
    for (var channel = 0u; channel < CHANNELS; channel++) {
        pbr_input.material_shadow_factors[channel] =
            hair_visibility(channel, in.world_position.xyz);
    }

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
