// Fragment shader for `HairShadowMaterial`. A standard PBR surface whose
// `HairShadowCaster` lights are attenuated by the hair's deep opacity map from
// each light. Bevy 0.20 has no per-light shadow hook, so this runs Bevy's
// lighting and then subtracts the occluded share of each caster's direct light.

import bevy_pbr::render::{
    clustered_forward as clustering,
    forward_io::{VertexOutput, FragmentOutput},
    mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT,
    mesh_view_bindings as view_bindings,
    mesh_view_types::{DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT, POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{
        alpha_discard, apply_pbr_lighting, calculate_diffuse_color, calculate_F0_dielectric,
        main_pass_post_lighting_processing,
    },
    pbr_lighting as lighting,
    pbr_lighting::{LAYER_BASE, LAYER_CLEARCOAT},
    pbr_types::PbrInput,
    shadows,
};

// Mirrors `ShadowLight` in `pass.rs`.
struct ShadowLight {
    light_clip_from_world: mat4x4<f32>,
    light_eye: vec4<f32>,
    light_forward: vec4<f32>,
    // x near, y far, z extinction per strand fragment, w 1 once a map exists.
    depth: vec4<f32>,
    // xyz direction to a directional light (w 1) or a point or spot light's
    // position (w 2).
    source: vec4<f32>,
}

// Keep in sync with `MAX_HAIR_SHADOW_CASTERS` in `shadow.rs`.
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

// The slot whose map belongs to a Bevy light, or CHANNELS for none.
fn directional_slot(direction_to_light: vec3<f32>) -> u32 {
    for (var slot = 0u; slot < CHANNELS; slot++) {
        let source = hair_lights[slot].source;
        if hair_lights[slot].depth.w != 0.0 && source.w == 1.0
                && dot(source.xyz, direction_to_light) > 0.9999 {
            return slot;
        }
    }
    return CHANNELS;
}

fn positional_slot(position: vec3<f32>) -> u32 {
    for (var slot = 0u; slot < CHANNELS; slot++) {
        let source = hair_lights[slot].source;
        if hair_lights[slot].depth.w != 0.0 && source.w == 2.0
                && distance(source.xyz, position) <= 1e-3 * max(length(position), 1.0) {
            return slot;
        }
    }
    return CHANNELS;
}

// The direct light hair blocks from every caster, as `apply_pbr_lighting`
// computes it: Bevy's own shadow times the hair's opacity.
fn occluded_light(in: PbrInput) -> vec3<f32> {
    if (in.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) == 0u {
        return vec3<f32>(0.0);
    }
    var blocked: array<f32, CHANNELS>;
    var any_blocked = false;
    for (var slot = 0u; slot < CHANNELS; slot++) {
        blocked[slot] = 1.0 - hair_visibility(slot, in.world_position.xyz);
        any_blocked = any_blocked || blocked[slot] > 0.0;
    }
    if !any_blocked {
        return vec3<f32>(0.0);
    }

    let base_color = in.material.base_color.rgb;
    let perceptual_roughness = in.material.perceptual_roughness;
    let NdotV = max(dot(in.N, in.V), 0.0001);
    var input: lighting::LightingInput;
    input.layers[LAYER_BASE].NdotV = NdotV;
    input.layers[LAYER_BASE].N = in.N;
    input.layers[LAYER_BASE].R = reflect(-in.V, in.N);
    input.layers[LAYER_BASE].perceptual_roughness = perceptual_roughness;
    input.layers[LAYER_BASE].roughness = lighting::perceptualRoughnessToRoughness(perceptual_roughness);
    input.P = in.world_position.xyz;
    input.V = in.V;
    input.diffuse_color = calculate_diffuse_color(
        base_color,
        in.material.metallic,
        in.material.specular_transmission,
        in.material.diffuse_transmission,
    );
    input.metallic = in.material.metallic;
    input.F0_dielectric = calculate_F0_dielectric(in.material.reflectance);
    input.F0_metallic = base_color;
    input.F_ab = lighting::F_AB(perceptual_roughness, NdotV);
@if(STANDARD_MATERIAL_CLEARCOAT) {
    let clearcoat_perceptual_roughness = in.material.clearcoat_perceptual_roughness;
    input.layers[LAYER_CLEARCOAT].NdotV = max(dot(in.clearcoat_N, in.V), 0.0001);
    input.layers[LAYER_CLEARCOAT].N = in.clearcoat_N;
    input.layers[LAYER_CLEARCOAT].R = reflect(-in.V, in.clearcoat_N);
    input.layers[LAYER_CLEARCOAT].perceptual_roughness = clearcoat_perceptual_roughness;
    input.layers[LAYER_CLEARCOAT].roughness =
        lighting::perceptualRoughnessToRoughness(clearcoat_perceptual_roughness);
    input.clearcoat_strength = in.material.clearcoat;
}
@if(STANDARD_MATERIAL_ANISOTROPY) {
    input.anisotropy = in.anisotropy_strength;
    input.Ta = in.anisotropy_T;
    input.Ba = in.anisotropy_B;
}

    let view_z = dot(vec4<f32>(
        view_bindings::view.view_from_world[0].z,
        view_bindings::view.view_from_world[1].z,
        view_bindings::view.view_from_world[2].z,
        view_bindings::view.view_from_world[3].z
    ), in.world_position);
    let cluster_index = clustering::view_fragment_cluster_index(in.frag_coord.xy, view_z, in.is_orthographic);
    let ranges = clustering::unpack_clusterable_object_index_ranges(cluster_index);

    var removed = vec3<f32>(0.0);
    for (var i = ranges.first_point_light_index_offset; i < ranges.first_spot_light_index_offset; i++) {
        let id = clustering::get_clusterable_object_id(i);
        let light = &view_bindings::clustered_lights.data[id];
        let slot = positional_slot((*light).position_radius.xyz);
        if slot == CHANNELS {
            continue;
        }
        var shadow = 1.0;
        if ((*light).flags & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = shadows::fetch_point_shadow(id, in.world_position, in.world_normal, in.frag_coord.xy);
        }
        removed += lighting::point_light(id, &input, true, true) * shadow * blocked[slot];
    }
    for (var i = ranges.first_spot_light_index_offset; i < ranges.first_rect_light_index_offset; i++) {
        let id = clustering::get_clusterable_object_id(i);
        let light = &view_bindings::clustered_lights.data[id];
        let slot = positional_slot((*light).position_radius.xyz);
        if slot == CHANNELS {
            continue;
        }
        var shadow = 1.0;
        if ((*light).flags & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = shadows::fetch_spot_shadow(
                id,
                in.world_position,
                in.world_normal,
                (*light).shadow_map_near_z,
                in.frag_coord.xy,
            );
        }
        removed += lighting::spot_light(id, &input, true) * shadow * blocked[slot];
    }
    for (var i = 0u; i < view_bindings::lights.n_directional_lights; i++) {
        let light = &view_bindings::lights.directional_lights[i];
        let slot = directional_slot((*light).direction_to_light);
        if slot == CHANNELS {
            continue;
        }
        var shadow = 1.0;
        if ((*light).flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = shadows::fetch_directional_shadow(i, in.world_position, in.world_normal, view_z, in.frag_coord.xy);
        }
        removed += lighting::directional_light(i, &input, true) * shadow * blocked[slot];
    }
    return removed * view_bindings::view.exposure;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color,
    );

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = vec4<f32>(max(out.color.rgb - occluded_light(pbr_input), vec3<f32>(0.0)), out.color.a);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
