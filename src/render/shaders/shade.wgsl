#import bevy_core_pipeline::fullscreen_vertex_shader::{
    FullscreenVertexOutput, fullscreen_vertex_shader,
}
#import bevy_pbr::clustered_forward as clustering
#import bevy_pbr::lighting::{getDistanceAttenuation, getRangeFalloff}
#import bevy_pbr::mesh_view_bindings as view_bindings
#import bevy_pbr::mesh_view_types::{
    DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT, POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
    POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE,
}
#import bevy_pbr::shadows::{fetch_directional_shadow, fetch_point_shadow, fetch_spot_shadow}

@group(2) @binding(0) var<uniform> params: HairParams;
@group(2) @binding(1) var<storage, read> center: array<u64>;
@group(2) @binding(2) var<storage, read> beta_buf: array<u32>;
@group(2) @binding(3) var<storage, read_write> shaded: array<vec4<f32>>;
@group(2) @binding(4) var<storage, read_write> shaded_depth: array<f32>;
@group(2) @binding(5) var<storage, read> dom: array<u32>;

const IOR: f32 = 1.55;

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32) -> FullscreenVertexOutput {
    return fullscreen_vertex_shader(vertex_index);
}

fn logistic(x: f32, s: f32) -> f32 {
    let ss = max(s, 1e-3);
    let d = exp(-abs(x) / ss);
    return d / (ss * (1.0 + d) * (1.0 + d));
}

fn logistic_cdf(x: f32, s: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x / max(s, 1e-3)));
}

fn trimmed_logistic(x: f32, s: f32) -> f32 {
    let a = max(logistic_cdf(PI, s) - logistic_cdf(-PI, s), 1e-4);
    return logistic(x, s) / a;
}

fn longitudinal(sin_i: f32, sin_o: f32, shift: f32, roughness: f32) -> f32 {
    let theta_i = asin(clamp(sin_i, -1.0, 1.0));
    let theta_o = asin(clamp(sin_o, -1.0, 1.0));
    return trimmed_logistic(theta_i + theta_o - shift, roughness);
}

fn hair_bsdf(tangent: vec3<f32>, view: vec3<f32>, light: vec3<f32>, albedo: vec3<f32>, roughness: f32, tilt: f32) -> vec3<f32> {
    let t = safe_normalize(tangent);
    let v = safe_normalize(view);
    let l = safe_normalize(light);
    let sin_i = clamp(dot(t, l), -1.0, 1.0);
    let sin_o = clamp(dot(t, v), -1.0, 1.0);
    let cos_i = sqrt(max(1.0 - sin_i * sin_i, 0.0));
    let t_i = safe_normalize(l - t * sin_i);
    let t_o = safe_normalize(v - t * sin_o);
    let phi = acos(clamp(dot(t_i, t_o), -1.0, 1.0));
    let beta = max(roughness, 0.04);
    let f0 = pow((1.0 - IOR) / (1.0 + IOR), 2.0);
    let fres = f0 + (1.0 - f0) * pow(1.0 - cos_i, 5.0);
    let m_r = longitudinal(sin_i, sin_o, tilt, beta);
    let m_tt = longitudinal(sin_i, sin_o, -tilt * 0.5, beta * 0.5);
    let m_trt = longitudinal(sin_i, sin_o, -tilt * 1.5, beta * 2.0);
    let azimuth_r = 0.25 * cos(phi * 0.5);
    let n_r = fres * azimuth_r;
    let n_tt = (1.0 - fres) * albedo * (1.0 / (2.0 * PI));
    let n_trt = fres * (1.0 - fres) * (1.0 - fres) * albedo * albedo * azimuth_r;
    let spec = m_r * n_r + m_tt * n_tt + m_trt * n_trt;
    // Kajiya-Kay diffuse stands in for multiple scattering.
    let diffuse = albedo * cos_i / PI;
    return spec * cos_i + diffuse * 0.35;
}

fn dom_visibility(world: vec3<f32>, beta: f32) -> f32 {
    if params.flags.w == 0u {
        return 1.0;
    }
    let clip = params.light_clip_from_world * vec4<f32>(world, 1.0);
    if clip.w <= 0.0 {
        return 1.0;
    }
    let ndc = clip.xy / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
    if uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0 {
        return 1.0;
    }
    let dom_w = u32(params.dom_info.x);
    let dom_h = u32(params.dom_info.y);
    let coord = vec2<u32>(
        min(u32(uv.x * f32(dom_w)), dom_w - 1u),
        min(u32(uv.y * f32(dom_h)), dom_h - 1u),
    );
    let near = params.dom_info.z;
    let far = params.dom_info.w;
    let dist = dot(world - params.light_eye.xyz, params.light_forward.xyz);
    let z = clamp((dist - near) / max(far - near, 1e-3), 0.0, 1.0);
    // Eq. 8, scaled into the light-view depth range.
    let delta = -log(max(beta, 1e-4)) * params.filter_params.z;
    let z_biased = clamp(z - delta / max(far - near, 1e-3), 0.0, 1.0);
    var occ = 0.0;
    let base = (coord.y * dom_w + coord.x) * 4u;
    for (var slice = 0u; slice < 4u; slice++) {
        let slice_z = (f32(slice) + 0.5) / 4.0;
        if slice_z < z_biased {
            occ += f32(dom[base + slice]);
        }
    }
    return exp(-occ * params.filter_params.w);
}

fn spot_mask(light_id: u32, world: vec3<f32>) -> f32 {
    let light = &view_bindings::clustered_lights.data[light_id];
    var spot_dir = vec3<f32>((*light).light_custom_data.x, 0.0, (*light).light_custom_data.y);
    spot_dir.y = sqrt(max(0.0, 1.0 - spot_dir.x * spot_dir.x - spot_dir.z * spot_dir.z));
    if ((*light).flags & POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE) != 0u {
        spot_dir.y = -spot_dir.y;
    }
    let light_to_frag = (*light).position_radius.xyz - world;
    let cd = dot(-spot_dir, safe_normalize(light_to_frag));
    let attenuation = clamp(
        cd * (*light).light_custom_data.z + (*light).light_custom_data.w,
        0.0,
        1.0,
    );
    return attenuation * attenuation;
}

fn quat_rotate(q: vec4<f32>, dir: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, dir);
    return dir + q.w * t + cross(q.xyz, t);
}

fn cubemap_dir(dir: vec3<f32>) -> vec3<f32> {
    var sample_dir = quat_rotate(view_bindings::light_probes.view_rotation, dir);
    // Cubemaps are left-handed.
    sample_dir.z = -sample_dir.z;
    return sample_dir;
}

fn probe_radiance(fiber_n: vec3<f32>, view: vec3<f32>, roughness: f32) -> vec3<f32> {
#ifdef ENVIRONMENT_MAP
    if view_bindings::light_probes.view_cubemap_index < 0 {
        return vec3<f32>(0.0);
    }
    let spec_dir = cubemap_dir(reflect(-view, fiber_n));
    let diff_dir = cubemap_dir(fiber_n);
#ifdef MULTIPLE_LIGHT_PROBES_IN_ARRAY
    let index = u32(view_bindings::light_probes.view_cubemap_index);
    let last_mip = f32(textureNumLevels(view_bindings::specular_environment_maps[index]) - 1u);
    let spec = textureSampleLevel(
        view_bindings::specular_environment_maps[index],
        view_bindings::environment_map_sampler,
        spec_dir,
        roughness * last_mip,
    ).rgb;
    let diff = textureSampleLevel(
        view_bindings::diffuse_environment_maps[index],
        view_bindings::environment_map_sampler,
        diff_dir,
        0.0,
    ).rgb;
#else
    let last_mip = f32(view_bindings::light_probes.smallest_specular_mip_level_for_view);
    let spec = textureSampleLevel(
        view_bindings::specular_environment_map,
        view_bindings::environment_map_sampler,
        spec_dir,
        roughness * last_mip,
    ).rgb;
    let diff = textureSampleLevel(
        view_bindings::diffuse_environment_map,
        view_bindings::environment_map_sampler,
        diff_dir,
        0.0,
    ).rgb;
#endif
    // One specular lobe stands in for R and TRT. The diffuse cubemap is already irradiance.
    return (spec + diff) * view_bindings::light_probes.intensity_for_view;
#else
    // This specialization has no cubemap bindings. Keep the arguments referenced.
    return vec3<f32>(dot(fiber_n, view) * roughness * 0.0);
#endif
}

fn direct_light(
    world: vec3<f32>,
    tangent: vec3<f32>,
    view: vec3<f32>,
    fiber_n: vec3<f32>,
    albedo: vec3<f32>,
    roughness: f32,
    tilt: f32,
    beta: f32,
    frag_xy: vec2<f32>,
    view_z: f32,
    ortho: bool,
) -> vec3<f32> {
    let world_h = vec4<f32>(world, 1.0);
    let cluster_index = clustering::view_fragment_cluster_index(frag_xy, view_z, ortho);
    let ranges = clustering::unpack_clusterable_object_index_ranges(cluster_index);
    var color = vec3<f32>(0.0);

    for (var i = ranges.first_point_light_index_offset; i < ranges.first_spot_light_index_offset; i++) {
        let light_id = clustering::get_clusterable_object_id(i);
        let light = &view_bindings::clustered_lights.data[light_id];
        let light_to_frag = (*light).position_radius.xyz - world;
        let dist2 = dot(light_to_frag, light_to_frag);
        let incident = safe_normalize(light_to_frag);
        let attenuation = getDistanceAttenuation(dist2, (*light).color_inverse_square_range.w);
        var shadow = 1.0;
        if ((*light).flags & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = fetch_point_shadow(light_id, world_h, fiber_n, frag_xy);
        }
        let lobe = hair_bsdf(tangent, view, incident, albedo, roughness, tilt);
        color += lobe * (*light).color_inverse_square_range.rgb * attenuation * shadow;
    }

    for (var i = ranges.first_spot_light_index_offset; i < ranges.first_reflection_probe_index_offset; i++) {
        let light_id = clustering::get_clusterable_object_id(i);
        let light = &view_bindings::clustered_lights.data[light_id];
        let light_to_frag = (*light).position_radius.xyz - world;
        let dist2 = dot(light_to_frag, light_to_frag);
        let incident = safe_normalize(light_to_frag);
        let attenuation = getDistanceAttenuation(dist2, (*light).color_inverse_square_range.w);
        var shadow = 1.0;
        if ((*light).flags & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = fetch_spot_shadow(light_id, world_h, fiber_n, (*light).shadow_map_near_z, frag_xy);
        }
        let lobe = hair_bsdf(tangent, view, incident, albedo, roughness, tilt);
        color += lobe * (*light).color_inverse_square_range.rgb * attenuation * spot_mask(light_id, world) * shadow;
    }

    var dom_left = true;
    let n_directional = view_bindings::lights.n_directional_lights;
    for (var i = 0u; i < n_directional; i++) {
        let light = &view_bindings::lights.directional_lights[i];
        let incident = (*light).direction_to_light;
        var shadow = 1.0;
        let casts = ((*light).flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u;
        if casts {
            shadow = fetch_directional_shadow(i, world_h, fiber_n, view_z, frag_xy);
        }
        if dom_left && casts {
            shadow *= dom_visibility(world, beta);
            dom_left = false;
        }
        let lobe = hair_bsdf(tangent, view, incident, albedo, roughness, tilt);
        color += lobe * (*light).color.rgb * shadow;
    }

#ifdef AREA_LIGHT_LUTS
    let n_rect = view_bindings::lights.n_rect_lights;
    for (var i = 0u; i < n_rect; i++) {
        let light = &view_bindings::lights.rect_lights[i];
        let to_center = (*light).position - world;
        let dist2 = max(dot(to_center, to_center), 1e-4);
        let incident = to_center * inverseSqrt(dist2);
        let normal = safe_normalize(cross((*light).up, (*light).right));
        let facing = max(dot(normal, incident), 0.0);
        let solid = (*light).width * (*light).height * facing / dist2;
        let range2 = max((*light).range * (*light).range, 1e-4);
        let falloff = getRangeFalloff(dist2, 1.0 / range2);
        let lobe = hair_bsdf(tangent, view, incident, albedo, roughness, tilt);
        color += lobe * (*light).color.rgb * solid * falloff;
    }
#endif

    return color;
}

@fragment
fn fs(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let width = u32(params.screen.x);
    let height = u32(params.screen.y);
    let x = u32(in.position.x);
    let y = u32(in.position.y);
    if x >= width || y >= height {
        return vec4<f32>(0.0);
    }
    let pix = y * width + x;
    let word = center[pix];
    if is_empty(word) {
        shaded[pix] = vec4<f32>(0.0);
        shaded_depth[pix] = 0.0;
        return vec4<f32>(0.0);
    }
    let unpacked = unpack_gb(word);
    let view_dist = dequantize_depth(unpacked.depth, params.screen.z, params.screen.w);
    let world = reconstruct_world(params, vec2<i32>(i32(x), i32(y)), view_dist);
    let tangent = oct_decode(unpacked.tx, unpacked.ty);
    let uvw = dequant_uvw(unpacked.uvw);
    var ao = f32(unpacked.ao) / 63.0;
    if params.flags.z == 0u {
        ao = 1.0;
    }
    let beta = f32(beta_buf[pix] & 0xffu) / 255.0;
    let roughness = max(params.appearance.x * (0.65 + 0.7 * uvw.z), 0.04);
    let tilt = params.appearance.y;
    let albedo = params.albedo.xyz * (0.65 + 0.35 * uvw.z);

    let world_h = vec4<f32>(world, 1.0);
    let view_z = dot(vec4<f32>(
        view_bindings::view.view_from_world[0].z,
        view_bindings::view.view_from_world[1].z,
        view_bindings::view.view_from_world[2].z,
        view_bindings::view.view_from_world[3].z,
    ), world_h);
    let ortho = view_bindings::view.clip_from_view[3].w == 1.0;
    var view: vec3<f32>;
    if ortho {
        view = safe_normalize(vec3<f32>(
            view_bindings::view.clip_from_world[0].z,
            view_bindings::view.clip_from_world[1].z,
            view_bindings::view.clip_from_world[2].z,
        ));
    } else {
        view = safe_normalize(view_bindings::view.world_position.xyz - world);
    }
    let fiber_n = safe_normalize(view - tangent * dot(view, tangent));

    var color = direct_light(
        world,
        tangent,
        view,
        fiber_n,
        albedo,
        roughness,
        tilt,
        beta,
        in.position.xy,
        view_z,
        ortho,
    );
    color += probe_radiance(fiber_n, view, roughness) * albedo * ao;
    // The mesh pipeline scales physical radiance by exposure before the HDR target.
    color *= view_bindings::view.exposure;

    shaded[pix] = vec4<f32>(color, 1.0);
    shaded_depth[pix] = view_dist;
    return vec4<f32>(0.0);
}
