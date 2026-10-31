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

// Karis 2016, "Physically Based Hair Shading in Unreal". Longitudinal lobes are
// Gaussians in θi + θo. Azimuthal TT and TRT use his fitted exponentials, and
// absorption raises the albedo to the internal path length.
const SCATTER: f32 = 0.5;

fn gaussian(b: f32, x: f32) -> f32 {
    return exp(-0.5 * x * x / (b * b)) / (sqrt(2.0 * PI) * b);
}

fn fresnel(cos_t: f32) -> f32 {
    let f0 = pow((1.0 - IOR) / (1.0 + IOR), 2.0);
    return f0 + (1.0 - f0) * pow(1.0 - clamp(cos_t, 0.0, 1.0), 5.0);
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Light that reaches the fiber through other hair is tinted by every bounce.
// The more of the direct light is shadowed, the more saturated the remainder.
fn scatter_tint(albedo: vec3<f32>, shadow: f32) -> vec3<f32> {
    // Half strength: dark linear albedos have extreme hue ratios.
    let hue = albedo / max(luminance(albedo), 1e-4);
    return pow(hue, vec3<f32>(0.5 - 0.5 * clamp(shadow, 0.0, 1.0)));
}

fn hair_bsdf(
    tangent: vec3<f32>,
    view: vec3<f32>,
    light: vec3<f32>,
    fiber_n: vec3<f32>,
    albedo: vec3<f32>,
    roughness: f32,
    tilt: f32,
    shadow: f32,
) -> vec3<f32> {
    let t = safe_normalize(tangent);
    let v = safe_normalize(view);
    let l = safe_normalize(light);
    let sin_i = clamp(dot(t, l), -1.0, 1.0);
    let sin_o = clamp(dot(t, v), -1.0, 1.0);
    let theta_i = asin(sin_i);
    let theta_o = asin(sin_o);
    let theta_h = theta_i + theta_o;
    let cos_theta_d = max(cos(0.5 * abs(theta_i - theta_o)), 1e-3);
    let t_i = l - t * sin_i;
    let t_o = v - t * sin_o;
    let proj = sqrt(max(dot(t_i, t_i) * dot(t_o, t_o), 1e-8));
    let cos_phi = clamp(dot(t_i, t_o) / proj, -1.0, 1.0);
    let cos_half_phi = sqrt(clamp(0.5 + 0.5 * cos_phi, 0.0, 1.0));
    let b = clamp(roughness * roughness, 0.02, 1.0);

    // R reflects off the cuticle and keeps the light's color.
    let m_r = gaussian(b, theta_h - tilt);
    let n_r = 0.25 * cos_half_phi;
    let f_r = fresnel(sqrt(clamp(0.5 + 0.5 * dot(v, l), 0.0, 1.0)));
    var spec = vec3<f32>(m_r * n_r * f_r);

    // TT passes through the fiber and glows when the hair is backlit.
    let m_tt = gaussian(0.5 * b, theta_h + 0.5 * tilt);
    let eta = 1.19 / cos_theta_d + 0.36 * cos_theta_d;
    let a = 1.0 / eta;
    let h = cos_half_phi * (1.0 + a * (0.6 - 0.8 * cos_phi));
    let f_tt = fresnel(cos_theta_d * sqrt(clamp(1.0 - h * h, 0.0, 1.0)));
    let t_tt = pow(albedo, vec3<f32>(0.5 * sqrt(clamp(1.0 - h * h * a * a, 0.0, 1.0)) / cos_theta_d));
    let n_tt = exp(-3.65 * cos_phi - 3.98);
    spec += m_tt * n_tt * (1.0 - f_tt) * (1.0 - f_tt) * t_tt;

    // TRT is the colored secondary highlight, shifted toward the tip.
    let m_trt = gaussian(2.0 * b, theta_h + 1.5 * tilt);
    let f_trt = fresnel(cos_theta_d * 0.5);
    let t_trt = pow(albedo, vec3<f32>(0.8 / cos_theta_d));
    let n_trt = exp(17.0 * cos_phi - 16.78);
    spec += m_trt * n_trt * (1.0 - f_trt) * (1.0 - f_trt) * f_trt * t_trt;

    // Multiple scattering: wrapped Kajiya-Kay on the view-facing fiber normal.
    let kajiya = 1.0 - abs(sin_i);
    let wrapped = clamp((dot(fiber_n, l) + 1.0) * 0.25, 0.0, 1.0);
    let diffuse = albedo * scatter_tint(albedo, shadow) * (SCATTER / PI) * mix(wrapped, kajiya, 0.33);

    return (spec + diffuse) * shadow;
}

fn dom_optical_depth(pix: u32, t: f32) -> f32 {
    let pixels = u32(params.dom_info.x) * u32(params.dom_info.y);
    let base = pixels + pix * DOM_LAYERS;
    var occ = 0.0;
    for (var slice = 0u; slice < DOM_LAYERS; slice++) {
        let count = f32(dom[base + slice]);
        let i = f32(slice);
        if i + 1.0 <= t {
            occ += count;
        } else if i < t {
            occ += count * (t - i);
        }
    }
    return occ;
}

fn dom_texel_visibility(x: i32, y: i32, dist: f32, near: f32, far: f32, dom_w: u32, dom_h: u32) -> f32 {
    if x < 0 || y < 0 || x >= i32(dom_w) || y >= i32(dom_h) {
        return 1.0;
    }
    let pix = u32(y) * dom_w + u32(x);
    let front_q = dom[pix];
    if front_q == DOM_EMPTY {
        return 1.0;
    }
    let z_front = dequantize_depth(front_q, near, far);
    let thickness = max(far - near, 1e-3) / f32(DOM_LAYERS);
    // A small slack keeps the front surface from reading the opacity it just
    // wrote when reconstruction sits a fraction of a layer behind the map.
    let rel = max(dist - z_front - thickness * 0.05, 0.0);
    return exp(-dom_optical_depth(pix, rel / thickness) * params.filter_params.w);
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
    let near = params.dom_info.z;
    let far = params.dom_info.w;
    // Eq. 8. `-log(β)` hair-mesh layers toward the light.
    let dist = dot(world - params.light_eye.xyz, params.light_forward.xyz)
        - (-log(max(beta, 1e-4)) * params.filter_params.z);
    let fx = uv.x * f32(dom_w) - 0.5;
    let fy = uv.y * f32(dom_h) - 0.5;
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let tx = fract(fx);
    let ty = fract(fy);
    let v00 = dom_texel_visibility(x0, y0, dist, near, far, dom_w, dom_h);
    let v10 = dom_texel_visibility(x0 + 1, y0, dist, near, far, dom_w, dom_h);
    let v01 = dom_texel_visibility(x0, y0 + 1, dist, near, far, dom_w, dom_h);
    let v11 = dom_texel_visibility(x0 + 1, y0 + 1, dist, near, far, dom_w, dom_h);
    return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
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

fn probe_radiance(
    tangent: vec3<f32>,
    fiber_n: vec3<f32>,
    view: vec3<f32>,
    albedo: vec3<f32>,
    roughness: f32,
    tilt: f32,
) -> vec3<f32> {
#ifdef ENVIRONMENT_MAP
    if view_bindings::light_probes.view_cubemap_index < 0 {
        return vec3<f32>(0.0);
    }
    // The R cone sits at θi = tilt - θo. Sample its line facing the camera.
    let theta_o = asin(clamp(dot(tangent, view), -1.0, 1.0));
    let theta_i = tilt - theta_o;
    let spec_dir = cubemap_dir(tangent * sin(theta_i) + fiber_n * cos(theta_i));
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
    // The azimuthal R lobe integrates to one, so R is the Fresnel-weighted
    // prefiltered radiance. TRT shares the lookup and adds its absorption.
    // TT from behind is mostly blocked by the head and is left to scattering.
    let cos_o = max(cos(theta_o), 1e-3);
    let f_r = fresnel(cos_o);
    let f_trt = fresnel(cos_o * 0.5);
    let trt = 0.76 * (1.0 - f_trt) * (1.0 - f_trt) * f_trt * pow(albedo, vec3<f32>(0.8 / cos_o));
    // The diffuse cubemap is already cosine-weighted irradiance over π.
    let scatter = albedo * scatter_tint(albedo, 0.5);
    return (spec * (f_r + trt) + diff * scatter) * view_bindings::light_probes.intensity_for_view;
#else
    // This specialization has no cubemap bindings. Keep the arguments referenced.
    return vec3<f32>(dot(fiber_n + tangent, view) * roughness * tilt * albedo.x * 0.0);
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
        let lobe = hair_bsdf(tangent, view, incident, fiber_n, albedo, roughness, tilt, shadow);
        color += lobe * (*light).color_inverse_square_range.rgb * attenuation;
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
        let lobe = hair_bsdf(tangent, view, incident, fiber_n, albedo, roughness, tilt, shadow);
        color += lobe * (*light).color_inverse_square_range.rgb * attenuation * spot_mask(light_id, world);
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
        let lobe = hair_bsdf(tangent, view, incident, fiber_n, albedo, roughness, tilt, shadow);
        color += lobe * (*light).color.rgb;
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
        let lobe = hair_bsdf(tangent, view, incident, fiber_n, albedo, roughness, tilt, 1.0);
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
    color += probe_radiance(tangent, fiber_n, view, albedo, roughness, tilt) * ao;
    // The mesh pipeline scales physical radiance by exposure before the HDR target.
    color *= view_bindings::view.exposure;

    shaded[pix] = vec4<f32>(color, 1.0);
    shaded_depth[pix] = view_dist;
    return vec4<f32>(0.0);
}
