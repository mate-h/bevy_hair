@group(0) @binding(0) var<uniform> params: HairParams;
@group(0) @binding(1) var<storage, read> center: array<u64>;
@group(0) @binding(2) var<storage, read> beta_buf: array<u32>;
@group(0) @binding(3) var<storage, read_write> shaded: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> shaded_depth: array<f32>;
@group(0) @binding(5) var<storage, read> dom: array<u32>;
@group(0) @binding(6) var<storage, read> env_tex: array<vec4<f32>>;

const IOR: f32 = 1.55;

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

fn sh_eval(n_in: vec3<f32>) -> vec3<f32> {
    let n = safe_normalize(n_in);
    let x = n.x;
    let y = n.y;
    let z = n.z;
    var e = params.sh[0].xyz * 0.282095;
    e += params.sh[1].xyz * (0.488603 * y);
    e += params.sh[2].xyz * (0.488603 * z);
    e += params.sh[3].xyz * (0.488603 * x);
    e += params.sh[4].xyz * (1.092548 * x * y);
    e += params.sh[5].xyz * (1.092548 * y * z);
    e += params.sh[6].xyz * (0.315392 * (3.0 * z * z - 1.0));
    e += params.sh[7].xyz * (1.092548 * x * z);
    e += params.sh[8].xyz * (0.546274 * (x * x - y * y));
    return max(e, vec3<f32>(0.0));
}

fn mip_info(mip: u32) -> vec4<u32> {
    if mip == 0u { return params.env_mip0; }
    if mip == 1u { return params.env_mip1; }
    if mip == 2u { return params.env_mip2; }
    return params.env_mip3;
}

fn sample_mip(mip: u32, u: f32, v: f32) -> vec4<f32> {
    let info = mip_info(mip);
    let width = max(info.y, 1u);
    let height = max(info.z, 1u);
    let x = min(u32(clamp(u, 0.0, 0.999) * f32(width)), width - 1u);
    let y = min(u32(clamp(v, 0.0, 0.999) * f32(height)), height - 1u);
    return env_tex[info.x + y * width + x];
}

fn sample_env(dir_in: vec3<f32>, roughness: f32) -> vec3<f32> {
    let dir = safe_normalize(dir_in);
    let u = atan2(dir.z, dir.x) / (2.0 * PI) + 0.5;
    let v = acos(clamp(dir.y, -1.0, 1.0)) / PI;
    let mip = clamp(roughness, 0.0, 1.0) * 3.0;
    let mip0 = u32(floor(mip));
    let mip1 = min(mip0 + 1u, 3u);
    let tm = mip - f32(mip0);
    return mix(sample_mip(mip0, u, v), sample_mip(mip1, u, v), tm).xyz;
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
    let dist = dot(world - params.lights[0].xyz, params.light_forward.xyz);
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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = u32(params.screen.x);
    let height = u32(params.screen.y);
    if id.x >= width || id.y >= height {
        return;
    }
    let pix = id.y * width + id.x;
    let word = center[pix];
    if is_empty(word) {
        shaded[pix] = vec4<f32>(0.0);
        shaded_depth[pix] = 0.0;
        return;
    }
    let unpacked = unpack_gb(word);
    let view_dist = dequantize_depth(unpacked.depth, params.screen.z, params.screen.w);
    let world = reconstruct_world(params, vec2<i32>(i32(id.x), i32(id.y)), view_dist);
    let tangent = oct_decode(unpacked.tx, unpacked.ty);
    let uvw = dequant_uvw(unpacked.uvw);
    var ao = f32(unpacked.ao) / 63.0;
    if params.flags.z == 0u {
        ao = 1.0;
    }
    let beta_word = beta_buf[pix];
    let beta = f32(beta_word & 0xffu) / 255.0;
    let roughness = max(params.appearance.x * (0.65 + 0.7 * uvw.z), 0.04);
    let tilt = params.appearance.y;
    let albedo = params.albedo.xyz * (0.65 + 0.35 * uvw.z);

    let view = safe_normalize(params.camera_pos.xyz - world);
    let fiber_n = safe_normalize(view - tangent * dot(view, tangent));
    var color = vec3<f32>(0.0);
    let vis = dom_visibility(world, beta);
    for (var i = 0u; i < 3u; i++) {
        let to_l = params.lights[i].xyz - world;
        let dist2 = max(dot(to_l, to_l), 1e-3);
        let light = to_l * inverseSqrt(dist2);
        var lobe = hair_bsdf(tangent, view, light, albedo, roughness, tilt);
        if i == 0u {
            lobe *= vis;
        }
        color += lobe * params.light_colors[i].xyz / (dist2 * 0.002 + 1.0);
    }

    let reflected = reflect(-view, fiber_n);
    let spec_env = sample_env(reflected, roughness);
    let diff_env = sh_eval(fiber_n) / PI;
    color += spec_env * mix(vec3<f32>(1.0), albedo, 0.35) * ao * 0.35;
    color += diff_env * albedo * ao;

    shaded[pix] = vec4<f32>(color, 1.0);
    shaded_depth[pix] = view_dist;
}
