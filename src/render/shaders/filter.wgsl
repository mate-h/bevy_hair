@group(0) @binding(0) var<uniform> params: HairParams;
@group(0) @binding(1) var<storage, read> center: array<u64>;
@group(0) @binding(2) var<storage, read> conservative: array<u64>;
@group(0) @binding(3) var<storage, read> shaded: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> shaded_depth: array<f32>;
@group(0) @binding(5) var<storage, read_write> filtered: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> filtered_depth: array<f32>;

fn screen_tangent(params: HairParams, world: vec3<f32>, tangent: vec3<f32>) -> vec2<f32> {
    let a = project_world(params, world);
    let b = project_world(params, world + tangent);
    let d = b.screen - a.screen;
    let len = length(d);
    if len < 1e-3 {
        return vec2<f32>(1.0, 0.0);
    }
    return d / len;
}

@compute @workgroup_size(8, 4, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = i32(params.screen.x);
    let height = i32(params.screen.y);
    if i32(id.x) >= width || i32(id.y) >= height {
        return;
    }
    let p = vec2<i32>(i32(id.x), i32(id.y));
    let pix = id.y * u32(width) + id.x;
    let center_word = center[pix];
    let cons_word = conservative[pix];
    let center_hit = !is_empty(center_word);
    let cons_hit = !is_empty(cons_word);
    if !center_hit && !cons_hit {
        filtered[pix] = vec4<f32>(0.0);
        filtered_depth[pix] = 0.0;
        return;
    }

    var tangent = vec3<f32>(0.0, 1.0, 0.0);
    var depth_q = 0u;
    if center_hit {
        let unpacked = unpack_gb(center_word);
        tangent = oct_decode(unpacked.tx, unpacked.ty);
        depth_q = unpacked.depth;
    } else {
        let unpacked = unpack_gb(cons_word);
        tangent = oct_decode(unpacked.tx, unpacked.ty);
        depth_q = unpacked.depth;
    }
    let view_dist = dequantize_depth(depth_q, params.screen.z, params.screen.w);
    let world = reconstruct_world(params, p, view_dist);
    let t_screen = screen_tangent(params, world, tangent);
    let center_color = shaded[pix];

    let radius = 5;
    let s_par = max(params.filter_params.x, 0.05);
    let s_perp = max(params.filter_params.y, 0.02);
    let sigma_par = f32(radius) * s_par;
    let sigma_perp = f32(radius) * s_perp;
    let sigma_c = 0.9;
    let depth_reject = 1.45e-3;
    let closer_eps = 5e-4;
    let strip = 0.5;

    var accum = vec3<f32>(0.0);
    var weight = 0.0;
    var h_center = 0.0;
    var h_cons = 0.0;
    let promote = !center_hit;

    for (var dy = -radius; dy <= radius; dy++) {
        for (var dx = -radius; dx <= radius; dx++) {
            let q = p + vec2<i32>(dx, dy);
            if q.x < 0 || q.y < 0 || q.x >= width || q.y >= height {
                continue;
            }
            let qpix = u32(q.y) * u32(width) + u32(q.x);
            let d = vec2<f32>(f32(dx), f32(dy));
            let d_par = dot(d, t_screen);
            let d_perp = length(d - t_screen * d_par);
            let q_center = center[qpix];
            let q_cons = conservative[qpix];
            if d_perp <= strip {
                if !is_empty(q_center) {
                    h_center += 1.0;
                } else if !is_empty(q_cons) {
                    h_cons += 1.0;
                }
            }
            if is_empty(q_center) {
                continue;
            }
            let q_unpacked = unpack_gb(q_center);
            let q_norm = f32(q_unpacked.depth) / 16777214.0;
            let p_norm = f32(depth_q) / 16777214.0;
            if abs(q_norm - p_norm) > depth_reject {
                continue;
            }
            let spatial = exp(-d_par * d_par / (sigma_par * sigma_par) - d_perp * d_perp / (sigma_perp * sigma_perp));
            var color_w = 1.0;
            let q_color = shaded[qpix].xyz;
            if !promote && center_color.a > 0.0 {
                let chroma = length(center_color.xyz - q_color);
                color_w = exp(-chroma * chroma / (sigma_c * sigma_c));
            }
            let w = spatial * color_w;
            accum += q_color * w;
            weight += w;
        }
    }

    var color = center_color.xyz;
    if weight > 1e-5 {
        color = accum / weight;
    }
    var alpha = 1.0;
    let coverage = h_center / max(h_center + h_cons, 1.0);
    // Section 3.3.3 counts conservative hits, including pixels that also have a center hit.
    // The loop above only increments h_cons on conservative-only pixels, so the ratio is
    // H_center / (H_center + H_cons_only) == H_center / H_cons.
    let cons_depth = select(1.0, f32(unpack_gb(cons_word).depth) / 16777214.0, cons_hit);
    let center_depth = f32(depth_q) / 16777214.0;
    let closer = cons_hit && center_hit && (center_depth - cons_depth) >= closer_eps;
    if closer {
        alpha = 1.0;
    } else {
        alpha = clamp(coverage, 0.0, 1.0);
    }
    if promote && weight <= 1e-5 {
        alpha = 0.0;
        color = vec3<f32>(0.0);
    }
    filtered[pix] = vec4<f32>(color, alpha);
    filtered_depth[pix] = view_dist;
}
