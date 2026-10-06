@group(0) @binding(0) var<uniform> params: HairParams;
@group(0) @binding(1) var<storage, read> bundles: array<BundleInfo>;
@group(0) @binding(2) var<storage, read> layers: array<LayerVert>;
@group(0) @binding(3) var<storage, read> style: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> refs: array<StrandRef>;
@group(0) @binding(5) var<storage, read_write> center: array<atomic<u64>>;
@group(0) @binding(6) var<storage, read_write> conservative: array<atomic<u64>>;
@group(0) @binding(7) var<storage, read_write> beta_buf: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> dom: array<atomic<u32>>;
@group(0) @binding(9) var scene_depth: texture_depth_2d;

struct SmLayer {
    pos_ao: vec4<f32>,
    tangent: vec4<f32>,
}

var<workgroup> sm: array<SmLayer, 16>;
var<workgroup> cp_pos: array<vec4<f32>, 128>;
var<workgroup> cp_tan: array<vec4<f32>, 128>;
var<workgroup> cp_uvw: array<vec4<f32>, 128>;

fn sample_layer(offset: u32, layer: u32, uv: vec2<f32>) -> SmLayer {
    let i = offset + layer * 4u;
    let c00 = layers[i];
    let c10 = layers[i + 1u];
    let c01 = layers[i + 2u];
    let c11 = layers[i + 3u];
    return SmLayer(
        bilinear4(c00.pos_ao, c10.pos_ao, c01.pos_ao, c11.pos_ao, uv),
        bilinear4(c00.tangent, c10.tangent, c01.tangent, c11.tangent, uv),
    );
}

fn sample_style(offset: u32, uv: vec2<f32>, w: f32) -> vec3<f32> {
    let u = clamp(uv.x, 0.0, 1.0) * f32(STYLE_U - 1u);
    let v = clamp(uv.y, 0.0, 1.0) * f32(STYLE_V - 1u);
    let ww = clamp(w, 0.0, 1.0) * f32(STYLE_W - 1u);
    let u0 = u32(floor(u));
    let v0 = u32(floor(v));
    let w0 = u32(floor(ww));
    let u1 = min(u0 + 1u, STYLE_U - 1u);
    let v1 = min(v0 + 1u, STYLE_V - 1u);
    let w1 = min(w0 + 1u, STYLE_W - 1u);
    let tu = u - f32(u0);
    let tv = v - f32(v0);
    let tw = ww - f32(w0);
    let c000 = style[style_index(offset, u0, v0, w0)].xyz;
    let c100 = style[style_index(offset, u1, v0, w0)].xyz;
    let c010 = style[style_index(offset, u0, v1, w0)].xyz;
    let c110 = style[style_index(offset, u1, v1, w0)].xyz;
    let c001 = style[style_index(offset, u0, v0, w1)].xyz;
    let c101 = style[style_index(offset, u1, v0, w1)].xyz;
    let c011 = style[style_index(offset, u0, v1, w1)].xyz;
    let c111 = style[style_index(offset, u1, v1, w1)].xyz;
    let c00 = mix(c000, c100, tu);
    let c01 = mix(c010, c110, tu);
    let c10 = mix(c001, c101, tu);
    let c11 = mix(c011, c111, tu);
    return mix(mix(c00, c01, tv), mix(c10, c11, tv), tw);
}

fn eval_cp(idx: u32, ncp: u32, uv: vec2<f32>, style_offset: u32) -> vec4<f32> {
    let w = f32(idx) / f32(max(ncp - 1u, 1u));
    let layers_f = w * f32(LAYER_COUNT - 1u);
    let i0 = min(u32(floor(layers_f)), LAYER_COUNT - 1u);
    let i1 = min(i0 + 1u, LAYER_COUNT - 1u);
    let t = layers_f - f32(i0);
    let a = sm[i0];
    let b = sm[i1];
    let base = hermite(a.pos_ao.xyz, b.pos_ao.xyz, a.tangent.xyz, b.tangent.xyz, t);
    let ao = mix(a.pos_ao.w, b.pos_ao.w, t);
    let residual = sample_style(style_offset, uv, w);
    cp_uvw[idx] = vec4<f32>(uv, w, ao);
    return vec4<f32>(base + residual, ao);
}

fn write_pixel(pixel: vec2<i32>, t: f32, p0: vec3<f32>, p1: vec3<f32>, t0: vec3<f32>, t1: vec3<f32>, uvw0: vec3<f32>, uvw1: vec3<f32>, ao0: f32, ao1: f32, center_hit: bool, beta: f32) {
    let width = i32(params.screen.x);
    let height = i32(params.screen.y);
    if pixel.x < 0 || pixel.y < 0 || pixel.x >= width || pixel.y >= height {
        return;
    }
    let world = mix(p0, p1, t);
    let tangent = safe_normalize(mix(t0, t1, t));
    let uvw = mix(uvw0, uvw1, t);
    let ao = mix(ao0, ao1, t);
    let dist = dot(world - params.camera_pos.xyz, params.camera_forward.xyz);
    if dist <= params.screen.z {
        return;
    }
    let clip = params.clip_from_world * vec4<f32>(world, 1.0);
    let clip_depth = clip.z / max(clip.w, 1e-5);
    let scene = textureLoad(scene_depth, pixel, 0);
    if params.pass_mode.z == 1u {
        if clip_depth < scene {
            return;
        }
    } else if clip_depth > scene {
        return;
    }

    let depth = quantize_depth(dist, params.screen.z, params.screen.w);
    let oct = oct_encode(tangent);
    var ao_q = u32(clamp(ao, 0.0, 1.0) * 63.0 + 0.5);
    if params.flags.z == 0u {
        ao_q = 63u;
    }
    let payload = pack_gb(depth, oct.x, oct.y, quant_uvw(uvw), ao_q);
    let pix = u32(pixel.y) * u32(width) + u32(pixel.x);
    atomicMin(&conservative[pix], payload);
    if center_hit {
        atomicMin(&center[pix], payload);
        let beta_q = u32(clamp(beta, 0.0, 1.0) * 255.0 + 0.5);
        atomicMin(&beta_buf[pix], (depth << 8u) | beta_q);
    }
}

fn write_dom_pixel(pixel: vec2<i32>, world: vec3<f32>) {
    let width = i32(params.screen.x);
    let height = i32(params.screen.y);
    if pixel.x < 0 || pixel.y < 0 || pixel.x >= width || pixel.y >= height {
        return;
    }
    let near = params.screen.z;
    let far = params.screen.w;
    let dist = dot(world - params.camera_pos.xyz, params.camera_forward.xyz);
    if dist <= near || dist >= far {
        return;
    }
    let dom_w = u32(params.dom_info.x);
    let pix = u32(pixel.y) * dom_w + u32(pixel.x);
    let q = quantize_depth(dist, near, far);
    // Pass 1 stores the nearest depth. Pass 2 bins opacity behind that depth.
    if params.pass_mode.x == 1u {
        atomicMin(&dom[pix], q);
        return;
    }
    let front_q = atomicLoad(&dom[pix]);
    if front_q == DOM_EMPTY {
        return;
    }
    let z_front = dequantize_depth(front_q, near, far);
    let thickness = max(far - near, 1e-3) / f32(DOM_LAYERS);
    let t = max(dist - z_front, 0.0) / thickness;
    let slice = min(u32(t), DOM_LAYERS - 1u);
    let pixels = dom_w * u32(params.dom_info.y);
    atomicAdd(&dom[pixels + pix * DOM_LAYERS + slice], 1u);
}

fn draw_dom_segment(p0: vec3<f32>, p1: vec3<f32>, include_start: bool) {
    let a = project_world(params, p0);
    let b = project_world(params, p1);
    if !a.in_front && !b.in_front {
        return;
    }
    var x0 = i32(floor(a.screen.x));
    var y0 = i32(floor(a.screen.y));
    let x1 = i32(floor(b.screen.x));
    let y1 = i32(floor(b.screen.y));
    let dx = abs(x1 - x0);
    let dy = abs(y1 - y0);
    let sx = select(-1, 1, x0 < x1);
    let sy = select(-1, 1, y0 < y1);
    var err = dx - dy;
    let seg = b.screen - a.screen;
    let seg_len2 = max(dot(seg, seg), 1e-6);
    // The next segment owns a shared screen-space endpoint. A segment that
    // stays in one pixel is moving along the light and has to bin its own depth.
    var skip = !include_start && (x0 != x1 || y0 != y1);
    for (var step = 0; step < 1024; step++) {
        if !skip {
            let center = vec2<f32>(f32(x0), f32(y0)) + 0.5;
            let t = clamp(dot(center - a.screen, seg) / seg_len2, 0.0, 1.0);
            write_dom_pixel(vec2<i32>(x0, y0), mix(p0, p1, t));
        }
        skip = false;
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = err + err;
        if e2 > -dy {
            err -= dy;
            x0 += sx;
        }
        if e2 < dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn draw_segment(p0: vec3<f32>, p1: vec3<f32>, t0: vec3<f32>, t1: vec3<f32>, uvw0: vec3<f32>, uvw1: vec3<f32>, ao0: f32, ao1: f32, beta: f32) {
    let a = project_world(params, p0);
    let b = project_world(params, p1);
    if !a.in_front && !b.in_front {
        return;
    }
    let delta = b.screen - a.screen;
    let steps = i32(ceil(max(abs(delta.x), abs(delta.y))));
    let n = clamp(steps, 1, 96);
    let diameter = max(params.appearance.w, 0.05);
    var prev = a.screen;
    for (var i = 0; i <= n; i++) {
        let t = f32(i) / f32(n);
        let screen = mix(a.screen, b.screen, t);
        let pix = vec2<i32>(floor(screen));
        let center = vec2<f32>(pix) + 0.5;
        let seg = b.screen - a.screen;
        let denom = max(dot(seg, seg), 1e-6);
        let closest = clamp(dot(center - a.screen, seg) / denom, 0.0, 1.0);
        let dist = length(center - (a.screen + seg * closest));
        let center_hit = dist <= diameter * 0.5;
        write_pixel(pix, t, p0, p1, t0, t1, uvw0, uvw1, ao0, ao1, center_hit, beta);
        if i > 0 {
            let prev_pix = vec2<i32>(floor(prev));
            if prev_pix.x != pix.x && prev_pix.y != pix.y {
                write_pixel(vec2<i32>(prev_pix.x, pix.y), t, p0, p1, t0, t1, uvw0, uvw1, ao0, ao1, false, beta);
                write_pixel(vec2<i32>(pix.x, prev_pix.y), t, p0, p1, t0, t1, uvw0, uvw1, ao0, ao1, false, beta);
            }
        }
        prev = screen;
    }
}

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(subgroup_size) subgroup_size: u32,
) {
    let strand = refs[wg.x];
    let bundle = bundles[strand.bundle];
    let uv = pcg_uv(strand.strand ^ (strand.bundle * 0x85ebca6bu));
    let use_shuffle = subgroup_size == 32u;

    var loaded = vec4<f32>(0.0);
    if lane < LAYER_COUNT {
        let sample = sample_layer(bundle.layer_offset, lane, uv);
        loaded = sample.pos_ao;
        sm[lane] = sample;
    }
    var prev_l = loaded;
    var next_l = loaded;
    if use_shuffle {
        prev_l = subgroupShuffleUp(loaded, 1u);
        next_l = subgroupShuffleDown(loaded, 1u);
    }
    workgroupBarrier();
    if lane < LAYER_COUNT {
        var prev_p = prev_l.xyz;
        var next_p = next_l.xyz;
        if !use_shuffle {
            let prev_i = select(lane - 1u, 0u, lane == 0u);
            let next_i = min(lane + 1u, LAYER_COUNT - 1u);
            prev_p = sm[prev_i].pos_ao.xyz;
            next_p = sm[next_i].pos_ao.xyz;
        }
        if lane == 0u {
            prev_p = loaded.xyz;
        }
        if lane + 1u >= LAYER_COUNT {
            next_p = loaded.xyz;
        }
        let tangent = calc_tangent(lane, LAYER_COUNT, prev_p, loaded.xyz, next_p);
        sm[lane].pos_ao = loaded;
        sm[lane].tangent = vec4<f32>(tangent, 0.0);
    }
    workgroupBarrier();

    let ncp = clamp(strand.control_points, 2u, 127u);
    // Points per lane. ceil((ncp - 1) / 32) covers segments only, and LOD
    // counts are 2^k+1, so the tip stayed zero and the last segment hit the origin.
    let s = max((ncp + 31u) / 32u, 1u);
    let first = lane * s;
    for (var k = 0u; k < 4u; k++) {
        if k >= s {
            break;
        }
        let idx = first + k;
        if idx < ncp {
            cp_pos[idx] = eval_cp(idx, ncp, uv, bundle.style_offset);
        }
    }
    workgroupBarrier();

    let owned_first = cp_pos[min(first, ncp - 1u)];
    let owned_last = cp_pos[min(first + s - 1u, ncp - 1u)];
    var prev_last = owned_first;
    var next_first = owned_last;
    if use_shuffle {
        prev_last = subgroupShuffleUp(owned_last, 1u);
        next_first = subgroupShuffleDown(owned_first, 1u);
    }
    for (var k = 0u; k < 4u; k++) {
        if k >= s {
            break;
        }
        let idx = first + k;
        if idx < ncp {
            var prev_p = cp_pos[idx].xyz;
            var next_p = cp_pos[idx].xyz;
            if idx > 0u {
                prev_p = cp_pos[idx - 1u].xyz;
            }
            if idx + 1u < ncp {
                next_p = cp_pos[idx + 1u].xyz;
            }
            if use_shuffle && idx == first && lane > 0u && idx > 0u {
                prev_p = prev_last.xyz;
            }
            if use_shuffle && k + 1u == s && idx + 1u < ncp {
                next_p = next_first.xyz;
            }
            cp_tan[idx] = vec4<f32>(safe_normalize(calc_tangent(idx, ncp, prev_p, cp_pos[idx].xyz, next_p)), 0.0);
        }
    }
    workgroupBarrier();

    let beta = f32(strand.n_lod) / f32(max(bundle.strand_count, 1u));
    for (var k = 0u; k < 4u; k++) {
        if k >= s {
            break;
        }
        let idx = first + k;
        if idx + 1u < ncp {
            let p0 = to_world(params, cp_pos[idx].xyz);
            let p1 = to_world(params, cp_pos[idx + 1u].xyz);
            if params.pass_mode.x != 0u {
                draw_dom_segment(p0, p1, idx == 0u);
            } else {
                let t0 = dir_world(params, cp_tan[idx].xyz);
                let t1 = dir_world(params, cp_tan[idx + 1u].xyz);
                draw_segment(p0, p1, t0, t1, cp_uvw[idx].xyz, cp_uvw[idx + 1u].xyz, cp_uvw[idx].w, cp_uvw[idx + 1u].w, beta);
            }
        }
    }
}
