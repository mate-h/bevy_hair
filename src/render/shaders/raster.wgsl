@group(0) @binding(0) var<uniform> params: HairParams;
@group(0) @binding(1) var<storage, read> bundles: array<BundleInfo>;
@group(0) @binding(2) var<storage, read> layers: array<LayerVert>;
// Three f16 per texel, `[x | y << 16, z]`. See `pack_residual` in mesh.rs.
@group(0) @binding(3) var<storage, read> style: array<vec2<u32>>;
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
// Workgroup memory bounds how many workgroups share a core, so this holds
// only xyz and ao per control point. Tangents come from the neighbors
// (`cp_tangent`) and uvw from the strand's uv and the point's w (`cp_w`).
var<workgroup> cp_pos: array<vec4<f32>, MAX_CONTROL_POINTS>;

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

fn style_texel(i: u32) -> vec3<f32> {
    let t = style[i];
    return vec3<f32>(unpack2x16float(t.x), unpack2x16float(t.y).x);
}

// The nearest column of the bundle (see `style_cell` in mesh.rs), at two
// neighbouring w samples of `level`. With 2^k + 1 control points on the
// matching level, w lands on a sample and the second read is skipped.
fn sample_style(offset: u32, uv: vec2<f32>, w: f32, level: u32) -> vec3<f32> {
    let w_count = style_level_w(level);
    let cell = min(
        vec2<u32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(f32(STYLE_U), f32(STYLE_V))),
        vec2<u32>(STYLE_U - 1u, STYLE_V - 1u),
    );
    let ww = clamp(w, 0.0, 1.0) * f32(w_count - 1u);
    let w0 = min(u32(floor(ww)), w_count - 1u);
    let tw = ww - f32(w0);
    let i = style_index(offset, level, cell.x, cell.y, w0);
    var r = style_texel(i);
    if tw > 1e-4 && w0 + 1u < w_count {
        r = mix(r, style_texel(i + 1u), tw);
    }
    return r;
}

fn cp_w(idx: u32, ncp: u32) -> f32 {
    return f32(idx) / f32(max(ncp - 1u, 1u));
}

fn cp_tangent(idx: u32, ncp: u32) -> vec3<f32> {
    let p = cp_pos[idx].xyz;
    var prev_p = p;
    var next_p = p;
    if idx > 0u {
        prev_p = cp_pos[idx - 1u].xyz;
    }
    if idx + 1u < ncp {
        next_p = cp_pos[idx + 1u].xyz;
    }
    return safe_normalize(calc_tangent(idx, ncp, prev_p, p, next_p));
}

fn eval_cp(idx: u32, ncp: u32, uv: vec2<f32>, style_offset: u32) -> vec4<f32> {
    let w = cp_w(idx, ncp);
    let layers_f = w * f32(LAYER_COUNT - 1u);
    let i0 = min(u32(floor(layers_f)), LAYER_COUNT - 1u);
    let i1 = min(i0 + 1u, LAYER_COUNT - 1u);
    let t = layers_f - f32(i0);
    let a = sm[i0];
    let b = sm[i1];
    let base = hermite(a.pos_ao.xyz, b.pos_ao.xyz, a.tangent.xyz, b.tangent.xyz, t);
    let ao = mix(a.pos_ao.w, b.pos_ao.w, t);
    let residual = sample_style(style_offset, uv, w, style_level_for(ncp));
    return vec4<f32>(base + residual, ao);
}

// The G-buffer word for one step of a segment. A step's corner pixels reuse
// the same t, so only the rect check and the scene depth test differ per pixel.
struct GbSample {
    // Complemented, so buffers clear to zero (see `unpack_gb`).
    payload: u64,
    beta_word: u32,
    clip_depth: f32,
    valid: bool,
}

fn encode_sample(t: f32, p0: vec3<f32>, p1: vec3<f32>, t0: vec3<f32>, t1: vec3<f32>, uvw0: vec3<f32>, uvw1: vec3<f32>, ao0: f32, ao1: f32, beta_q: u32) -> GbSample {
    var s: GbSample;
    let world = mix(p0, p1, t);
    let dist = dot(world - params.camera_pos.xyz, params.camera_forward.xyz);
    s.valid = dist > params.screen.z;
    if !s.valid {
        return s;
    }
    let clip = params.clip_from_world * vec4<f32>(world, 1.0);
    s.clip_depth = clip.z / max(clip.w, 1e-5);
    let tangent = safe_normalize(mix(t0, t1, t));
    let uvw = mix(uvw0, uvw1, t);
    let ao = mix(ao0, ao1, t);
    let depth = quantize_depth(dist, params.screen.z, params.screen.w);
    let oct = oct_encode(tangent);
    var ao_q = u32(clamp(ao, 0.0, 1.0) * 63.0 + 0.5);
    if params.flags.z == 0u {
        ao_q = 63u;
    }
    s.payload = ~pack_gb(depth, oct.x, oct.y, quant_uvw(uvw), ao_q);
    s.beta_word = ~((depth << 8u) | beta_q);
    return s;
}

fn write_pixel(pixel: vec2<i32>, s: GbSample, center_hit: bool) {
    // Outside the rectangle the G-buffer is not cleared.
    let rect = vec4<i32>(params.rect);
    if any(pixel < rect.xy) || any(pixel >= rect.zw) {
        return;
    }
    let scene = textureLoad(scene_depth, pixel, 0);
    if params.pass_mode.z == 1u {
        if s.clip_depth < scene {
            return;
        }
    } else if s.clip_depth > scene {
        return;
    }
    let pix = u32(pixel.y) * u32(params.screen.x) + u32(pixel.x);
    atomicMax(&conservative[pix], s.payload);
    if center_hit {
        atomicMax(&center[pix], s.payload);
        atomicMax(&beta_buf[pix], s.beta_word);
    }
}

// `coverage` is the strand's projected length inside the texel, in
// `DOM_COVERAGE` units. Only the opacity pass uses it.
fn write_dom_pixel(pixel: vec2<i32>, world: vec3<f32>, coverage: u32) {
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
    // Slice-major, so neighboring texels of one slice share cache lines.
    atomicAdd(&dom[pixels * (slice + 1u) + pix], coverage);
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
    // A strand of width d covers d * length / w² of a texel of width w, so
    // each texel gets its share of the projected length, not a whole hit.
    // Strands running along the light cover little of the texels they cross.
    let written = max(dx, dy) + select(1, 0, skip);
    let coverage = u32(sqrt(dot(seg, seg)) / f32(written) * DOM_COVERAGE + 0.5);
    for (var step = 0; step < 1024; step++) {
        if !skip {
            let center = vec2<f32>(f32(x0), f32(y0)) + 0.5;
            let t = clamp(dot(center - a.screen, seg) / seg_len2, 0.0, 1.0);
            write_dom_pixel(vec2<i32>(x0, y0), mix(p0, p1, t), coverage);
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
    let seg = b.screen - a.screen;
    let denom = max(dot(seg, seg), 1e-6);
    let beta_q = u32(clamp(beta, 0.0, 1.0) * 255.0 + 0.5);
    var prev = a.screen;
    for (var i = 0; i <= n; i++) {
        let t = f32(i) / f32(n);
        let screen = mix(a.screen, b.screen, t);
        let pix = vec2<i32>(floor(screen));
        let s = encode_sample(t, p0, p1, t0, t1, uvw0, uvw1, ao0, ao1, beta_q);
        if s.valid {
            let center = vec2<f32>(pix) + 0.5;
            let closest = clamp(dot(center - a.screen, seg) / denom, 0.0, 1.0);
            let dist = length(center - (a.screen + seg * closest));
            write_pixel(pix, s, dist <= diameter * 0.5);
            if i > 0 {
                let prev_pix = vec2<i32>(floor(prev));
                if prev_pix.x != pix.x && prev_pix.y != pix.y {
                    write_pixel(vec2<i32>(prev_pix.x, pix.y), s, false);
                    write_pixel(vec2<i32>(pix.x, prev_pix.y), s, false);
                }
            }
        }
        prev = screen;
    }
}

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let strand = refs[wg.x];
    let bundle = bundles[strand.bundle];
    let uv = strand_uv(strand.bundle, strand.strand);

    // Neighbors come from shared memory after each barrier. Subgroup shuffles
    // would assume the subgroup is the workgroup, which WGSL does not promise.
    if lane < LAYER_COUNT {
        sm[lane] = sample_layer(bundle.layer_offset, lane, uv);
    }
    workgroupBarrier();
    if lane < LAYER_COUNT {
        let prev_p = sm[select(lane - 1u, 0u, lane == 0u)].pos_ao.xyz;
        let next_p = sm[min(lane + 1u, LAYER_COUNT - 1u)].pos_ao.xyz;
        let tangent = calc_tangent(lane, LAYER_COUNT, prev_p, sm[lane].pos_ao.xyz, next_p);
        sm[lane].tangent = vec4<f32>(tangent, 0.0);
    }
    workgroupBarrier();

    let ncp = clamp(strand.control_points, 2u, MAX_CONTROL_POINTS);
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

    let beta = f32(strand.n_lod) / f32(max(bundle.strand_count, 1u));
    // Segments go round-robin, so 2^k segments keep every lane busy where the
    // point split above would leave half of them idle. The G-buffer and opacity
    // writes are atomic max/min/add, so their order does not matter.
    for (var k = 0u; k < 2u; k++) {
        let idx = lane + k * 32u;
        if idx + 1u < ncp {
            let p0 = to_world(params, cp_pos[idx].xyz);
            let p1 = to_world(params, cp_pos[idx + 1u].xyz);
            if params.pass_mode.x != 0u {
                draw_dom_segment(p0, p1, idx == 0u);
            } else {
                let t0 = dir_world(params, cp_tangent(idx, ncp));
                let t1 = dir_world(params, cp_tangent(idx + 1u, ncp));
                let uvw0 = vec3<f32>(uv, cp_w(idx, ncp));
                let uvw1 = vec3<f32>(uv, cp_w(idx + 1u, ncp));
                draw_segment(p0, p1, t0, t1, uvw0, uvw1, cp_pos[idx].w, cp_pos[idx + 1u].w, beta);
            }
        }
    }
}
