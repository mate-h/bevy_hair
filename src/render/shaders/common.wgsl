const LAYER_COUNT: u32 = 16u;
const STYLE_U: u32 = 8u;
const STYLE_V: u32 = 8u;
const STYLE_W: u32 = 16u;
const PI: f32 = 3.14159265;

struct HairParams {
    clip_from_world: mat4x4<f32>,
    world_from_clip: mat4x4<f32>,
    world_from_model: mat4x4<f32>,
    light_clip_from_world: mat4x4<f32>,
    camera_pos: vec4<f32>,
    camera_forward: vec4<f32>,
    screen: vec4<f32>,
    appearance: vec4<f32>,
    filter_params: vec4<f32>,
    albedo: vec4<f32>,
    flags: vec4<u32>,
    pass_mode: vec4<u32>,
    lights: array<vec4<f32>, 3>,
    light_colors: array<vec4<f32>, 3>,
    sh: array<vec4<f32>, 9>,
    light_forward: vec4<f32>,
    dom_info: vec4<f32>,
    env_mip0: vec4<u32>,
    env_mip1: vec4<u32>,
    env_mip2: vec4<u32>,
    env_mip3: vec4<u32>,
}

struct BundleInfo {
    layer_offset: u32,
    style_offset: u32,
    strand_count: u32,
    _pad: u32,
}

struct LayerVert {
    pos_ao: vec4<f32>,
    tangent: vec4<f32>,
}

struct LodRecord {
    n_lod: u32,
    control_points: u32,
    strand_count: u32,
    _pad: u32,
}

struct StrandRef {
    bundle: u32,
    strand: u32,
    n_lod: u32,
    control_points: u32,
}

fn pcg(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn pcg_uv(seed: u32) -> vec2<f32> {
    let a = pcg(seed);
    let b = pcg(a ^ 0x9e3779b9u);
    return vec2<f32>(f32(a), f32(b)) * (1.0 / 4294967296.0);
}

fn pack_gb(depth: u32, tx: u32, ty: u32, uvw: vec3<u32>, ao: u32) -> u64 {
    let high = u64((depth << 8u) | (tx & 0xffu));
    let low = u64(((ty & 0xffu) << 24u)
        | ((uvw.x & 0x3fu) << 18u)
        | ((uvw.y & 0x3fu) << 12u)
        | ((uvw.z & 0x3fu) << 6u)
        | (ao & 0x3fu));
    return (high << 32u) | low;
}

fn gb_low(word: u64) -> u32 {
    return u32(word & 0xfffffffflu);
}

fn gb_high(word: u64) -> u32 {
    return u32(word >> 32u);
}

struct Unpacked {
    depth: u32,
    tx: u32,
    ty: u32,
    uvw: vec3<u32>,
    ao: u32,
}

fn unpack_gb(word: u64) -> Unpacked {
    let low = gb_low(word);
    let high = gb_high(word);
    return Unpacked(
        high >> 8u,
        high & 0xffu,
        low >> 24u,
        vec3<u32>((low >> 18u) & 0x3fu, (low >> 12u) & 0x3fu, (low >> 6u) & 0x3fu),
        low & 0x3fu,
    );
}

fn is_empty(word: u64) -> bool {
    return (gb_high(word) >> 8u) == 0x00ffffffu;
}

fn quantize_depth(dist: f32, near: f32, far: f32) -> u32 {
    let q = clamp((dist - near) / max(far - near, 1e-3), 0.0, 1.0);
    return min(u32(q * 16777214.0), 0x00fffffeu);
}

fn dequantize_depth(depth: u32, near: f32, far: f32) -> f32 {
    return mix(near, far, f32(depth) / 16777214.0);
}

fn oct_encode(n_in: vec3<f32>) -> vec2<u32> {
    let n = normalize(n_in);
    let denom = abs(n.x) + abs(n.y) + abs(n.z);
    var p = n.xy / max(denom, 1e-6);
    if n.z < 0.0 {
        let old = p;
        p.x = (1.0 - abs(old.y)) * select(-1.0, 1.0, old.x >= 0.0);
        p.y = (1.0 - abs(old.x)) * select(-1.0, 1.0, old.y >= 0.0);
    }
    let e = clamp(p * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(1.0));
    return vec2<u32>(u32(e.x * 255.0 + 0.5), u32(e.y * 255.0 + 0.5));
}

fn oct_decode(tx: u32, ty: u32) -> vec3<f32> {
    let e = vec2<f32>(f32(tx), f32(ty)) / 255.0;
    var f = e * 2.0 - 1.0;
    var n = vec3<f32>(f.xy, 1.0 - abs(f.x) - abs(f.y));
    if n.z < 0.0 {
        let old = n.xy;
        n.x = (1.0 - abs(old.y)) * select(-1.0, 1.0, old.x >= 0.0);
        n.y = (1.0 - abs(old.x)) * select(-1.0, 1.0, old.y >= 0.0);
    }
    return normalize(n);
}

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len2 = dot(v, v);
    if len2 < 1e-10 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return v * inverseSqrt(len2);
}

fn calc_tangent(idx: u32, count: u32, prev: vec3<f32>, curr: vec3<f32>, next: vec3<f32>) -> vec3<f32> {
    var dir: vec3<f32>;
    var scale: f32;
    if idx == 0u {
        dir = safe_normalize(next - curr);
        scale = length(next - curr);
    } else if idx + 1u >= count {
        dir = safe_normalize(curr - prev);
        scale = length(curr - prev);
    } else {
        dir = safe_normalize(next - prev);
        scale = length(next - prev) * 0.5;
    }
    return dir * max(scale, 1e-4);
}

fn hermite(p0: vec3<f32>, p1: vec3<f32>, m0: vec3<f32>, m1: vec3<f32>, t: f32) -> vec3<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
}

fn bilinear4(c00: vec4<f32>, c10: vec4<f32>, c01: vec4<f32>, c11: vec4<f32>, uv: vec2<f32>) -> vec4<f32> {
    let u = uv.x;
    let v = uv.y;
    return c00 * ((1.0 - u) * (1.0 - v)) + c10 * (u * (1.0 - v)) + c01 * ((1.0 - u) * v) + c11 * (u * v);
}

fn quant_uvw(uvw: vec3<f32>) -> vec3<u32> {
    let c = clamp(uvw, vec3<f32>(0.0), vec3<f32>(1.0));
    return clamp(
        vec3<u32>(u32(c.x * 63.0 + 0.5), u32(c.y * 63.0 + 0.5), u32(c.z * 63.0 + 0.5)),
        vec3<u32>(0u),
        vec3<u32>(63u),
    );
}

fn dequant_uvw(uvw: vec3<u32>) -> vec3<f32> {
    return vec3<f32>(f32(uvw.x), f32(uvw.y), f32(uvw.z)) / 63.0;
}

fn to_world(params: HairParams, p: vec3<f32>) -> vec3<f32> {
    return (params.world_from_model * vec4<f32>(p, 1.0)).xyz;
}

fn dir_world(params: HairParams, d: vec3<f32>) -> vec3<f32> {
    return safe_normalize((params.world_from_model * vec4<f32>(d, 0.0)).xyz);
}

struct Projected {
    screen: vec2<f32>,
    view_dist: f32,
    clip_depth: f32,
    in_front: bool,
}

fn project_world(params: HairParams, world: vec3<f32>) -> Projected {
    let clip = params.clip_from_world * vec4<f32>(world, 1.0);
    let ndc = clip.xyz / max(clip.w, 1e-5);
    let sx = (ndc.x * 0.5 + 0.5) * params.screen.x;
    let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * params.screen.y;
    let dist = dot(world - params.camera_pos.xyz, params.camera_forward.xyz);
    return Projected(vec2<f32>(sx, sy), dist, clip.z / max(clip.w, 1e-5), clip.w > 0.0 && dist > params.screen.z * 0.5);
}

fn reconstruct_world(params: HairParams, pixel: vec2<i32>, view_dist: f32) -> vec3<f32> {
    let uv = (vec2<f32>(pixel) + 0.5) / max(params.screen.xy, vec2<f32>(1.0));
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let far_h = params.world_from_clip * vec4<f32>(ndc, 0.0, 1.0);
    let far_p = far_h.xyz / max(far_h.w, 1e-5);
    let dir = far_p - params.camera_pos.xyz;
    let denom = max(dot(dir, params.camera_forward.xyz), 1e-4);
    return params.camera_pos.xyz + dir * (view_dist / denom);
}

fn control_points_for_lod(l: f32) -> u32 {
    let c_raw = max(floor(sqrt(clamp(l, 0.0, 1.0)) * 127.0), f32(LAYER_COUNT));
    if c_raw < 3.0 {
        return u32(max(c_raw, 2.0));
    }
    let snapped = pow(2.0, floor(log2(c_raw - 1.0))) + 1.0;
    return u32(min(snapped, 127.0));
}

fn style_index(offset: u32, u: u32, v: u32, w: u32) -> u32 {
    return offset + (w * STYLE_V + v) * STYLE_U + u;
}
