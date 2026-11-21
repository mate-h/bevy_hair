@group(0) @binding(0) var<uniform> params: HairParams;
@group(0) @binding(1) var<storage, read> color_buf: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> depth_buf: array<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
}

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
    var x = -1.0;
    var y = -1.0;
    if index == 1u {
        x = 3.0;
    }
    if index == 2u {
        y = 3.0;
    }
    return VsOut(vec4<f32>(x, y, 0.0, 1.0));
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

struct Sample {
    color: vec4<f32>,
    world: vec3<f32>,
    depth: f32,
}

fn load_sample(frag: vec4<f32>) -> Sample {
    let width = i32(params.screen.x);
    let height = i32(params.screen.y);
    let p = vec2<i32>(i32(frag.x), i32(frag.y));
    if p.x < 0 || p.y < 0 || p.x >= width || p.y >= height {
        discard;
    }
    let pix = u32(p.y) * u32(width) + u32(p.x);
    let color = color_buf[pix];
    if color.a < 0.001 {
        discard;
    }
    let view_dist = depth_buf[pix];
    let world = reconstruct_world(params, p, view_dist);
    let clip = params.clip_from_world * vec4<f32>(world, 1.0);
    let depth = clamp(clip.z / max(clip.w, 1e-5), 0.0, 1.0);
    return Sample(color, world, depth);
}

@fragment
fn fs(in: VsOut) -> FsOut {
    let s = load_sample(in.clip);
    return FsOut(s.color, s.depth);
}

struct FsMotionOut {
    @location(0) color: vec4<f32>,
    @location(1) motion: vec2<f32>,
    @builtin(frag_depth) depth: f32,
}

// Also writes the prepass motion vector, in Bevy's convention: the UV offset
// from last frame's position, from unjittered clip positions.
@fragment
fn fs_motion(in: VsOut) -> FsMotionOut {
    let s = load_sample(in.clip);
    let now = params.motion_clip * vec4<f32>(s.world, 1.0);
    let before = params.prev_clip * vec4<f32>(s.world, 1.0);
    let delta = now.xy / max(now.w, 1e-5) - before.xy / max(before.w, 1e-5);
    return FsMotionOut(s.color, delta * vec2<f32>(0.5, -0.5), s.depth);
}
