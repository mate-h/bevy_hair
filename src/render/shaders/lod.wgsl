@group(0) @binding(0) var<uniform> params: HairParams;
@group(0) @binding(1) var<storage, read> bundles: array<BundleInfo>;
@group(0) @binding(2) var<storage, read> layers: array<LayerVert>;
@group(0) @binding(3) var<storage, read_write> lod_out: array<LodRecord>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let bundle_id = id.x;
    if bundle_id >= params.pass_mode.y {
        return;
    }
    let bundle = bundles[bundle_id];
    var aabb_min = vec2<f32>(1e20, 1e20);
    var aabb_max = vec2<f32>(-1e20, -1e20);
    var any_front = false;
    var any_point = false;
    for (var layer = 0u; layer < LAYER_COUNT; layer++) {
        for (var corner = 0u; corner < 4u; corner++) {
            let vert = layers[bundle.layer_offset + layer * 4u + corner];
            let world = to_world(params, vert.pos_ao.xyz);
            let clip = params.clip_from_world * vec4<f32>(world, 1.0);
            let dist = dot(world - params.camera_pos.xyz, params.camera_forward.xyz);
            if clip.w > 0.0 && dist > 0.0 {
                any_front = true;
            }
            let projected = project_world(params, world);
            aabb_min = min(aabb_min, projected.screen);
            aabb_max = max(aabb_max, projected.screen);
            any_point = true;
        }
    }

    let culled = !any_point || !any_front || aabb_max.x < 0.0 || aabb_max.y < 0.0
        || aabb_min.x > params.screen.x || aabb_min.y > params.screen.y;

    var n_lod: u32 = 0u;
    var cps: u32 = LAYER_COUNT;
    if !culled {
        // pass_mode.w == 1: deep opacity map. Full strand count, layer-resolution polylines.
        if params.pass_mode.w == 1u {
            n_lod = bundle.strand_count;
            cps = LAYER_COUNT;
        } else if params.flags.x == 0u {
            n_lod = bundle.strand_count;
            cps = 127u;
        } else {
            let diag = length(aabb_max - aabb_min);
            let l = clamp(diag / max(params.screen.y, 1.0) * params.appearance.z, 0.0, 1.0);
            let delta = f32(pcg(bundle_id ^ 0x9e3779b9u)) * (1.0 / 4294967296.0);
            let n = f32(bundle.strand_count);
            let raw = ceil(l * (n + delta));
            n_lod = u32(clamp(raw, 1.0, n));
            cps = control_points_for_lod(l);
        }
    }

    lod_out[bundle_id] = LodRecord(n_lod, cps, bundle.strand_count, 0u);
}
