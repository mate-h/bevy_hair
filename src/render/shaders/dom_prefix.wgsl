@group(0) @binding(0) var<uniform> params: HairParams;
@group(0) @binding(1) var<storage, read_write> dom: array<u32>;

// Runs after the opacity raster. Replaces each texel's slice counts with the
// optical depth from the front of the texel through the end of that slice, as
// f32 bits, so a lookup reads two words instead of looping over every slice.
// `dom_optical_depth` in `shade.wesl` and `hair_shadow.wesl` read this layout.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixels = u32(params.dom_info.x) * u32(params.dom_info.y);
    let pix = id.x;
    if pix >= pixels {
        return;
    }
    let front_q = dom[pix];
    if front_q == DOM_EMPTY {
        return;
    }
    let near = params.dom_info.z;
    let far = params.dom_info.w;
    let z_front = dequantize_depth(front_q, near, far);
    let thickness = max(far - near, 1e-3) / f32(DOM_LAYERS);
    let extinction = params.filter_params.w;
    let perspective = params.light_forward.w > 0.5;
    // Slice `s` of texel `pix` is word `pixels * (s + 1) + pix`.
    var occ = 0.0;
    for (var slice = 0u; slice < DOM_LAYERS; slice++) {
        let z = z_front + (f32(slice) + 0.5) * thickness;
        let e = select(extinction, extinction / max(z, 1e-3), perspective);
        let i = pixels * (slice + 1u) + pix;
        occ += f32(dom[i]) * e;
        dom[i] = bitcast<u32>(occ);
    }
}
