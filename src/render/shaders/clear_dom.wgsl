// Keep in sync with `DOM_SIZE` and `DOM_LAYERS` in pass.rs.
const DOM_SIZE: u32 = 512u;
const DOM_LAYERS: u32 = 16u;
const DOM_PIXELS: u32 = DOM_SIZE * DOM_SIZE;

@group(0) @binding(0) var<storage, read_write> dom: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&dom) {
        return;
    }
    // Texels first, then one opacity count per layer. Empty depth is the
    // atomicMin identity so an unwritten texel stays "no hair".
    if i < DOM_PIXELS {
        dom[i] = 0xffffffffu;
    } else {
        dom[i] = 0u;
    }
}
