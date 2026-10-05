@group(0) @binding(0) var<storage, read_write> dom: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&dom) {
        return;
    }
    dom[i] = 0u;
}
