@group(0) @binding(0) var<storage, read_write> center: array<u64>;
@group(0) @binding(1) var<storage, read_write> conservative: array<u64>;
@group(0) @binding(2) var<storage, read_write> beta: array<u32>;

const EMPTY_U64: u64 = 18446744073709551615lu;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&center) {
        return;
    }
    center[i] = EMPTY_U64;
    conservative[i] = EMPTY_U64;
    beta[i] = 0xffffffffu;
}
