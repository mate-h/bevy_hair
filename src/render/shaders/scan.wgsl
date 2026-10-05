@group(0) @binding(0) var<uniform> params: HairParams;
@group(0) @binding(1) var<storage, read> lod_out: array<LodRecord>;
@group(0) @binding(2) var<storage, read_write> refs: array<StrandRef>;
@group(0) @binding(3) var<storage, read_write> indirect: array<u32>;

@compute @workgroup_size(1)
fn main() {
    var total = 0u;
    let bundle_count = params.pass_mode.y;
    let capacity = arrayLength(&refs);
    for (var bundle = 0u; bundle < bundle_count; bundle++) {
        let rec = lod_out[bundle];
        for (var strand = 0u; strand < rec.n_lod; strand++) {
            if total >= capacity {
                break;
            }
            refs[total] = StrandRef(bundle, strand, rec.n_lod, rec.control_points);
            total += 1u;
        }
    }
    indirect[0] = total;
    indirect[1] = 1u;
    indirect[2] = 1u;
}
