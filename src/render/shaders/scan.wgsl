@group(0) @binding(0) var<uniform> params: HairParams;
@group(0) @binding(1) var<storage, read> lod_out: array<LodRecord>;
@group(0) @binding(2) var<storage, read_write> refs: array<StrandRef>;
@group(0) @binding(3) var<storage, read_write> indirect: array<u32>;

const SCAN_LANES: u32 = 256u;

// Inclusive prefix of `n_lod` over the bundles of the current chunk.
var<workgroup> prefix: array<u32, SCAN_LANES>;
var<workgroup> chunk_total: u32;

// One workgroup walks the bundles in chunks of `SCAN_LANES`: a shared-memory
// scan gives each bundle its offset, then the lanes write the chunk's refs
// together, each finding its bundle by binary search. Refs come out in the
// same bundle-major order as a serial loop, so raster sees the same list.
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lane: u32) {
    let bundle_count = params.pass_mode.y;
    let capacity = arrayLength(&refs);
    var base = 0u;
    for (var chunk = 0u; chunk < bundle_count; chunk += SCAN_LANES) {
        let bundle = chunk + lane;
        var n = 0u;
        if bundle < bundle_count {
            n = lod_out[bundle].n_lod;
        }
        prefix[lane] = n;
        workgroupBarrier();
        for (var d = 1u; d < SCAN_LANES; d <<= 1u) {
            var v = prefix[lane];
            if lane >= d {
                v += prefix[lane - d];
            }
            workgroupBarrier();
            prefix[lane] = v;
            workgroupBarrier();
        }
        if lane == SCAN_LANES - 1u {
            chunk_total = prefix[lane];
        }
        let total = workgroupUniformLoad(&chunk_total);

        for (var i = lane; i < total; i += SCAN_LANES) {
            // First bundle in the chunk whose inclusive prefix passes i.
            var lo = 0u;
            var hi = SCAN_LANES - 1u;
            while lo < hi {
                let mid = (lo + hi) / 2u;
                if prefix[mid] > i {
                    hi = mid;
                } else {
                    lo = mid + 1u;
                }
            }
            let start = select(0u, prefix[max(lo, 1u) - 1u], lo > 0u);
            let out = base + i;
            if out < capacity {
                let b = chunk + lo;
                let rec = lod_out[b];
                refs[out] = StrandRef(b, i - start, rec.n_lod, rec.control_points);
            }
        }
        base += total;
        // `prefix` is rewritten by the next chunk.
        workgroupBarrier();
    }
    if lane == 0u {
        indirect[0] = min(base, capacity);
        indirect[1] = 1u;
        indirect[2] = 1u;
    }
}
