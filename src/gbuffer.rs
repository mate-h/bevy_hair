//! 64-bit G-buffer packing from Lipp et al. 2026, Section 3.1.2.
//!
//! High bits to low bits:
//! 24 depth, 16 octahedral tangent, 18 styling uvw, 6 ambient occlusion.
//! Depth occupies the top of the word so `atomicMin` keeps the closest strand.
//! Quantized depth is clamped to `0x00FF_FFFE` so the all-ones word stays the empty sentinel.

pub const EMPTY_GB: u64 = u64::MAX;
const DEPTH_MAX: u32 = 0x00FF_FFFE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GBufferSample {
    pub depth: u32,
    pub tangent_oct: [u8; 2],
    pub uvw: [u8; 3],
    pub ao: u8,
}

pub fn pack_gbuffer(sample: GBufferSample) -> u64 {
    let depth = sample.depth.min(DEPTH_MAX) as u64;
    let tx = sample.tangent_oct[0] as u64;
    let ty = sample.tangent_oct[1] as u64;
    let u = (sample.uvw[0] as u64) & 0x3F;
    let v = (sample.uvw[1] as u64) & 0x3F;
    let w = (sample.uvw[2] as u64) & 0x3F;
    let ao = (sample.ao as u64) & 0x3F;
    (depth << 40) | (tx << 32) | (ty << 24) | (u << 18) | (v << 12) | (w << 6) | ao
}

pub fn unpack_gbuffer(word: u64) -> GBufferSample {
    GBufferSample {
        depth: ((word >> 40) & 0xFF_FFFF) as u32,
        tangent_oct: [((word >> 32) & 0xFF) as u8, ((word >> 24) & 0xFF) as u8],
        uvw: [
            ((word >> 18) & 0x3F) as u8,
            ((word >> 12) & 0x3F) as u8,
            ((word >> 6) & 0x3F) as u8,
        ],
        ao: (word & 0x3F) as u8,
    }
}

pub fn is_empty_gbuffer(word: u64) -> bool {
    word == EMPTY_GB
}

/// Side channel so deferred DOM sampling can recover the winning strand's LOD fraction.
/// Depth sits in the high 24 bits, so `atomicMin` selects the same strand as the G-buffer.
pub fn pack_beta_word(depth: u32, beta: f32) -> u32 {
    let depth = depth.min(DEPTH_MAX);
    let q = (beta.clamp(0.0, 1.0) * 255.0).round() as u32;
    (depth << 8) | (q & 0xFF)
}

pub fn unpack_beta_word(word: u32) -> (u32, f32) {
    let depth = word >> 8;
    let beta = (word & 0xFF) as f32 / 255.0;
    (depth, beta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_preserves_fields() {
        let sample = GBufferSample {
            depth: 0x12_34_56,
            tangent_oct: [3, 250],
            uvw: [1, 17, 63],
            ao: 40,
        };
        let word = pack_gbuffer(sample);
        assert!(!is_empty_gbuffer(word));
        assert_eq!(unpack_gbuffer(word), sample);
    }

    #[test]
    fn closer_depth_is_numerically_smaller() {
        let far = pack_gbuffer(GBufferSample {
            depth: 1000,
            tangent_oct: [255, 255],
            uvw: [63, 63, 63],
            ao: 63,
        });
        let near = pack_gbuffer(GBufferSample {
            depth: 999,
            tangent_oct: [0, 0],
            uvw: [0, 0, 0],
            ao: 0,
        });
        assert!(near < far);
        assert!(near < EMPTY_GB);
    }

    #[test]
    fn beta_word_orders_by_depth() {
        let far = pack_beta_word(50, 0.0);
        let near = pack_beta_word(10, 1.0);
        assert!(near < far);
        let (depth, beta) = unpack_beta_word(pack_beta_word(10, 0.5));
        assert_eq!(depth, 10);
        assert!((beta - 0.5).abs() < 1.0 / 255.0);
    }
}
