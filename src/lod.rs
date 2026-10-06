//! Level-of-detail selection from Lipp et al. 2026, Section 3.2, Eqs. 1–5.

/// `L = clamp(||AABB_max - AABB_min|| / R_y * lambda, 0, 1)`.
pub fn lod_selection(aabb_extent: f32, screen_height: f32, lambda: f32) -> f32 {
    if screen_height <= f32::EPSILON {
        return 0.0;
    }
    (aabb_extent / screen_height * lambda).clamp(0.0, 1.0)
}

/// Eq. 1. Frustum-culled bundles return 0; every visible bundle keeps at least one strand.
pub fn strand_count_lod(l: f32, strand_count: u32, delta: f32, culled: bool) -> u32 {
    if culled || strand_count == 0 {
        return 0;
    }
    let n = strand_count as f32;
    let raw = (l.clamp(0.0, 1.0) * (n + delta.clamp(0.0, 0.999_999))).ceil();
    raw.clamp(1.0, n) as u32
}

/// Eqs. 4–5. `c_layers` is the hard lower bound on `C_raw` before the power-of-two snap.
pub fn control_point_count(l: f32, layer_count: u32) -> u32 {
    const C_MAX: f32 = 127.0;
    let l = l.clamp(0.0, 1.0);
    let c_raw = (l.sqrt() * C_MAX).floor().max(layer_count as f32);
    if c_raw < 3.0 {
        return c_raw.max(2.0) as u32;
    }
    let snapped = 2.0f32.powf((c_raw - 1.0).log2().floor()) + 1.0;
    snapped.min(C_MAX) as u32
}

/// Eq. 8. Unitless optical-depth shift toward the light. The shader multiplies this
/// by one hair-mesh layer of light-view depth and subtracts it before the deep
/// opacity lookup, so a thinned groom does not sample the shell that was removed.
pub fn depth_correction(beta: f32) -> f32 {
    let beta = beta.clamp(1e-4, 1.0);
    -beta.ln()
}

/// Hits in front of a shaded point, from Yuksel and Keyser 2008.
///
/// `t` is the depth past the texel’s front surface, in layers. Completed layers
/// count fully. The layer that contains the point counts only the fraction in
/// front of it. Hair at the same depth or behind the point does not count.
/// `dom_optical_depth` in `shade.wgsl` follows this.
pub fn dom_optical_depth(layers: &[u32], t: f32) -> f32 {
    let mut occ = 0.0;
    for (slice, count) in layers.iter().copied().enumerate() {
        let i = slice as f32;
        if i + 1.0 <= t {
            occ += count as f32;
        } else if i < t {
            occ += count as f32 * (t - i);
        }
    }
    occ
}

/// Screen AABB is completely outside the viewport.
pub fn aabb_outside_viewport(
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    width: f32,
    height: f32,
) -> bool {
    max_x < 0.0 || max_y < 0.0 || min_x > width || min_y > height
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lod_selection_clamps() {
        assert!((lod_selection(1080.0, 1080.0, 1.0) - 1.0).abs() < 1e-5);
        assert!((lod_selection(100.0, 1080.0, 3.0) - (300.0 / 1080.0)).abs() < 1e-5);
        assert_eq!(lod_selection(10_000.0, 1080.0, 3.0), 1.0);
    }

    #[test]
    fn strand_count_respects_cull_and_bounds() {
        assert_eq!(strand_count_lod(1.0, 100, 0.2, true), 0);
        assert_eq!(strand_count_lod(1.0, 100, 0.2, false), 100);
        assert_eq!(strand_count_lod(0.0, 100, 0.0, false), 1);
        let mid = strand_count_lod(0.5, 100, 0.0, false);
        assert_eq!(mid, 50);
    }

    #[test]
    fn control_points_snap_to_power_of_two_plus_one() {
        let full = control_point_count(1.0, 16);
        assert!(full >= 3);
        assert_eq!(
            (full - 1).count_ones(),
            1,
            "C-1 is a power of two, got {full}"
        );
        assert!(full <= 127);
        let low = control_point_count(0.0, 16);
        // C_raw is at least the layer count, then snapped.
        assert!(low >= 3);
        assert_eq!((low - 1).count_ones(), 1);
    }

    #[test]
    fn viewport_cull() {
        assert!(aabb_outside_viewport(
            -10.0, -10.0, -1.0, -1.0, 100.0, 100.0
        ));
        assert!(!aabb_outside_viewport(-10.0, -10.0, 5.0, 5.0, 100.0, 100.0));
    }

    #[test]
    fn depth_correction_grows_as_strands_are_removed() {
        assert!(depth_correction(1.0).abs() < 1e-4);
        assert!(depth_correction(0.5) > depth_correction(1.0));
        assert!(depth_correction(0.1) > depth_correction(0.5));
    }

    #[test]
    fn deep_opacity_ignores_hair_at_or_behind_the_sample() {
        let layers = [100, 40, 7];
        assert_eq!(dom_optical_depth(&layers, 0.0), 0.0);
        assert_eq!(dom_optical_depth(&layers, -1.0), 0.0);
        // Halfway through the front layer: half of that layer, nothing behind it.
        assert!((dom_optical_depth(&layers, 0.5) - 50.0).abs() < 1e-4);
        // The boundary includes the layer just completed and excludes the next one.
        assert!((dom_optical_depth(&layers, 1.0) - 100.0).abs() < 1e-4);
        assert!((dom_optical_depth(&layers, 1.25) - 110.0).abs() < 1e-4);
        assert!((dom_optical_depth(&layers, 3.0) - 147.0).abs() < 1e-4);
    }
}
