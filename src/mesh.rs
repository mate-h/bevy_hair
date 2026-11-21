//! Hair-mesh groom consumed by the software rasterizer.
//!
//! A groom is a set of quad bundles. Each bundle has [`LAYER_COUNT`] layers of four
//! corners, plus a 4×4×65 styling volume of residuals from the Hermite cage,
//! with 33- and 17-sample levels along the strand for coarser LOD.

use bevy::asset::Asset;
use bevy::math::{Vec2, Vec3};
use bevy::reflect::TypePath;
use half::f16;

pub const LAYER_COUNT: u32 = 16;
pub const STYLE_U: u32 = 4;
pub const STYLE_V: u32 = 4;
/// Residual samples along the strand at full detail. LOD control point counts
/// are 2^k + 1, so with 2^6 + 1 samples every control point lands on one.
pub const STYLE_W: u32 = 65;
/// Levels along w: 65, 33 and 17 samples, for 65, 33 and 17 control points.
pub const STYLE_LEVELS: u32 = 3;
pub const STYLE_TEXELS: u32 = style_level_base(STYLE_LEVELS);

/// Samples along w at `level`.
pub const fn style_level_w(level: u32) -> u32 {
    ((STYLE_W - 1) >> level) + 1
}

/// Texels before `level` in a bundle's style block.
pub const fn style_level_base(level: u32) -> u32 {
    let mut base = 0;
    let mut l = 0;
    while l < level {
        base += STYLE_U * STYLE_V * style_level_w(l);
        l += 1;
    }
    base
}

/// Texel of `(u, v, w)` at `level` within a bundle's block. w is innermost, so
/// a strand, whose uv is fixed, reads contiguous runs.
pub fn style_index(level: u32, u: u32, v: u32, w: u32) -> usize {
    (style_level_base(level) + (v * STYLE_U + u) * style_level_w(level) + w) as usize
}

/// The style column a uv reads. Nearest, not bilinear: a bundle holds a few
/// strands, most with a column to themselves, and blending columns of
/// out-of-phase curls would flatten them.
pub fn style_cell(uv: Vec2) -> (u32, u32) {
    let cell = |x: f32, n: u32| ((x.clamp(0.0, 1.0) * n as f32) as u32).min(n - 1);
    (cell(uv.x, STYLE_U), cell(uv.y, STYLE_V))
}

/// A style residual as three f16, `[x | y << 16, z]`.
pub fn pack_residual(r: Vec3) -> [u32; 2] {
    let h = |x: f32| u32::from(f16::from_f32(x).to_bits());
    [h(r.x) | (h(r.y) << 16), h(r.z)]
}

pub fn unpack_residual(texel: [u32; 2]) -> Vec3 {
    let f = |bits: u32| f16::from_bits(bits as u16).to_f32();
    Vec3::new(
        f(texel[0] & 0xffff),
        f(texel[0] >> 16),
        f(texel[1] & 0xffff),
    )
}

#[derive(Clone, Copy, Debug)]
pub struct BundleDesc {
    pub layer_offset: u32,
    pub style_offset: u32,
    pub strand_count: u32,
}

/// One cage corner. `tangent` is the derivative of position with respect to the layer index.
#[derive(Clone, Copy, Debug)]
pub struct CageCorner {
    pub position: Vec3,
    pub tangent: Vec3,
    pub ao: f32,
}

#[derive(Asset, TypePath, Clone)]
pub struct HairMesh {
    pub bundles: Vec<BundleDesc>,
    /// `LAYER_COUNT * 4` corners per bundle, layer-major then corner order
    /// `c00, c10, c01, c11`.
    pub corners: Vec<CageCorner>,
    /// Residuals from the Hermite cage, `STYLE_TEXELS` per bundle, packed by
    /// [`pack_residual`] and laid out by [`style_index`].
    pub style: Vec<[u32; 2]>,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
}

impl HairMesh {
    pub fn strand_count(&self) -> u32 {
        self.bundles.iter().map(|b| b.strand_count).sum()
    }

    /// Layer quad in corner order `c00, c10, c01, c11`.
    pub fn layer_corners(&self, bundle: usize, layer: usize) -> [Vec3; 4] {
        let offset = self.bundles[bundle].layer_offset as usize + layer * 4;
        [
            self.corners[offset].position,
            self.corners[offset + 1].position,
            self.corners[offset + 2].position,
            self.corners[offset + 3].position,
        ]
    }

    /// Barycentric uv of `point` on a parallelogram quad (`c00, c10, c01, c11`).
    pub fn quad_uv(quad: [Vec3; 4], point: Vec3) -> Vec2 {
        let edge_u = quad[1] - quad[0];
        let edge_v = quad[2] - quad[0];
        let rel = point - quad[0];
        let normal = edge_u.cross(edge_v);
        let denom = normal.length_squared();
        if denom < 1e-12 {
            let u = if edge_u.length_squared() > edge_v.length_squared()
                && edge_u.length_squared() > 1e-12
            {
                rel.dot(edge_u) / edge_u.length_squared()
            } else {
                0.5
            };
            let v = if edge_v.length_squared() > edge_u.length_squared()
                && edge_v.length_squared() > 1e-12
            {
                rel.dot(edge_v) / edge_v.length_squared()
            } else {
                0.5
            };
            return Vec2::new(u, v);
        }
        let u = rel.cross(edge_v).dot(normal) / denom;
        let v = edge_u.cross(rel).dot(normal) / denom;
        Vec2::new(u, v)
    }

    /// Distance from `point` to the plane of a quad.
    pub fn quad_plane_distance(quad: [Vec3; 4], point: Vec3) -> f32 {
        let normal = (quad[1] - quad[0]).cross(quad[2] - quad[0]);
        let len = normal.length();
        if len < 1e-8 {
            return point.distance(quad[0]);
        }
        (point - quad[0]).dot(normal) / len
    }

    /// Strand position at fixed `uv` and `w`, matching the rasterizer: bilinear layers,
    /// finite-difference tangents, Hermite base, then the styling residual.
    pub fn styled_position(&self, bundle: usize, uv: Vec2, w: f32) -> Vec3 {
        cage_position(|layer| self.layer_corners(bundle, layer), uv, w)
            + sample_style(self, bundle, uv, w)
    }
}

/// The unstyled strand at `uv` and `w`, as the rasterizer builds it: bilinear
/// layers, finite-difference tangents, then a Hermite segment. `layer` returns
/// a layer's quad. The bake fits style residuals against this curve.
pub fn cage_position(layer: impl Fn(usize) -> [Vec3; 4], uv: Vec2, w: f32) -> Vec3 {
    let layers_f = w.clamp(0.0, 1.0) * (LAYER_COUNT - 1) as f32;
    let i0 = (layers_f.floor() as usize).min(LAYER_COUNT as usize - 1);
    let i1 = (i0 + 1).min(LAYER_COUNT as usize - 1);
    let t = layers_f - i0 as f32;
    let p0 = bilinear(layer(i0), uv);
    let p1 = bilinear(layer(i1), uv);
    let m0 = layer_tangent(&layer, i0, uv);
    let m1 = layer_tangent(&layer, i1, uv);
    hermite(p0, p1, m0, m1, t)
}

fn layer_tangent(quad: &impl Fn(usize) -> [Vec3; 4], layer: usize, uv: Vec2) -> Vec3 {
    let curr = bilinear(quad(layer), uv);
    let prev = bilinear(quad(layer.saturating_sub(1)), uv);
    let next = bilinear(quad((layer + 1).min(LAYER_COUNT as usize - 1)), uv);
    let count = LAYER_COUNT as usize;
    let (dir, scale) = if layer == 0 {
        let delta = next - curr;
        (delta.normalize_or(Vec3::Y), delta.length())
    } else if layer + 1 >= count {
        let delta = curr - prev;
        (delta.normalize_or(Vec3::Y), delta.length())
    } else {
        let delta = next - prev;
        (delta.normalize_or(Vec3::Y), delta.length() * 0.5)
    };
    dir * scale.max(1e-4)
}

fn hermite(p0: Vec3, p1: Vec3, m0: Vec3, m1: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    p0 * (2.0 * t3 - 3.0 * t2 + 1.0)
        + m0 * (t3 - 2.0 * t2 + t)
        + p1 * (-2.0 * t3 + 3.0 * t2)
        + m1 * (t3 - t2)
}

fn bilinear(quad: [Vec3; 4], uv: Vec2) -> Vec3 {
    let u = uv.x;
    let v = uv.y;
    quad[0] * (1.0 - u) * (1.0 - v)
        + quad[1] * u * (1.0 - v)
        + quad[2] * (1.0 - u) * v
        + quad[3] * u * v
}

/// Full-detail residual, as the rasterizer samples it at 65 control points.
fn sample_style(mesh: &HairMesh, bundle: usize, uv: Vec2, w: f32) -> Vec3 {
    let offset = mesh.bundles[bundle].style_offset as usize;
    let (u, v) = style_cell(uv);
    let ww = w.clamp(0.0, 1.0) * (STYLE_W - 1) as f32;
    let w0 = (ww.floor() as u32).min(STYLE_W - 1);
    let w1 = (w0 + 1).min(STYLE_W - 1);
    let at = |iw: u32| unpack_residual(mesh.style[offset + style_index(0, u, v, iw)]);
    at(w0).lerp(at(w1), ww - w0 as f32)
}
