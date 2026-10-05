//! Hair-mesh groom consumed by the software rasterizer.
//!
//! A groom is a set of quad bundles. Each bundle has [`LAYER_COUNT`] layers of four
//! corners, plus an 8×8×16 styling volume of residuals from the Hermite cage.

use bevy::asset::Asset;
use bevy::math::{Vec2, Vec3};
use bevy::reflect::TypePath;

pub const LAYER_COUNT: u32 = 16;
pub const STYLE_U: u32 = 8;
pub const STYLE_V: u32 = 8;
pub const STYLE_W: u32 = 16;
pub const STYLE_TEXELS: u32 = STYLE_U * STYLE_V * STYLE_W;

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
    /// Residual xyz from the bilinear cage. W is unused.
    pub style: Vec<[f32; 4]>,
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
        let layers_f = w.clamp(0.0, 1.0) * (LAYER_COUNT - 1) as f32;
        let i0 = (layers_f.floor() as usize).min(LAYER_COUNT as usize - 1);
        let i1 = (i0 + 1).min(LAYER_COUNT as usize - 1);
        let t = layers_f - i0 as f32;
        let p0 = bilinear(self.layer_corners(bundle, i0), uv);
        let p1 = bilinear(self.layer_corners(bundle, i1), uv);
        let m0 = layer_tangent(self, bundle, i0, uv);
        let m1 = layer_tangent(self, bundle, i1, uv);
        hermite(p0, p1, m0, m1, t) + sample_style(self, bundle, uv, w)
    }
}

fn layer_tangent(mesh: &HairMesh, bundle: usize, layer: usize, uv: Vec2) -> Vec3 {
    let curr = bilinear(mesh.layer_corners(bundle, layer), uv);
    let prev = bilinear(mesh.layer_corners(bundle, layer.saturating_sub(1)), uv);
    let next = bilinear(
        mesh.layer_corners(bundle, (layer + 1).min(LAYER_COUNT as usize - 1)),
        uv,
    );
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

fn sample_style(mesh: &HairMesh, bundle: usize, uv: Vec2, w: f32) -> Vec3 {
    let offset = mesh.bundles[bundle].style_offset as usize;
    let u = uv.x.clamp(0.0, 1.0) * (STYLE_U - 1) as f32;
    let v = uv.y.clamp(0.0, 1.0) * (STYLE_V - 1) as f32;
    let ww = w.clamp(0.0, 1.0) * (STYLE_W - 1) as f32;
    let u0 = u.floor() as usize;
    let v0 = v.floor() as usize;
    let w0 = ww.floor() as usize;
    let u1 = (u0 + 1).min(STYLE_U as usize - 1);
    let v1 = (v0 + 1).min(STYLE_V as usize - 1);
    let w1 = (w0 + 1).min(STYLE_W as usize - 1);
    let tu = u - u0 as f32;
    let tv = v - v0 as f32;
    let tw = ww - w0 as f32;
    let at = |iu: usize, iv: usize, iw: usize| {
        let texel = mesh.style[offset + (iw * STYLE_V as usize + iv) * STYLE_U as usize + iu];
        Vec3::new(texel[0], texel[1], texel[2])
    };
    let c00 = at(u0, v0, w0).lerp(at(u1, v0, w0), tu);
    let c01 = at(u0, v1, w0).lerp(at(u1, v1, w0), tu);
    let c10 = at(u0, v0, w1).lerp(at(u1, v0, w1), tu);
    let c11 = at(u0, v1, w1).lerp(at(u1, v1, w1), tu);
    c00.lerp(c01, tv).lerp(c10.lerp(c11, tv), tw)
}
