//! Hair-mesh groom consumed by the software rasterizer.
//!
//! A groom is a set of quad bundles. Each bundle has [`LAYER_COUNT`] layers of four
//! corners, plus an 8×8×16 styling volume of residuals from the Hermite cage.

use bevy::asset::Asset;
use bevy::math::Vec3;
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
}
