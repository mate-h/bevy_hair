//! Deferred software rasterization for strand-based hair.
//!
//! Implements the pipeline from Lipp, Jarabo, Wimmer, and Bode,
//! "Deferred Software Rasterization for Efficient Real-time Hair Rendering"
//! (2026): hair-mesh strand generation, a 64-bit atomic G-buffer, level of detail,
//! Chiang shading with a deep opacity map, and the reconnection filter.

mod bake;
mod env;
#[cfg(test)]
mod gbuffer;
mod hair_file;
#[cfg(test)]
mod lod;
mod mesh;
mod obj;
mod render;

use bevy::prelude::*;

pub use bake::{BAKE_FINGERPRINT, bake_hair_mesh};
pub use hair_file::{HairError, HairStrands, load_hair_path, parse_hair};
pub use mesh::{BundleDesc, CageCorner, HairMesh, LAYER_COUNT};
pub use obj::load_obj_path;
pub use render::HairPlugin;

/// A groom instance. The mesh is a baked hair mesh; shading parameters are uniform
/// across the groom and varied along each strand by the stored `uvw` coordinate.
#[derive(Component, Clone)]
pub struct HairGroom {
    pub mesh: Handle<HairMesh>,
    pub lambda: f32,
    pub albedo: Color,
    pub roughness: f32,
    pub tilt: f32,
    pub lod: bool,
    pub filter: bool,
    pub ambient_occlusion: bool,
    pub deep_opacity: bool,
    /// Diameter, in pixels, of the center-sample acceptance disk.
    pub center_diameter: f32,
}

impl Default for HairGroom {
    fn default() -> Self {
        Self {
            mesh: Handle::default(),
            lambda: 3.0,
            albedo: Color::srgb(0.16, 0.07, 0.035),
            roughness: 0.2,
            tilt: 0.08,
            lod: true,
            filter: true,
            ambient_occlusion: true,
            deep_opacity: true,
            center_diameter: 1.0,
        }
    }
}
