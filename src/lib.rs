//! Deferred software rasterization for strand-based hair.
//!
//! Implements the pipeline from Lipp, Jarabo, Wimmer, and Bode,
//! "Deferred Software Rasterization for Efficient Real-time Hair Rendering"
//! (2026): hair-mesh strand generation, a 64-bit atomic G-buffer, level of detail,
//! Chiang shading with a deep opacity map, and the reconnection filter.

mod bake;
#[cfg(test)]
mod env;
#[cfg(test)]
mod gbuffer;
mod hair_file;
#[cfg(test)]
mod lod;
mod mesh;
mod obj;
mod render;
mod shadow;

use bevy::prelude::*;

pub use bake::{BAKE_FINGERPRINT, Scalp, bake_hair_mesh, bake_hair_mesh_with_scalp};
pub use hair_file::{HairError, HairStrands, load_hair_path, parse_hair};
pub use mesh::{BundleDesc, CageCorner, HairMesh, LAYER_COUNT};
pub use obj::load_obj_path;
pub use render::HairPlugin;
pub use shadow::{HairShadow, HairShadowMaterial, HairShadowReceiver};

/// A groom instance. The mesh is a baked hair mesh; shading parameters are set per groom,
/// varied along each strand by the stored `uvw` coordinate and between strands
/// by a hash of its `uv`.
#[derive(Component, Clone)]
pub struct HairGroom {
    pub mesh: Handle<HairMesh>,
    pub lambda: f32,
    /// Multiple-scattered hair color. The shader solves for the fiber absorption
    /// that produces it.
    pub albedo: Color,
    pub roughness: f32,
    /// Per-strand spread of melanin: each strand's albedo is raised to a power
    /// in `[2^-v, 2^v]`, which scales its absorption by about that factor.
    pub melanin_variation: f32,
    /// Per-strand relative spread of roughness, `roughness * (1 ± v)`.
    pub roughness_variation: f32,
    pub tilt: f32,
    pub lod: bool,
    pub filter: bool,
    pub ambient_occlusion: bool,
    pub deep_opacity: bool,
    /// Diameter, in pixels, of the center-sample acceptance disk.
    pub center_diameter: f32,
    /// Fiber diameter in the hair mesh's units. Sets how much light each strand
    /// blocks in the deep opacity maps, independent of their resolution. The
    /// default is 100 µm in metres. A groom with fewer strands than real hair
    /// needs proportionally wider fibers to stay as dense.
    pub fiber_width: f32,
}

impl Default for HairGroom {
    fn default() -> Self {
        Self {
            mesh: Handle::default(),
            lambda: 3.0,
            albedo: Color::srgb(0.16, 0.07, 0.035),
            roughness: 0.2,
            melanin_variation: 0.35,
            roughness_variation: 0.25,
            tilt: 0.08,
            lod: true,
            filter: true,
            ambient_occlusion: true,
            deep_opacity: true,
            center_diameter: 1.0,
            fiber_width: 1e-4,
        }
    }
}
