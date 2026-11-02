//! Hair shadows on other surfaces.

use bevy::pbr::{ExtendedMaterial, MaterialExtension, StandardMaterial};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;

/// The most lights that can carry [`HairShadowCaster`]. Extra ones are ignored.
pub const MAX_HAIR_SHADOW_CASTERS: usize = 4;

/// Marks a directional, point, or spot light whose light hair blocks on
/// [`HairShadowMaterial`] surfaces. [`crate::HairPlugin`] renders a deep opacity
/// map from each one.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Default)]
pub struct HairShadowCaster;

/// A [`StandardMaterial`] that hair shadows. Each [`HairShadowCaster`] light is
/// attenuated by the strands between the surface and that light. Build the
/// extension with [`HairShadow::receiver`].
///
/// Bevy 0.20 has no per-light shadow hook, so the shader runs Bevy's lighting,
/// then removes the occluded share of each caster's direct light. That share
/// is evaluated for the base layer, without clearcoat, anisotropy, or
/// transmission.
///
/// The map is the one the previous frame rendered, so a moving light or groom
/// lags its shadow by a frame.
pub type HairShadowMaterial = ExtendedMaterial<StandardMaterial, HairShadowReceiver>;

/// Binds the deep opacity map and its light view for [`HairShadowMaterial`].
#[derive(Asset, AsBindGroup, Reflect, Clone, Debug)]
pub struct HairShadowReceiver {
    #[storage(100, read_only)]
    dom: Handle<ShaderBuffer>,
    #[storage(101, read_only)]
    light: Handle<ShaderBuffer>,
}

impl MaterialExtension for HairShadowReceiver {
    fn fragment_shader() -> ShaderRef {
        "embedded://bevy_hair/render/shaders/hair_shadow.wesl".into()
    }
}

/// The deep opacity maps [`crate::HairPlugin`] publishes for materials, one per
/// [`HairShadowCaster`]. With several grooms, the last one drawn casts.
#[derive(Resource, Clone)]
pub struct HairShadow {
    pub(crate) dom: Handle<ShaderBuffer>,
    pub(crate) light: Handle<ShaderBuffer>,
}

impl HairShadow {
    pub fn receiver(&self) -> HairShadowReceiver {
        HairShadowReceiver {
            dom: self.dom.clone(),
            light: self.light.clone(),
        }
    }
}
