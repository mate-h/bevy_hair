//! Hair shadows on other surfaces.

use bevy::pbr::{ExtendedMaterial, MaterialExtension, StandardMaterial};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;

/// A [`StandardMaterial`] that hair shadows. [`crate::HairPlugin`] renders a
/// deep opacity map from every directional, point, and spot light with a
/// [`CustomShadowChannel`](bevy::light::CustomShadowChannel), and the material
/// attenuates each light by the strands between the surface and that light.
/// Give each light its own channel. Build the extension with
/// [`HairShadow::receiver`].
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
/// custom shadow channel. With several grooms, the last one drawn casts.
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
