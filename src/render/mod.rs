mod pass;
mod pipeline;

use std::collections::HashMap;

use bevy::asset::AssetId;
use bevy::camera::visibility::Visibility;
use bevy::core_pipeline::core_3d::{main_opaque_pass_3d, main_transparent_pass_3d};
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::ecs::system::ResMut;
use bevy::math::{Mat4, Vec3, Vec4};
use bevy::pbr::MeshPipelineSystems;
use bevy::prelude::*;
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems};

use crate::HairGroom;
use crate::mesh::HairMesh;

use pass::{GpuGroom, HairPassState};

pub struct HairPlugin;

impl Plugin for HairPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<HairMesh>();

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<ExtractedFrame>()
            .init_resource::<HairPassState>()
            .add_systems(
                RenderStartup,
                pipeline::init_pipelines.after(MeshPipelineSystems),
            )
            .add_systems(ExtractSchedule, extract_hair)
            .add_systems(Render, prepare_gpu.in_set(RenderSystems::PrepareResources))
            .add_systems(
                Core3d,
                pass::hair_pass
                    .after(main_opaque_pass_3d)
                    .before(main_transparent_pass_3d)
                    .in_set(Core3dSystems::MainPass),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(handles) = app
            .world_mut()
            .get_resource_mut::<Assets<Shader>>()
            .map(|mut shaders| pipeline::load_shaders(&mut shaders))
        else {
            return;
        };
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.insert_resource(handles);
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct ExtractedFrame {
    pub grooms: Vec<ExtractedGroom>,
    pub directionals: Vec<ExtractedDirectional>,
    meshes: HashMap<AssetId<HairMesh>, HairMesh>,
}

pub(crate) struct ExtractedGroom {
    asset: AssetId<HairMesh>,
    world_from_model: Mat4,
    lambda: f32,
    albedo: Vec4,
    roughness: f32,
    tilt: f32,
    lod: bool,
    filter: bool,
    ambient_occlusion: bool,
    deep_opacity: bool,
    center_diameter: f32,
}

pub(crate) struct ExtractedDirectional {
    direction_to_light: Vec3,
    casts_shadows: bool,
}

fn extract_hair(
    grooms: Extract<Query<(&HairGroom, &GlobalTransform, Option<&Visibility>)>>,
    lights: Extract<Query<(&DirectionalLight, &GlobalTransform)>>,
    meshes: Extract<Res<Assets<HairMesh>>>,
    mut extracted: ResMut<ExtractedFrame>,
) {
    extracted.grooms.clear();
    extracted.directionals.clear();
    for (groom, transform, visibility) in &grooms {
        if visibility.is_some_and(|visibility| *visibility == Visibility::Hidden) {
            continue;
        }
        let Some(mesh) = meshes.get(&groom.mesh) else {
            continue;
        };
        let id = groom.mesh.id();
        extracted.meshes.entry(id).or_insert_with(|| mesh.clone());
        let linear = groom.albedo.to_linear();
        extracted.grooms.push(ExtractedGroom {
            asset: id,
            world_from_model: transform.to_matrix(),
            lambda: groom.lambda,
            albedo: Vec4::new(linear.red, linear.green, linear.blue, 1.0),
            roughness: groom.roughness,
            tilt: groom.tilt,
            lod: groom.lod,
            filter: groom.filter,
            ambient_occlusion: groom.ambient_occlusion,
            deep_opacity: groom.deep_opacity,
            center_diameter: groom.center_diameter,
        });
    }
    for (light, transform) in &lights {
        extracted.directionals.push(ExtractedDirectional {
            direction_to_light: transform.back().as_vec3(),
            casts_shadows: light.shadow_maps_enabled,
        });
    }
}

fn prepare_gpu(
    mut state: ResMut<HairPassState>,
    extracted: Res<ExtractedFrame>,
    device: Res<bevy::render::renderer::RenderDevice>,
) {
    for groom in &extracted.grooms {
        if state.grooms.contains_key(&groom.asset) {
            continue;
        }
        let Some(mesh) = extracted.meshes.get(&groom.asset) else {
            continue;
        };
        state
            .grooms
            .insert(groom.asset, GpuGroom::upload(&device, mesh));
    }
}

pub(crate) fn flag(value: bool) -> u32 {
    u32::from(value)
}
