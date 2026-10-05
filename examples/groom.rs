//! Fly around the Cem Yuksel woman grooms through the deferred hair rasterizer.
//!
//! ```text
//! cargo run --example groom
//! ```
//!
//! Groom controls are logged after the free-camera help. `C` draws each baked bundle
//! as a colored wireframe cage.
//!
//! Baked meshes are reused from `target/groom-cache/` until the hair file, the
//! scalp, or the baker changes. `cargo run --example bake` fills that cache
//! without opening a window.

mod bake;

use std::f32::consts::FRAC_PI_2;
use std::fmt;

use bevy::app::{PluginGroup, RunFixedMainLoop, RunFixedMainLoopSystems};
use bevy::camera::Hdr;
use bevy::light::Skybox;
use bevy::log::LogPlugin;
use bevy::pbr::StandardMaterial;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::render_resource::{TextureUsages, WgpuFeatures};
use bevy::render::settings::WgpuSettings;
use bevy_camera_controller::free_camera::{
    FreeCamera, FreeCameraPlugin, run_freecamera_controller,
};
use bevy_hair::{HairGroom, HairMesh, HairPlugin, LAYER_COUNT, Scalp};

fn main() {
    let mut wgpu = WgpuSettings::default();
    wgpu.features |= WgpuFeatures::SUBGROUP
        | WgpuFeatures::SHADER_INT64
        | WgpuFeatures::SHADER_INT64_ATOMIC_MIN_MAX;

    let prepared = prepare_assets();

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(RenderPlugin {
                    render_creation: wgpu.into(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "bevy_hair".into(),
                        ..default()
                    }),
                    ..default()
                })
                .build()
                .disable::<LogPlugin>(),
        )
        .add_plugins((HairPlugin, FreeCameraPlugin))
        .insert_resource(ClearColor(Color::srgb(0.04, 0.045, 0.05)))
        .insert_resource(prepared)
        .insert_resource(ShowCage(false))
        .add_systems(Startup, setup)
        .add_systems(
            RunFixedMainLoop,
            print_groom_controls
                .after(run_freecamera_controller)
                .in_set(RunFixedMainLoopSystems::BeforeFixedMainLoop),
        )
        .add_systems(Update, (controls, draw_cages))
        .run();
}

struct GroomControls;

impl fmt::Display for GroomControls {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "
bevy_hair Controls:
    {:?} & {:?} & {:?}\t- Switch straight, wavy, and curly groom
    {:?}\t- Toggle level of detail
    {:?}\t- Toggle reconnection filter
    {:?}\t- Toggle ambient occlusion
    {:?}\t- Toggle deep opacity map
    {:?} & {:?}\t- Decrease and increase lambda
    {:?}\t- Toggle bundle cage wireframe",
            KeyCode::Digit1,
            KeyCode::Digit2,
            KeyCode::Digit3,
            KeyCode::KeyL,
            KeyCode::KeyF,
            KeyCode::KeyO,
            KeyCode::KeyM,
            KeyCode::BracketLeft,
            KeyCode::BracketRight,
            KeyCode::KeyC,
        )
    }
}

fn print_groom_controls(mut printed: Local<bool>, cameras: Query<(), With<Camera>>) {
    if *printed || cameras.is_empty() {
        return;
    }
    *printed = true;
    info!("{}", GroomControls);
}

#[derive(Resource)]
struct ShowCage(bool);

#[derive(Resource)]
struct Prepared {
    grooms: [HairMesh; 3],
    head: Mesh,
}

#[derive(Resource)]
struct GroomLibrary {
    handles: [Handle<HairMesh>; 3],
    active: usize,
}

fn prepare_assets() -> Prepared {
    let head_path = bake::asset_path("woman.obj");
    let head = bevy_hair::load_obj_path(&head_path).unwrap_or_else(|err| {
        panic!("failed to read {}: {err}", head_path.display());
    });
    let scalp = Scalp::from_mesh(&head);
    let grooms = bake::GROOM_FILES.map(|file| bake::load(&bake::asset_path(file), &scalp));
    Prepared { grooms, head }
}

fn setup(
    mut commands: Commands,
    mut prepared: ResMut<Prepared>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut hair: ResMut<Assets<HairMesh>>,
    asset_server: Res<AssetServer>,
) {
    let handles = [
        hair.add(std::mem::replace(&mut prepared.grooms[0], empty_groom())),
        hair.add(std::mem::replace(&mut prepared.grooms[1], empty_groom())),
        hair.add(std::mem::replace(&mut prepared.grooms[2], empty_groom())),
    ];
    let head = std::mem::replace(
        &mut prepared.head,
        Mesh::new(
            bevy::mesh::PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::default(),
        ),
    );

    // The Cem Yuksel grooms are Z-up. Bevy is Y-up.
    let model = Transform::from_rotation(Quat::from_rotation_x(-FRAC_PI_2));
    let focus = Vec3::new(-5.0, 15.0, 0.0);
    commands.spawn((
        Mesh3d(meshes.add(head)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.73, 0.52, 0.44),
            perceptual_roughness: 0.45,
            metallic: 0.0,
            specular_tint: Color::srgb(1.0, 0.74, 0.62),
            diffuse_transmission: 0.18,
            attenuation_color: Color::srgb(0.9, 0.22, 0.12),
            attenuation_distance: 6.0,
            ..default()
        })),
        model,
    ));
    commands.spawn((
        HairGroom {
            mesh: handles[0].clone(),
            // Roughness picks a prefiltered specular mip. Tilt shifts the primary R highlight.
            albedo: Color::srgb(0.26, 0.11, 0.05),
            roughness: 0.45,
            tilt: 0.05,
            ..default()
        },
        model,
    ));
    commands.insert_resource(GroomLibrary { handles, active: 0 });

    // One value for the face IBL, the skybox, and the hair probe lookup.
    const ENV_INTENSITY: f32 = 200.0;
    let diffuse = asset_server.load("env/little_paris_eiffel_tower_2k_diffuse.ktx2");
    let specular = asset_server.load("env/little_paris_eiffel_tower_2k_specular.ktx2");
    let skybox = asset_server.load("env/little_paris_eiffel_tower_2k_skybox.ktx2");
    commands.spawn((
        Camera3d {
            depth_texture_usages: (TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING)
                .into(),
            ..default()
        },
        Hdr,
        Msaa::Off,
        EnvironmentMapLight {
            diffuse_map: diffuse,
            specular_map: specular,
            intensity: ENV_INTENSITY,
            ..default()
        },
        Skybox {
            image: Some(skybox),
            brightness: ENV_INTENSITY,
            ..default()
        },
        Transform::from_xyz(focus.x + 150.0, focus.y + 18.0, focus.z + 40.0)
            .looking_at(focus, Vec3::Y),
        FreeCamera {
            // The head is about 120 units tall.
            walk_speed: 40.0,
            run_speed: 120.0,
            // `M` toggles the deep opacity map in `controls`.
            keyboard_key_toggle_cursor_grab: KeyCode::KeyG,
            ..default()
        },
    ));

    // The bake's white point removes the sun. This key replaces it.
    // Exposure stays at Bevy's default, EV 9.7.
    let key = focus + Vec3::new(70.0, 90.0, 50.0);
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.95, 0.86),
            illuminance: 60_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_translation(key).looking_at(focus, Vec3::Y),
    ));
}

fn empty_groom() -> HairMesh {
    HairMesh {
        bundles: Vec::new(),
        corners: Vec::new(),
        style: Vec::new(),
        bounds_min: Vec3::ZERO,
        bounds_max: Vec3::ZERO,
    }
}

fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    mut grooms: Query<&mut HairGroom>,
    mut library: ResMut<GroomLibrary>,
    mut cages: ResMut<ShowCage>,
) {
    let Ok(mut groom) = grooms.single_mut() else {
        return;
    };
    for (index, key) in [KeyCode::Digit1, KeyCode::Digit2, KeyCode::Digit3]
        .into_iter()
        .enumerate()
    {
        if keys.just_pressed(key) {
            library.active = index;
            groom.mesh = library.handles[index].clone();
            info!("groom: {}", bake::GROOM_FILES[index]);
        }
    }
    if keys.just_pressed(KeyCode::KeyL) {
        groom.lod = !groom.lod;
        info!("level of detail: {}", on_off(groom.lod));
    }
    if keys.just_pressed(KeyCode::KeyF) {
        groom.filter = !groom.filter;
        info!("reconnection filter: {}", on_off(groom.filter));
    }
    if keys.just_pressed(KeyCode::KeyO) {
        groom.ambient_occlusion = !groom.ambient_occlusion;
        info!("ambient occlusion: {}", on_off(groom.ambient_occlusion));
    }
    if keys.just_pressed(KeyCode::KeyM) {
        groom.deep_opacity = !groom.deep_opacity;
        info!("deep opacity map: {}", on_off(groom.deep_opacity));
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        groom.lambda = (groom.lambda - 0.5).max(0.5);
        info!("lambda: {:.1}", groom.lambda);
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        groom.lambda = (groom.lambda + 0.5).min(12.0);
        info!("lambda: {:.1}", groom.lambda);
    }
    if keys.just_pressed(KeyCode::KeyC) {
        cages.0 = !cages.0;
        info!("cage wireframe: {}", on_off(cages.0));
    }
}

fn on_off(enabled: bool) -> &'static str {
    if enabled { "on" } else { "off" }
}

fn draw_cages(
    show: Res<ShowCage>,
    grooms: Query<(&HairGroom, &GlobalTransform)>,
    meshes: Res<Assets<HairMesh>>,
    mut gizmos: Gizmos,
) {
    if !show.0 {
        return;
    }
    for (groom, transform) in &grooms {
        let Some(mesh) = meshes.get(&groom.mesh) else {
            continue;
        };
        for bundle in 0..mesh.bundles.len() {
            let color = bundle_color(bundle);
            for layer in 0..LAYER_COUNT as usize {
                let quad = mesh.layer_corners(bundle, layer);
                let world = [
                    transform.transform_point(quad[0]),
                    transform.transform_point(quad[1]),
                    transform.transform_point(quad[3]),
                    transform.transform_point(quad[2]),
                ];
                gizmos.line(world[0], world[1], color);
                gizmos.line(world[1], world[2], color);
                gizmos.line(world[2], world[3], color);
                gizmos.line(world[3], world[0], color);
                if layer + 1 == LAYER_COUNT as usize {
                    continue;
                }
                let next = mesh.layer_corners(bundle, layer + 1);
                for corner in 0..4 {
                    gizmos.line(
                        transform.transform_point(quad[corner]),
                        transform.transform_point(next[corner]),
                        color,
                    );
                }
            }
            let mut prev =
                transform.transform_point(mesh.styled_position(bundle, Vec2::splat(0.5), 0.0));
            for step in 1..LAYER_COUNT {
                let w = step as f32 / (LAYER_COUNT - 1) as f32;
                let next =
                    transform.transform_point(mesh.styled_position(bundle, Vec2::splat(0.5), w));
                gizmos.line(prev, next, color);
                prev = next;
            }
        }
    }
}

fn bundle_color(index: usize) -> Color {
    Color::hsl((index as f32 * 137.508) % 360.0, 0.72, 0.62)
}
