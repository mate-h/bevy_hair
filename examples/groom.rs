//! Fly around the Cem Yuksel woman grooms through the deferred hair rasterizer.
//!
//! ```text
//! cargo run --example groom
//! ```
//!
//! Right-drag to look, WASD to move, Q/E to rise and fall, scroll to change
//! speed. Keys: 1/2/3 switch grooms, L/F/O/D toggle LOD, the reconnection
//! filter, ambient occlusion, and the deep opacity map, and `[` / `]` change
//! lambda.

use std::f32::consts::FRAC_PI_2;
use std::path::PathBuf;
use std::time::Instant;

use bevy::camera::Hdr;
use bevy::pbr::StandardMaterial;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::render_resource::{TextureUsages, WgpuFeatures};
use bevy::render::settings::WgpuSettings;
use bevy_camera_controller::free_camera::{FreeCamera, FreeCameraPlugin};
use bevy_hair::{HairGroom, HairMesh, HairPlugin, bake_hair_mesh, load_hair_path, load_obj_path};

const GROOM_FILES: [&str; 3] = ["wStraight.hair", "wWavy.hair", "wCurly.hair"];
const GROOM_NAMES: [&str; 3] = ["straight", "wavy", "curly"];

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
                }),
        )
        .add_plugins((HairPlugin, FreeCameraPlugin))
        .insert_resource(ClearColor(Color::srgb(0.04, 0.045, 0.05)))
        .insert_resource(prepared)
        .add_systems(Startup, setup)
        .add_systems(Update, (controls, hud))
        .run();
}

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
    let mut grooms = Vec::new();
    for file in GROOM_FILES {
        let path = asset_path(file);
        eprintln!("loading {path:?}");
        let started = Instant::now();
        let strands = load_hair_path(&path).unwrap_or_else(|err| {
            panic!("failed to read {}: {err}", path.display());
        });
        eprintln!(
            "  {} strands, {} points",
            strands.strands.len(),
            strands.points.len()
        );
        let mesh = bake_hair_mesh(&strands);
        eprintln!(
            "  baked {} bundles / {} strands in {:.1}s",
            mesh.bundles.len(),
            mesh.strand_count(),
            started.elapsed().as_secs_f32()
        );
        grooms.push(mesh);
    }
    let head_path = asset_path("woman.obj");
    let head = load_obj_path(&head_path).unwrap_or_else(|err| {
        panic!("failed to read {}: {err}", head_path.display());
    });
    Prepared {
        grooms: grooms.try_into().ok().expect("three grooms"),
        head,
    }
}

fn asset_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets/hair")
        .join(name)
}

fn setup(
    mut commands: Commands,
    mut prepared: ResMut<Prepared>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut hair: ResMut<Assets<HairMesh>>,
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
            base_color: Color::srgb(0.72, 0.55, 0.46),
            perceptual_roughness: 0.65,
            ..default()
        })),
        model,
    ));
    commands.spawn((
        HairGroom {
            mesh: handles[0].clone(),
            ..default()
        },
        model,
    ));
    commands.insert_resource(GroomLibrary { handles, active: 0 });

    commands.spawn((
        Camera3d {
            depth_texture_usages: (TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING)
                .into(),
            ..default()
        },
        Hdr,
        Msaa::Off,
        Transform::from_xyz(focus.x + 150.0, focus.y + 18.0, focus.z + 40.0)
            .looking_at(focus, Vec3::Y),
        FreeCamera {
            // The head is about 120 units tall.
            walk_speed: 40.0,
            run_speed: 120.0,
            ..default()
        },
    ));

    let key = focus + Vec3::new(70.0, 90.0, 50.0);
    let fill = focus + Vec3::new(20.0, 40.0, -80.0);
    let rim = focus + Vec3::new(-100.0, 30.0, -20.0);
    commands.spawn((
        PointLight {
            color: Color::srgb(1.0, 0.93, 0.82),
            intensity: 12_000_000.0,
            range: 500.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_translation(key),
    ));
    commands.spawn((
        PointLight {
            color: Color::srgb(0.65, 0.75, 1.0),
            intensity: 4_000_000.0,
            range: 500.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_translation(fill),
    ));
    commands.spawn((
        PointLight {
            color: Color::srgb(1.0, 0.85, 0.7),
            intensity: 3_000_000.0,
            range: 500.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_translation(rim),
    ));

    commands.spawn((
        Text::new("bevy_hair"),
        TextFont {
            font_size: FontSize::Px(16.0),
            ..default()
        },
        TextColor(Color::srgb(0.92, 0.9, 0.86)),
        Node {
            position_type: PositionType::Absolute,
            top: px(12),
            left: px(12),
            ..default()
        },
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
        }
    }
    if keys.just_pressed(KeyCode::KeyL) {
        groom.lod = !groom.lod;
    }
    if keys.just_pressed(KeyCode::KeyF) {
        groom.filter = !groom.filter;
    }
    if keys.just_pressed(KeyCode::KeyO) {
        groom.ambient_occlusion = !groom.ambient_occlusion;
    }
    if keys.just_pressed(KeyCode::KeyD) {
        groom.deep_opacity = !groom.deep_opacity;
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        groom.lambda = (groom.lambda - 0.5).max(0.5);
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        groom.lambda = (groom.lambda + 0.5).min(12.0);
    }
}

fn hud(grooms: Query<&HairGroom>, library: Res<GroomLibrary>, mut text: Query<&mut Text>) {
    let Ok(groom) = grooms.single() else {
        return;
    };
    let Ok(mut text) = text.single_mut() else {
        return;
    };
    text.0 = format!(
        "{}\nlambda {:.1}   LOD {}   filter {}   AO {}   DOM {}\n1/2/3 groom   L F O D toggles   [ ] lambda\nright-drag look   WASD move   QE up/down   scroll speed",
        GROOM_NAMES[library.active],
        groom.lambda,
        on(groom.lod),
        on(groom.filter),
        on(groom.ambient_occlusion),
        on(groom.deep_opacity),
    );
}

fn on(value: bool) -> &'static str {
    if value { "on" } else { "off" }
}
