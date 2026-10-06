use std::collections::HashMap;

use bevy::asset::AssetId;
use bevy::ecs::system::SystemParam;
use bevy::math::{Mat4, Vec3, Vec4};
use bevy::pbr::{MeshPipelineViewLayoutKey, MeshViewBindGroup, ViewKeyCache};
use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::view::{ViewDepthTexture, ViewTarget};
use bytemuck::{Pod, Zeroable};

use crate::mesh::{HairMesh, LAYER_COUNT};
use crate::render::pipeline::{
    HAIR_PARAMS_SIZE, HairPipelines, HairShadePipeline, queue_composite,
};
use crate::render::{ExtractedDirectional, ExtractedFrame, flag};

pub const DOM_SIZE: u32 = 512;
/// Opacity slices per light-space texel. Keep in sync with `DOM_LAYERS` in the shaders.
pub const DOM_LAYERS: u32 = 16;
/// Beer-Lambert extinction per strand fragment. Visibility is `exp(-hits * DOM_SIGMA)`.
const DOM_SIGMA: f32 = 0.05;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HairParams {
    clip_from_world: [[f32; 4]; 4],
    world_from_clip: [[f32; 4]; 4],
    world_from_model: [[f32; 4]; 4],
    light_clip_from_world: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    camera_forward: [f32; 4],
    screen: [f32; 4],
    appearance: [f32; 4],
    filter_params: [f32; 4],
    albedo: [f32; 4],
    flags: [u32; 4],
    pass_mode: [u32; 4],
    light_eye: [f32; 4],
    light_forward: [f32; 4],
    dom_info: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<HairParams>() == HAIR_PARAMS_SIZE as usize);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLayer {
    pos_ao: [f32; 4],
    tangent: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBundle {
    layer_offset: u32,
    style_offset: u32,
    strand_count: u32,
    _pad: u32,
}

pub struct GpuGroom {
    pub bundles: Buffer,
    pub layers: Buffer,
    pub style: Buffer,
    pub bundle_count: u32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub camera_lod: Buffer,
    pub camera_refs: Buffer,
    pub camera_indirect: Buffer,
    pub dom_lod: Buffer,
    pub dom_refs: Buffer,
    pub dom_indirect: Buffer,
    pub camera_params: Buffer,
    pub dom_params: Buffer,
    pub dom_opacity_params: Buffer,
}

impl GpuGroom {
    pub fn upload(device: &RenderDevice, mesh: &HairMesh) -> Self {
        let layers: Vec<GpuLayer> = mesh
            .corners
            .iter()
            .map(|corner| GpuLayer {
                pos_ao: corner.position.extend(corner.ao).to_array(),
                tangent: corner.tangent.extend(0.0).to_array(),
            })
            .collect();
        let bundles: Vec<GpuBundle> = mesh
            .bundles
            .iter()
            .map(|bundle| GpuBundle {
                layer_offset: bundle.layer_offset,
                style_offset: bundle.style_offset,
                strand_count: bundle.strand_count,
                _pad: 0,
            })
            .collect();
        let strand_count = mesh.strand_count().max(1);
        let bundle_count = mesh.bundles.len().max(1) as u32;
        Self {
            bundles: init_buffer(device, "hair_bundles", &bundles),
            layers: init_buffer(device, "hair_layers", &layers),
            style: init_buffer(device, "hair_style", &mesh.style),
            bundle_count,
            bounds_min: mesh.bounds_min,
            bounds_max: mesh.bounds_max,
            camera_lod: zeros(
                device,
                "hair_lod",
                bundle_count as u64 * 16,
                BufferKind::Storage,
            ),
            camera_refs: zeros(
                device,
                "hair_refs",
                strand_count as u64 * 16,
                BufferKind::Storage,
            ),
            camera_indirect: zeros(device, "hair_indirect", 16, BufferKind::Indirect),
            dom_lod: zeros(
                device,
                "hair_dom_lod",
                bundle_count as u64 * 16,
                BufferKind::Storage,
            ),
            dom_refs: zeros(
                device,
                "hair_dom_refs",
                strand_count as u64 * 16,
                BufferKind::Storage,
            ),
            dom_indirect: zeros(device, "hair_dom_indirect", 16, BufferKind::Indirect),
            camera_params: zeros(device, "hair_params", HAIR_PARAMS_SIZE, BufferKind::Uniform),
            dom_params: zeros(
                device,
                "hair_dom_params",
                HAIR_PARAMS_SIZE,
                BufferKind::Uniform,
            ),
            dom_opacity_params: zeros(
                device,
                "hair_dom_opacity_params",
                HAIR_PARAMS_SIZE,
                BufferKind::Uniform,
            ),
        }
    }
}

fn init_buffer<T: Pod>(device: &RenderDevice, label: &str, data: &[T]) -> Buffer {
    let contents = if data.is_empty() {
        vec![0u8; 16]
    } else {
        bytemuck::cast_slice(data).to_vec()
    };
    device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some(label),
        contents: &contents,
        usage: BufferUsages::STORAGE,
    })
}

enum BufferKind {
    Storage,
    Indirect,
    Uniform,
}

fn zeros(device: &RenderDevice, label: &str, size: u64, kind: BufferKind) -> Buffer {
    let mut usage = BufferUsages::COPY_DST;
    match kind {
        BufferKind::Storage => usage |= BufferUsages::STORAGE,
        BufferKind::Indirect => usage |= BufferUsages::STORAGE | BufferUsages::INDIRECT,
        BufferKind::Uniform => usage |= BufferUsages::UNIFORM,
    }
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage,
        mapped_at_creation: false,
    })
}

#[derive(Default)]
pub struct FrameGpu {
    pub targets: Option<ScreenTargets>,
    pub dummy_depth: Option<TextureView>,
    shade_target: Option<ShadeTarget>,
}

struct ShadeTarget {
    width: u32,
    height: u32,
    _texture: Texture,
    view: TextureView,
}

pub struct ScreenTargets {
    pub width: u32,
    pub height: u32,
    pub center: Buffer,
    pub conservative: Buffer,
    pub beta: Buffer,
    pub shaded: Buffer,
    pub shaded_depth: Buffer,
    pub filtered: Buffer,
    pub filtered_depth: Buffer,
    pub dom: Buffer,
}

#[derive(Default, Resource)]
pub struct HairPassState {
    pub grooms: HashMap<AssetId<HairMesh>, GpuGroom>,
    pub frame: FrameGpu,
    composite_greater: Option<CachedRenderPipelineId>,
    composite_less: Option<CachedRenderPipelineId>,
    composite_key: Option<(TextureFormat, TextureFormat)>,
}

#[derive(SystemParam)]
pub(crate) struct HairPass<'w> {
    pipelines: Res<'w, HairPipelines>,
    shade_pipeline: Res<'w, HairShadePipeline>,
    cache: Res<'w, PipelineCache>,
    queue: Res<'w, RenderQueue>,
    device: Res<'w, RenderDevice>,
    extracted: Res<'w, ExtractedFrame>,
    view_keys: Res<'w, ViewKeyCache>,
}

pub fn hair_pass(
    view: ViewQuery<(
        &bevy::render::view::ExtractedView,
        &ViewTarget,
        &ViewDepthTexture,
        &MeshViewBindGroup,
    )>,
    mut ctx: RenderContext,
    hair: HairPass,
    mut state: ResMut<HairPassState>,
    mut shade_pipelines: ResMut<SpecializedRenderPipelines<HairShadePipeline>>,
) {
    let HairPass {
        pipelines,
        shade_pipeline,
        cache,
        queue,
        device,
        extracted,
        view_keys,
    } = hair;
    let (view, target, depth, mesh_view) = view.into_inner();
    let Some(mesh_key) = view_keys.get(&view.retained_view_entity) else {
        return;
    };
    let layout_key = MeshPipelineViewLayoutKey::from(*mesh_key);
    let shade_id = shade_pipelines.specialize(&cache, &shade_pipeline, layout_key);
    let Some(shade) = cache.get_render_pipeline(shade_id) else {
        return;
    };
    let Some(clear) = cache.get_compute_pipeline(pipelines.clear) else {
        return;
    };
    let Some(clear_dom) = cache.get_compute_pipeline(pipelines.clear_dom) else {
        return;
    };
    let Some(lod) = cache.get_compute_pipeline(pipelines.lod) else {
        return;
    };
    let Some(scan) = cache.get_compute_pipeline(pipelines.scan) else {
        return;
    };
    let Some(raster) = cache.get_compute_pipeline(pipelines.raster) else {
        return;
    };
    let Some(filter) = cache.get_compute_pipeline(pipelines.filter) else {
        return;
    };

    let size = target.main_texture().size();
    let width = size.width;
    let height = size.height;
    if width == 0 || height == 0 {
        return;
    }
    ensure_targets(&device, &mut state.frame, width, height);
    ensure_shade_target(&device, &mut state.frame, width, height);
    ensure_dummy_depth(&device, &mut ctx, &mut state.frame);
    let shade_view = state.frame.shade_target.as_ref().unwrap().view.clone();
    let environment_map = layout_key.contains(MeshPipelineViewLayoutKey::ENVIRONMENT_MAP);

    let world_from_view = view.world_from_view.to_matrix();
    let camera_pos = world_from_view.transform_point3(Vec3::ZERO);
    let forward = (-world_from_view.transform_vector3(Vec3::Z)).normalize_or(Vec3::NEG_Z);
    let clip_from_world = view
        .clip_from_world
        .unwrap_or_else(|| view.clip_from_view * world_from_view.inverse());
    let world_from_clip = clip_from_world.inverse();
    let reverse = depth_is_reverse(clip_from_world, camera_pos, forward);

    let depth_format = depth.texture.format();
    let color_format = target.main_texture_format();
    let composite_id = ensure_composite(
        &cache,
        &pipelines,
        &mut state,
        color_format,
        depth_format,
        reverse,
    );
    let Some(composite) = cache.get_render_pipeline(composite_id) else {
        return;
    };

    let depth_view = if depth
        .texture
        .usage()
        .contains(TextureUsages::TEXTURE_BINDING)
    {
        depth.view()
    } else {
        state.frame.dummy_depth.as_ref().unwrap()
    };

    let targets = state.frame.targets.as_ref().unwrap();
    let pixels = width * height;

    for groom in &extracted.grooms {
        let Some(gpu) = state.grooms.get(&groom.asset) else {
            continue;
        };
        {
            let group = ctx.render_device().create_bind_group(
                None,
                &cache.get_bind_group_layout(&pipelines.clear_layout),
                &BindGroupEntries::sequential((
                    targets.center.as_entire_binding(),
                    targets.conservative.as_entire_binding(),
                    targets.beta.as_entire_binding(),
                )),
            );
            let mut pass = ctx
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("hair_clear"),
                    ..default()
                });
            pass.set_pipeline(clear);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(pixels.div_ceil(256), 1, 1);
        }
        let (light_eye, light_dir, light_clip, light_near, light_far) = light_fit(
            gpu,
            groom.world_from_model,
            key_direction(&extracted.directionals),
        );
        let (center, radius) = world_extent(gpu, groom.world_from_model);
        let dist = camera_pos.distance(center).max(0.5);
        // 24-bit quantization stays precise across the whole view, so the range
        // only has to contain the groom. A tight near plane drops the front strands.
        let cam_near = 0.05;
        let cam_far = (dist + radius * 4.0).max(cam_near + 1.0);
        // Eq. 8 is one hair-mesh layer of light-view depth per e-fold of strand removal.
        let dom_scale = (light_far - light_near) / LAYER_COUNT as f32;

        let camera_params = make_params(ViewArgs {
            clip_from_world,
            world_from_clip,
            world_from_model: groom.world_from_model,
            light_clip_from_world: light_clip,
            camera_pos,
            camera_forward: forward,
            screen: Vec4::new(width as f32, height as f32, cam_near, cam_far),
            groom,
            light_forward: light_dir,
            dom_near: light_near,
            dom_far: light_far,
            dom_scale,
            pass_mode: [0, gpu.bundle_count, u32::from(reverse), 0],
            light_eye,
        });
        let dom_params = make_params(ViewArgs {
            clip_from_world: light_clip,
            world_from_clip: light_clip.inverse(),
            world_from_model: groom.world_from_model,
            light_clip_from_world: light_clip,
            camera_pos: light_eye,
            camera_forward: light_dir,
            screen: Vec4::new(DOM_SIZE as f32, DOM_SIZE as f32, light_near, light_far),
            groom,
            light_forward: light_dir,
            dom_near: light_near,
            dom_far: light_far,
            dom_scale,
            pass_mode: [1, gpu.bundle_count, u32::from(reverse), 1],
            light_eye,
        });
        let mut dom_opacity_params = dom_params;
        dom_opacity_params.pass_mode[0] = 2;
        queue.write_buffer(&gpu.camera_params, 0, bytemuck::bytes_of(&camera_params));
        queue.write_buffer(&gpu.dom_params, 0, bytemuck::bytes_of(&dom_params));
        queue.write_buffer(
            &gpu.dom_opacity_params,
            0,
            bytemuck::bytes_of(&dom_opacity_params),
        );

        let groups = (gpu.bundle_count.div_ceil(64), 1, 1);
        dispatch(
            &mut ctx,
            &cache,
            lod,
            &pipelines.lod_layout,
            BindGroupEntries::sequential((
                gpu.camera_params.as_entire_binding(),
                gpu.bundles.as_entire_binding(),
                gpu.layers.as_entire_binding(),
                gpu.camera_lod.as_entire_binding(),
            )),
            groups,
        );
        dispatch(
            &mut ctx,
            &cache,
            scan,
            &pipelines.scan_layout,
            BindGroupEntries::sequential((
                gpu.camera_params.as_entire_binding(),
                gpu.camera_lod.as_entire_binding(),
                gpu.camera_refs.as_entire_binding(),
                gpu.camera_indirect.as_entire_binding(),
            )),
            (1, 1, 1),
        );
        dispatch_raster(
            &mut ctx,
            &cache,
            raster,
            &pipelines.raster_layout,
            RasterBuffers {
                gpu,
                params: &gpu.camera_params,
                refs: &gpu.camera_refs,
                targets,
                depth_view,
                indirect: &gpu.camera_indirect,
            },
        );

        if groom.deep_opacity {
            // Depth texels, then DOM_LAYERS opacity counts. Two rasters: the
            // second bins fragments relative to the nearest depth from the first.
            let dom_words = DOM_SIZE * DOM_SIZE * (1 + DOM_LAYERS);
            {
                let group = ctx.render_device().create_bind_group(
                    None,
                    &cache.get_bind_group_layout(&pipelines.clear_dom_layout),
                    &BindGroupEntries::sequential((targets.dom.as_entire_binding(),)),
                );
                let mut pass = ctx
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("hair_clear_dom"),
                        ..default()
                    });
                pass.set_pipeline(clear_dom);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(dom_words.div_ceil(256), 1, 1);
            }
            dispatch(
                &mut ctx,
                &cache,
                lod,
                &pipelines.lod_layout,
                BindGroupEntries::sequential((
                    gpu.dom_params.as_entire_binding(),
                    gpu.bundles.as_entire_binding(),
                    gpu.layers.as_entire_binding(),
                    gpu.dom_lod.as_entire_binding(),
                )),
                groups,
            );
            dispatch(
                &mut ctx,
                &cache,
                scan,
                &pipelines.scan_layout,
                BindGroupEntries::sequential((
                    gpu.dom_params.as_entire_binding(),
                    gpu.dom_lod.as_entire_binding(),
                    gpu.dom_refs.as_entire_binding(),
                    gpu.dom_indirect.as_entire_binding(),
                )),
                (1, 1, 1),
            );
            dispatch_raster(
                &mut ctx,
                &cache,
                raster,
                &pipelines.raster_layout,
                RasterBuffers {
                    gpu,
                    params: &gpu.dom_params,
                    refs: &gpu.dom_refs,
                    targets,
                    depth_view,
                    indirect: &gpu.dom_indirect,
                },
            );
            dispatch_raster(
                &mut ctx,
                &cache,
                raster,
                &pipelines.raster_layout,
                RasterBuffers {
                    gpu,
                    params: &gpu.dom_opacity_params,
                    refs: &gpu.dom_refs,
                    targets,
                    depth_view,
                    indirect: &gpu.dom_indirect,
                },
            );
        }

        draw_shade(
            &mut ctx,
            &cache,
            ShadeDraw {
                pipeline: shade,
                layout: &shade_pipeline.layout,
                mesh_view,
                environment_map,
                shade_view: &shade_view,
                buffers: ShadeBuffers {
                    params: &gpu.camera_params,
                    targets,
                },
            },
        );

        let (color, depth_buf) = if groom.filter {
            dispatch(
                &mut ctx,
                &cache,
                filter,
                &pipelines.filter_layout,
                BindGroupEntries::sequential((
                    gpu.camera_params.as_entire_binding(),
                    targets.center.as_entire_binding(),
                    targets.conservative.as_entire_binding(),
                    targets.shaded.as_entire_binding(),
                    targets.shaded_depth.as_entire_binding(),
                    targets.filtered.as_entire_binding(),
                    targets.filtered_depth.as_entire_binding(),
                )),
                (width.div_ceil(8), height.div_ceil(4), 1),
            );
            (&targets.filtered, &targets.filtered_depth)
        } else {
            (&targets.shaded, &targets.shaded_depth)
        };

        let group = ctx.render_device().create_bind_group(
            None,
            &cache.get_bind_group_layout(&pipelines.composite_layout),
            &BindGroupEntries::sequential((
                gpu.camera_params.as_entire_binding(),
                color.as_entire_binding(),
                depth_buf.as_entire_binding(),
            )),
        );
        let color_attachment = [Some(target.get_color_attachment())];
        let depth_attachment = Some(depth.get_attachment(StoreOp::Store));
        let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("hair_composite"),
            color_attachments: &color_attachment,
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_render_pipeline(composite);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
}

struct ViewArgs<'a> {
    clip_from_world: Mat4,
    world_from_clip: Mat4,
    world_from_model: Mat4,
    light_clip_from_world: Mat4,
    camera_pos: Vec3,
    camera_forward: Vec3,
    screen: Vec4,
    groom: &'a crate::render::ExtractedGroom,
    light_forward: Vec3,
    dom_near: f32,
    dom_far: f32,
    dom_scale: f32,
    pass_mode: [u32; 4],
    light_eye: Vec3,
}

fn make_params(args: ViewArgs) -> HairParams {
    HairParams {
        clip_from_world: args.clip_from_world.to_cols_array_2d(),
        world_from_clip: args.world_from_clip.to_cols_array_2d(),
        world_from_model: args.world_from_model.to_cols_array_2d(),
        light_clip_from_world: args.light_clip_from_world.to_cols_array_2d(),
        camera_pos: args.camera_pos.extend(1.0).to_array(),
        camera_forward: args.camera_forward.extend(0.0).to_array(),
        screen: args.screen.to_array(),
        appearance: [
            args.groom.roughness,
            args.groom.tilt,
            args.groom.lambda,
            args.groom.center_diameter,
        ],
        filter_params: [1.0, 0.2, args.dom_scale.max(1e-3), DOM_SIGMA],
        albedo: args.groom.albedo.to_array(),
        flags: [
            flag(args.groom.lod),
            flag(args.groom.filter),
            flag(args.groom.ambient_occlusion),
            flag(args.groom.deep_opacity),
        ],
        pass_mode: args.pass_mode,
        light_eye: args.light_eye.extend(1.0).to_array(),
        light_forward: args.light_forward.extend(0.0).to_array(),
        dom_info: [
            DOM_SIZE as f32,
            DOM_SIZE as f32,
            args.dom_near,
            args.dom_far,
        ],
    }
}

fn key_direction(lights: &[ExtractedDirectional]) -> Vec3 {
    lights
        .iter()
        .find(|light| light.casts_shadows)
        .or_else(|| lights.first())
        .map(|light| light.direction_to_light)
        .unwrap_or_else(|| Vec3::new(0.35, 0.82, 0.45))
        .normalize_or(Vec3::Y)
}

fn world_corners(gpu: &GpuGroom, world_from_model: Mat4) -> [Vec3; 8] {
    let min = gpu.bounds_min;
    let max = gpu.bounds_max;
    let mut corners = [Vec3::ZERO; 8];
    let mut index = 0;
    for x in [min.x, max.x] {
        for y in [min.y, max.y] {
            for z in [min.z, max.z] {
                corners[index] = world_from_model.transform_point3(Vec3::new(x, y, z));
                index += 1;
            }
        }
    }
    corners
}

fn world_extent(gpu: &GpuGroom, world_from_model: Mat4) -> (Vec3, f32) {
    let corners = world_corners(gpu, world_from_model);
    let center = corners.iter().copied().sum::<Vec3>() / corners.len() as f32;
    let radius = corners
        .iter()
        .map(|corner| corner.distance(center))
        .fold(1.0, f32::max);
    (center, radius)
}

fn light_fit(
    gpu: &GpuGroom,
    world_from_model: Mat4,
    direction_to_light: Vec3,
) -> (Vec3, Vec3, Mat4, f32, f32) {
    let (center, radius) = world_extent(gpu, world_from_model);
    let to_light = direction_to_light.normalize_or(Vec3::Y);
    let forward = -to_light;
    let eye = center + to_light * radius;
    let mut near = f32::MAX;
    let mut far = f32::MIN;
    for corner in world_corners(gpu, world_from_model) {
        let depth = (corner - eye).dot(forward);
        near = near.min(depth);
        far = far.max(depth);
    }
    near = (near - radius * 0.05).max(0.05);
    far = (far + radius * 0.05).max(near + 1.0);
    let up = if forward.dot(Vec3::Y).abs() > 0.95 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let view = Mat4::look_at_rh(eye, eye + forward, up);
    let proj = Mat4::orthographic_rh(-radius, radius, -radius, radius, near, far);
    (eye, forward, proj * view, near, far)
}

#[cfg(test)]
mod dom_layout {
    use super::{DOM_LAYERS, DOM_SIZE};

    #[test]
    fn shader_constants_match_the_buffer_layout() {
        let layers = format!("const DOM_LAYERS: u32 = {DOM_LAYERS}u;");
        let size = format!("const DOM_SIZE: u32 = {DOM_SIZE}u;");
        let common = include_str!("shaders/common.wgsl");
        let clear = include_str!("shaders/clear_dom.wgsl");
        assert!(
            common.contains(&layers),
            "{layers} missing from common.wgsl"
        );
        assert!(
            clear.contains(&layers),
            "{layers} missing from clear_dom.wgsl"
        );
        assert!(clear.contains(&size), "{size} missing from clear_dom.wgsl");
    }
}

fn depth_is_reverse(clip: Mat4, camera: Vec3, forward: Vec3) -> bool {
    let near = clip_z(clip, camera + forward * 0.5);
    let far = clip_z(clip, camera + forward * 30.0);
    near > far
}

fn clip_z(clip: Mat4, point: Vec3) -> f32 {
    let c = clip * point.extend(1.0);
    c.z / c.w.max(1e-5)
}

fn ensure_targets(device: &RenderDevice, frame: &mut FrameGpu, width: u32, height: u32) {
    if frame
        .targets
        .as_ref()
        .is_some_and(|t| t.width == width && t.height == height)
    {
        return;
    }
    let pixels = width as u64 * height as u64;
    frame.targets = Some(ScreenTargets {
        width,
        height,
        center: zeros(device, "hair_center", pixels * 8, BufferKind::Storage),
        conservative: zeros(device, "hair_cons", pixels * 8, BufferKind::Storage),
        beta: zeros(device, "hair_beta", pixels * 4, BufferKind::Storage),
        shaded: zeros(device, "hair_shaded", pixels * 16, BufferKind::Storage),
        shaded_depth: zeros(device, "hair_shaded_z", pixels * 4, BufferKind::Storage),
        filtered: zeros(device, "hair_filtered", pixels * 16, BufferKind::Storage),
        filtered_depth: zeros(device, "hair_filtered_z", pixels * 4, BufferKind::Storage),
        dom: zeros(
            device,
            "hair_dom",
            DOM_SIZE as u64 * DOM_SIZE as u64 * (1 + DOM_LAYERS as u64) * 4,
            BufferKind::Storage,
        ),
    });
}

fn ensure_shade_target(device: &RenderDevice, frame: &mut FrameGpu, width: u32, height: u32) {
    if frame
        .shade_target
        .as_ref()
        .is_some_and(|target| target.width == width && target.height == height)
    {
        return;
    }
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("hair_shade_discard"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    frame.shade_target = Some(ShadeTarget {
        width,
        height,
        _texture: texture,
        view,
    });
}

fn ensure_dummy_depth(device: &RenderDevice, ctx: &mut RenderContext, frame: &mut FrameGpu) {
    if frame.dummy_depth.is_some() {
        return;
    }
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("hair_dummy_depth"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Depth32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    {
        let _pass = ctx
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("hair_dummy_depth_clear"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(0.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
    }
    frame.dummy_depth = Some(view);
    let _ = texture;
}

fn ensure_composite(
    cache: &PipelineCache,
    pipelines: &HairPipelines,
    state: &mut HairPassState,
    format: TextureFormat,
    depth_format: TextureFormat,
    reverse: bool,
) -> CachedRenderPipelineId {
    if state.composite_key != Some((format, depth_format)) {
        state.composite_greater = Some(queue_composite(
            cache,
            pipelines.composite_shader.clone(),
            &pipelines.composite_layout,
            format,
            depth_format,
            CompareFunction::GreaterEqual,
        ));
        state.composite_less = Some(queue_composite(
            cache,
            pipelines.composite_shader.clone(),
            &pipelines.composite_layout,
            format,
            depth_format,
            CompareFunction::LessEqual,
        ));
        state.composite_key = Some((format, depth_format));
    }
    if reverse {
        state.composite_greater.unwrap()
    } else {
        state.composite_less.unwrap()
    }
}

fn dispatch<const N: usize>(
    ctx: &mut RenderContext,
    cache: &PipelineCache,
    pipeline: &ComputePipeline,
    layout: &BindGroupLayoutDescriptor,
    entries: BindGroupEntries<'_, N>,
    groups: (u32, u32, u32),
) {
    let group =
        ctx.render_device()
            .create_bind_group(None, &cache.get_bind_group_layout(layout), &entries);
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("hair"),
            ..default()
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(groups.0, groups.1, groups.2);
}

struct RasterBuffers<'a> {
    gpu: &'a GpuGroom,
    params: &'a Buffer,
    refs: &'a Buffer,
    targets: &'a ScreenTargets,
    depth_view: &'a TextureView,
    indirect: &'a Buffer,
}

struct ShadeBuffers<'a> {
    params: &'a Buffer,
    targets: &'a ScreenTargets,
}

struct ShadeDraw<'a> {
    pipeline: &'a RenderPipeline,
    layout: &'a BindGroupLayoutDescriptor,
    mesh_view: &'a MeshViewBindGroup,
    environment_map: bool,
    shade_view: &'a TextureView,
    buffers: ShadeBuffers<'a>,
}

fn draw_shade(ctx: &mut RenderContext, cache: &PipelineCache, draw: ShadeDraw<'_>) {
    let ShadeDraw {
        pipeline,
        layout,
        mesh_view,
        environment_map,
        shade_view,
        buffers: ShadeBuffers { params, targets },
    } = draw;
    let hair = ctx.render_device().create_bind_group(
        None,
        &cache.get_bind_group_layout(layout),
        &BindGroupEntries::sequential((
            params.as_entire_binding(),
            targets.center.as_entire_binding(),
            targets.beta.as_entire_binding(),
            targets.shaded.as_entire_binding(),
            targets.shaded_depth.as_entire_binding(),
            targets.dom.as_entire_binding(),
        )),
    );
    let color_attachment = [Some(RenderPassColorAttachment {
        view: shade_view,
        depth_slice: None,
        resolve_target: None,
        ops: Operations {
            load: LoadOp::Clear(Default::default()),
            store: StoreOp::Discard,
        },
    })];
    let mut pass = ctx
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("hair_shade"),
            color_attachments: &color_attachment,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &mesh_view.main, &mesh_view.main_offsets);
    if environment_map {
        pass.set_bind_group(1, &mesh_view.binding_array, &[]);
    } else {
        pass.set_bind_group(1, &mesh_view.empty, &[]);
    }
    pass.set_bind_group(2, &hair, &[]);
    pass.draw(0..3, 0..1);
}

fn dispatch_raster(
    ctx: &mut RenderContext,
    cache: &PipelineCache,
    pipeline: &ComputePipeline,
    layout: &BindGroupLayoutDescriptor,
    buffers: RasterBuffers<'_>,
) {
    let RasterBuffers {
        gpu,
        params,
        refs,
        targets,
        depth_view,
        indirect,
    } = buffers;
    let group = ctx.render_device().create_bind_group(
        None,
        &cache.get_bind_group_layout(layout),
        &BindGroupEntries::sequential((
            params.as_entire_binding(),
            gpu.bundles.as_entire_binding(),
            gpu.layers.as_entire_binding(),
            gpu.style.as_entire_binding(),
            refs.as_entire_binding(),
            targets.center.as_entire_binding(),
            targets.conservative.as_entire_binding(),
            targets.beta.as_entire_binding(),
            targets.dom.as_entire_binding(),
            depth_view,
        )),
    );
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("hair_raster"),
            ..default()
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups_indirect(indirect, 0);
}
