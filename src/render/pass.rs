use std::collections::HashMap;

use bevy::asset::AssetId;
use bevy::ecs::system::SystemParam;
use bevy::math::{Mat4, Vec2, Vec3, Vec4};
use bevy::pbr::{MeshPipelineViewLayoutKey, MeshViewBindGroup, ViewKeyCache};
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::storage::GpuShaderBuffer;
use bevy::render::view::{ViewDepthStencilTexture, ViewTarget};
use bytemuck::{Pod, Zeroable};

use crate::mesh::{HairMesh, LAYER_COUNT};
use crate::render::pipeline::{
    HAIR_PARAMS_SIZE, HairPipelines, HairShadePipeline, queue_composite,
};
use crate::render::{ExtractedDirectional, ExtractedFrame, ExtractedGroom, flag};
use crate::shadow::{HairShadow, MAX_HAIR_SHADOW_CASTERS};

pub const DOM_SIZE: u32 = 512;
/// Opacity slices per light-space texel. Keep in sync with `DOM_LAYERS` in the shaders.
pub const DOM_LAYERS: u32 = 16;
/// Beer-Lambert extinction per strand fragment. Visibility is `exp(-hits * DOM_SIGMA)`.
const DOM_SIGMA: f32 = 0.05;
/// Front depth texels, then `DOM_LAYERS` opacity counts per texel.
pub const DOM_BYTES: u64 = (DOM_SIZE * DOM_SIZE * (1 + DOM_LAYERS)) as u64 * 4;
pub const SHADOW_LIGHT_BYTES: u64 = size_of::<ShadowLight>() as u64;

/// Shadow slots, one per `HairShadowCaster`.
pub const SHADOW_CHANNELS: usize = MAX_HAIR_SHADOW_CASTERS;
/// One deep opacity map per caster, plus the key light's when it is not one.
const DOM_VIEWS: usize = SHADOW_CHANNELS + 1;

/// A deep opacity map's light view for `hair_shadow.wesl`, one per caster slot.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct ShadowLight {
    light_clip_from_world: [[f32; 4]; 4],
    light_eye: [f32; 4],
    light_forward: [f32; 4],
    // x near, y far, z extinction, w 1 once a map exists.
    depth: [f32; 4],
    // Identifies the Bevy light: xyz direction to a directional light (w 1)
    // or a point or spot light's position (w 2).
    source: [f32; 4],
}

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
    rect: [u32; 4],
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
    /// Per light view, since every map's parameters are written before the
    /// frame's commands run.
    pub dom_params: Vec<DomParams>,
}

pub struct DomParams {
    /// Pass 1: nearest strand depth.
    pub depth: Buffer,
    /// Pass 2: opacity slices behind it.
    pub opacity: Buffer,
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
            dom_params: (0..DOM_VIEWS)
                .map(|_| DomParams {
                    depth: zeros(
                        device,
                        "hair_dom_params",
                        HAIR_PARAMS_SIZE,
                        BufferKind::Uniform,
                    ),
                    opacity: zeros(
                        device,
                        "hair_dom_opacity_params",
                        HAIR_PARAMS_SIZE,
                        BufferKind::Uniform,
                    ),
                })
                .collect(),
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
    /// Storage that is also copied out, like the deep opacity map.
    SharedStorage,
    Indirect,
    Uniform,
}

fn zeros(device: &RenderDevice, label: &str, size: u64, kind: BufferKind) -> Buffer {
    let mut usage = BufferUsages::COPY_DST;
    match kind {
        BufferKind::Storage => usage |= BufferUsages::STORAGE,
        BufferKind::SharedStorage => usage |= BufferUsages::STORAGE | BufferUsages::COPY_SRC,
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
    /// Staging for `HairShadow::light`. Copied in command order so receivers
    /// read the light view that matches the copied map.
    shadow_light: Option<Buffer>,
    /// The light views and groom placement the deep opacity maps were last
    /// rendered for. While they hold, the maps are reused.
    dom_signature: Option<Vec<f32>>,
    /// The screen rectangle the last groom wrote, as `[x0, y0, x1, y1)`.
    written_rect: Option<[u32; 4]>,
    shadow_lights: [ShadowLight; SHADOW_CHANNELS],
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
    shadow: Option<Res<'w, HairShadow>>,
    shader_buffers: Res<'w, RenderAssets<GpuShaderBuffer>>,
}

pub fn hair_pass(
    view: ViewQuery<(
        &bevy::render::view::ExtractedView,
        &ViewTarget,
        &ViewDepthStencilTexture,
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
        shadow,
        shader_buffers,
    } = hair;
    // Materials that receive hair shadows read these between hair passes.
    let shadow_dom = shadow
        .as_ref()
        .and_then(|shadow| shader_buffers.get(&shadow.dom));
    let shadow_light_dst = shadow
        .as_ref()
        .and_then(|shadow| shader_buffers.get(&shadow.light));
    let (view, target, depth, mesh_view) = view.into_inner();
    let Some(mesh_key) = view_keys.get(&view.retained_view_entity) else {
        return;
    };
    let layout_key = MeshPipelineViewLayoutKey::from(*mesh_key);
    let shade_id = shade_pipelines.specialize(&cache, &shade_pipeline, layout_key);
    let Some(shade) = cache.get_render_pipeline(shade_id) else {
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
    let shadow_light_src = state
        .frame
        .shadow_light
        .get_or_insert_with(|| {
            device.create_buffer(&BufferDescriptor {
                label: Some("hair_shadow_light"),
                size: SHADOW_LIGHT_BYTES * SHADOW_CHANNELS as u64,
                usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .clone();
    let views = light_views(&extracted);
    let dom_signature = shadow_dom.and(dom_signature(&extracted, &state.grooms, &views));
    let reuse_doms = dom_signature.is_some() && dom_signature == state.frame.dom_signature;
    let mut shadow_lights = if reuse_doms {
        state.frame.shadow_lights
    } else {
        [ShadowLight::zeroed(); SHADOW_CHANNELS]
    };
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

    let depth_format = depth.texture().format();
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

    let depth_view = depth
        .attachment
        .depth_stencil_views()
        .depth_only_view()
        .filter(|_| {
            depth
                .texture()
                .usage()
                .contains(TextureUsages::TEXTURE_BINDING)
        })
        .unwrap_or_else(|| state.frame.dummy_depth.as_ref().unwrap());

    let targets = state.frame.targets.as_ref().unwrap();
    let mut written = state.frame.written_rect;

    for groom in &extracted.grooms {
        let Some(gpu) = state.grooms.get(&groom.asset) else {
            continue;
        };
        // Every per-pixel pass covers only the groom's screen rectangle. The
        // G-buffer is cleared over the rows of that rectangle and of the last
        // one written, so everything outside stays empty.
        let rect = screen_rect(gpu, groom.world_from_model, clip_from_world, width, height);
        let rows = written.map_or([rect[1], rect[3]], |last| {
            [rect[1].min(last[1]), rect[3].max(last[3])]
        });
        written = Some(rect);
        for (buffer, texel) in [
            (&targets.center, 8),
            (&targets.conservative, 8),
            (&targets.beta, 4),
        ] {
            let row = u64::from(width) * texel;
            // Empty is zero in all three (see `unpack_gb`), so a blit fill clears them.
            ctx.command_encoder().clear_buffer(
                buffer,
                u64::from(rows[0]) * row,
                Some(u64::from(rows[1] - rows[0]) * row),
            );
        }
        let key = views.last().expect("the key view is always present");
        let Some(LightFit {
            eye: light_eye,
            forward: light_dir,
            clip: light_clip,
            near: light_near,
            far: light_far,
        }) = key.source.fit(gpu, groom.world_from_model)
        else {
            continue;
        };
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
            rect,
        });
        queue.write_buffer(&gpu.camera_params, 0, bytemuck::bytes_of(&camera_params));

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

        if groom.deep_opacity && !reuse_doms {
            let dom = DomPasses {
                clear_dom,
                lod,
                scan,
                raster,
                pipelines: &pipelines,
                targets,
                depth_view,
                queue: &queue,
                reverse,
            };
            // The key's map renders last so shading reads it from `targets.dom`.
            for (slot, view) in views.iter().enumerate() {
                let Some(fit) = view.source.fit(gpu, groom.world_from_model) else {
                    continue;
                };
                render_dom(
                    &mut ctx,
                    &cache,
                    &dom,
                    gpu,
                    groom,
                    &gpu.dom_params[slot],
                    &fit,
                );
                let Some(channel) = view.channel else {
                    continue;
                };
                if let Some(dst) = shadow_dom {
                    ctx.command_encoder().copy_buffer_to_buffer(
                        &targets.dom,
                        0,
                        &dst.buffer,
                        channel as u64 * DOM_BYTES,
                        DOM_BYTES,
                    );
                    shadow_lights[channel as usize] = ShadowLight {
                        light_clip_from_world: fit.clip.to_cols_array_2d(),
                        light_eye: fit.eye.extend(1.0).to_array(),
                        light_forward: fit.forward.extend(0.0).to_array(),
                        depth: [fit.near, fit.far, DOM_SIGMA, 1.0],
                        source: match view.source {
                            LightSource::Directional(to_light) => to_light.extend(1.0),
                            LightSource::Positional(position) => position.extend(2.0),
                        }
                        .to_array(),
                    };
                }
            }
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
                rect,
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
                (
                    (rect[2] - rect[0]).div_ceil(16),
                    (rect[3] - rect[1]).div_ceil(8),
                    1,
                ),
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
        pass.set_scissor_rect(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }

    state.frame.written_rect = written;
    state.frame.dom_signature = dom_signature;
    state.frame.shadow_lights = shadow_lights;

    // The queue write lands before this frame's commands, so the copy below
    // publishes it alongside the map copied above.
    if let Some(dst) = shadow_light_dst {
        queue.write_buffer(&shadow_light_src, 0, bytemuck::cast_slice(&shadow_lights));
        ctx.command_encoder().copy_buffer_to_buffer(
            &shadow_light_src,
            0,
            &dst.buffer,
            0,
            SHADOW_LIGHT_BYTES * SHADOW_CHANNELS as u64,
        );
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
    rect: [u32; 4],
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
        rect: args.rect,
    }
}

/// Where a deep opacity map looks from.
enum LightSource {
    Directional(Vec3),
    Positional(Vec3),
}

struct LightView {
    source: LightSource,
    channel: Option<u8>,
}

/// The maps to render this frame: every channel light, then the key
/// directional, whose map also drives hair self-shadowing.
fn light_views(extracted: &ExtractedFrame) -> Vec<LightView> {
    let lights = &extracted.directionals;
    let key = key_index(lights);
    let mut views: Vec<LightView> = lights
        .iter()
        .enumerate()
        .filter(|&(index, _)| Some(index) != key)
        .filter_map(|(_, light)| {
            light.channel.map(|channel| LightView {
                source: LightSource::Directional(light.direction_to_light),
                channel: Some(channel),
            })
        })
        .chain(extracted.positionals.iter().map(|light| LightView {
            source: LightSource::Positional(light.position),
            channel: Some(light.channel),
        }))
        .collect();
    views.push(LightView {
        source: LightSource::Directional(
            key.map(|index| lights[index].direction_to_light)
                .unwrap_or_else(|| Vec3::new(0.35, 0.82, 0.45)),
        ),
        channel: key.and_then(|index| lights[index].channel),
    });
    views
}

/// What the deep opacity maps depend on: the groom's placement and every
/// light view. `None` unless exactly one groom renders maps, since grooms share
/// `targets.dom` and the key map has to survive until the next frame's shading.
fn dom_signature(
    extracted: &ExtractedFrame,
    grooms: &HashMap<AssetId<HairMesh>, GpuGroom>,
    views: &[LightView],
) -> Option<Vec<f32>> {
    let mut casting = extracted
        .grooms
        .iter()
        .filter(|groom| groom.deep_opacity && grooms.contains_key(&groom.asset));
    let groom = casting.next()?;
    if casting.next().is_some() {
        return None;
    }
    let gpu = &grooms[&groom.asset];
    let mut signature = groom.world_from_model.to_cols_array().to_vec();
    for view in views {
        signature.push(view.channel.map_or(-1.0, f32::from));
        match view.source.fit(gpu, groom.world_from_model) {
            Some(fit) => signature.extend(fit.clip.to_cols_array()),
            None => signature.push(f32::NAN),
        }
    }
    Some(signature)
}

fn key_index(lights: &[ExtractedDirectional]) -> Option<usize> {
    lights
        .iter()
        .position(|light| light.casts_shadows)
        .or((!lights.is_empty()).then_some(0))
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

struct LightFit {
    eye: Vec3,
    forward: Vec3,
    clip: Mat4,
    near: f32,
    far: f32,
}

impl LightSource {
    /// A light view that just contains the groom: orthographic for a
    /// directional, a perspective cone for a point or spot. `None` when the
    /// light is inside the groom's bounds.
    fn fit(&self, gpu: &GpuGroom, world_from_model: Mat4) -> Option<LightFit> {
        let (center, radius) = world_extent(gpu, world_from_model);
        let (eye, forward) = match *self {
            LightSource::Directional(to_light) => {
                let to_light = to_light.normalize_or(Vec3::Y);
                (center + to_light * radius, -to_light)
            }
            LightSource::Positional(position) => {
                if position.distance(center) <= radius * 1.05 {
                    return None;
                }
                (position, (center - position).normalize())
            }
        };
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
        let view = Transform::from_translation(eye)
            .looking_to(forward, up)
            .to_matrix()
            .inverse();
        let proj = match *self {
            LightSource::Directional(_) => {
                bevy::math::proj::orthographic(-radius, radius, -radius, radius, near, far)
            }
            LightSource::Positional(_) => {
                let half = (radius / eye.distance(center)).asin();
                bevy::math::proj::perspective(half * 2.0, 1.0, near, far)
            }
        };
        Some(LightFit {
            eye,
            forward,
            clip: proj * view,
            near,
            far,
        })
    }
}

struct DomPasses<'a> {
    clear_dom: &'a ComputePipeline,
    lod: &'a ComputePipeline,
    scan: &'a ComputePipeline,
    raster: &'a ComputePipeline,
    pipelines: &'a HairPipelines,
    targets: &'a ScreenTargets,
    depth_view: &'a TextureView,
    queue: &'a RenderQueue,
    reverse: bool,
}

/// Renders the groom's deep opacity map from `fit` into `targets.dom`. Depth
/// texels, then `DOM_LAYERS` opacity counts. Two rasters: the second bins
/// fragments relative to the nearest depth from the first.
fn render_dom(
    ctx: &mut RenderContext,
    cache: &PipelineCache,
    dom: &DomPasses,
    gpu: &GpuGroom,
    groom: &ExtractedGroom,
    params: &DomParams,
    fit: &LightFit,
) {
    let DomPasses {
        clear_dom,
        lod,
        scan,
        raster,
        pipelines,
        targets,
        depth_view,
        queue,
        reverse,
    } = *dom;
    let depth_params = make_params(ViewArgs {
        clip_from_world: fit.clip,
        world_from_clip: fit.clip.inverse(),
        world_from_model: groom.world_from_model,
        light_clip_from_world: fit.clip,
        camera_pos: fit.eye,
        camera_forward: fit.forward,
        screen: Vec4::new(DOM_SIZE as f32, DOM_SIZE as f32, fit.near, fit.far),
        groom,
        light_forward: fit.forward,
        dom_near: fit.near,
        dom_far: fit.far,
        dom_scale: (fit.far - fit.near) / LAYER_COUNT as f32,
        pass_mode: [1, gpu.bundle_count, u32::from(reverse), 1],
        light_eye: fit.eye,
        rect: [0, 0, DOM_SIZE, DOM_SIZE],
    });
    let mut opacity_params = depth_params;
    opacity_params.pass_mode[0] = 2;
    queue.write_buffer(&params.depth, 0, bytemuck::bytes_of(&depth_params));
    queue.write_buffer(&params.opacity, 0, bytemuck::bytes_of(&opacity_params));

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
        ctx,
        cache,
        lod,
        &pipelines.lod_layout,
        BindGroupEntries::sequential((
            params.depth.as_entire_binding(),
            gpu.bundles.as_entire_binding(),
            gpu.layers.as_entire_binding(),
            gpu.dom_lod.as_entire_binding(),
        )),
        (gpu.bundle_count.div_ceil(64), 1, 1),
    );
    dispatch(
        ctx,
        cache,
        scan,
        &pipelines.scan_layout,
        BindGroupEntries::sequential((
            params.depth.as_entire_binding(),
            gpu.dom_lod.as_entire_binding(),
            gpu.dom_refs.as_entire_binding(),
            gpu.dom_indirect.as_entire_binding(),
        )),
        (1, 1, 1),
    );
    for params in [&params.depth, &params.opacity] {
        dispatch_raster(
            ctx,
            cache,
            raster,
            &pipelines.raster_layout,
            RasterBuffers {
                gpu,
                params,
                refs: &gpu.dom_refs,
                targets,
                depth_view,
                indirect: &gpu.dom_indirect,
            },
        );
    }
}

/// Strand pixels reach past the projected center line by the diagonal fill.
const RECT_PAD: f32 = 2.0;

/// The groom's projected bounds in pixels, as `[x0, y0, x1, y1)`, never empty.
/// The whole screen when a corner is behind the camera.
fn screen_rect(
    gpu: &GpuGroom,
    world_from_model: Mat4,
    clip_from_world: Mat4,
    width: u32,
    height: u32,
) -> [u32; 4] {
    let size = Vec2::new(width as f32, height as f32);
    let mut min = Vec2::splat(f32::MAX);
    let mut max = Vec2::splat(f32::MIN);
    for corner in world_corners(gpu, world_from_model) {
        let clip = clip_from_world * corner.extend(1.0);
        if clip.w <= 1e-3 {
            return [0, 0, width, height];
        }
        let ndc = clip.truncate().truncate() / clip.w;
        let screen = Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * size;
        min = min.min(screen);
        max = max.max(screen);
    }
    let min = (min - RECT_PAD).floor().clamp(Vec2::ZERO, size - 1.0);
    let max = (max + RECT_PAD).ceil().clamp(min + 1.0, size);
    [min.x as u32, min.y as u32, max.x as u32, max.y as u32]
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
    frame.dom_signature = None;
    frame.written_rect = None;
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
        dom: zeros(device, "hair_dom", DOM_BYTES, BufferKind::SharedStorage),
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
    rect: [u32; 4],
    buffers: ShadeBuffers<'a>,
}

fn draw_shade(ctx: &mut RenderContext, cache: &PipelineCache, draw: ShadeDraw<'_>) {
    let ShadeDraw {
        pipeline,
        layout,
        mesh_view,
        environment_map,
        shade_view,
        rect,
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
    pass.set_scissor_rect(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]);
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
        let receiver = include_str!("shaders/hair_shadow.wesl");
        let channels = format!("const CHANNELS: u32 = {}u;", super::SHADOW_CHANNELS);
        assert!(
            receiver.contains(&channels),
            "{channels} missing from hair_shadow.wesl"
        );
        assert!(
            receiver.contains(&layers),
            "{layers} missing from hair_shadow.wesl"
        );
        assert!(
            receiver.contains(&size),
            "{size} missing from hair_shadow.wesl"
        );
    }
}
