use bevy::prelude::*;
use std::num::NonZeroU64;

use bevy::pbr::{MeshPipeline, MeshPipelineViewLayoutKey, MeshPipelineViewLayouts};
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, texture_depth_2d, uniform_buffer_sized,
};
use bevy::render::render_resource::*;
const COMMON: &str = include_str!("shaders/common.wgsl");

/// `HairParams` uniform size. Keep in sync with `shaders/common.wgsl`.
pub const HAIR_PARAMS_SIZE: u64 = 432;

#[derive(Resource, Clone)]
pub struct HairShaderHandles {
    pub clear: Handle<Shader>,
    pub clear_dom: Handle<Shader>,
    pub lod: Handle<Shader>,
    pub scan: Handle<Shader>,
    pub raster: Handle<Shader>,
    pub shade: Handle<Shader>,
    pub filter: Handle<Shader>,
    pub composite: Handle<Shader>,
}

pub fn load_shaders(shaders: &mut Assets<Shader>) -> HairShaderHandles {
    HairShaderHandles {
        clear: shader(
            shaders,
            "hair_clear.wgsl",
            include_str!("shaders/clear.wgsl"),
            false,
        ),
        clear_dom: shader(
            shaders,
            "hair_clear_dom.wgsl",
            include_str!("shaders/clear_dom.wgsl"),
            false,
        ),
        lod: shader(
            shaders,
            "hair_lod.wgsl",
            include_str!("shaders/lod.wgsl"),
            true,
        ),
        scan: shader(
            shaders,
            "hair_scan.wgsl",
            include_str!("shaders/scan.wgsl"),
            true,
        ),
        raster: shader(
            shaders,
            "hair_raster.wgsl",
            include_str!("shaders/raster.wgsl"),
            true,
        ),
        shade: shaders.add(Shader::from_wgsl(shade_source(), "hair_shade.wgsl")),
        filter: shader(
            shaders,
            "hair_filter.wgsl",
            include_str!("shaders/filter.wgsl"),
            true,
        ),
        composite: shader(
            shaders,
            "hair_composite.wgsl",
            include_str!("shaders/composite.wgsl"),
            true,
        ),
    }
}

#[derive(Resource)]
pub struct HairPipelines {
    pub clear: CachedComputePipelineId,
    pub clear_layout: BindGroupLayoutDescriptor,
    pub clear_dom: CachedComputePipelineId,
    pub clear_dom_layout: BindGroupLayoutDescriptor,
    pub lod: CachedComputePipelineId,
    pub lod_layout: BindGroupLayoutDescriptor,
    pub scan: CachedComputePipelineId,
    pub scan_layout: BindGroupLayoutDescriptor,
    pub raster: CachedComputePipelineId,
    pub raster_layout: BindGroupLayoutDescriptor,
    pub filter: CachedComputePipelineId,
    pub filter_layout: BindGroupLayoutDescriptor,
    pub composite_layout: BindGroupLayoutDescriptor,
    pub composite_shader: Handle<Shader>,
}

pub fn init_pipelines(
    mut commands: Commands,
    shader_handles: Res<HairShaderHandles>,
    pipeline_cache: Res<PipelineCache>,
    mesh_view_layouts: Res<MeshPipelineViewLayouts>,
    mesh_pipeline: Res<MeshPipeline>,
) {
    let HairShaderHandles {
        clear,
        clear_dom,
        lod,
        scan,
        raster,
        shade,
        filter,
        composite,
    } = shader_handles.clone();

    let clear_layout = BindGroupLayoutDescriptor::new(
        "hair_clear",
        &BindGroupLayoutEntries::sequential(ShaderStages::COMPUTE, (rw(8), rw(8), rw(4))),
    );
    let clear_dom_layout = BindGroupLayoutDescriptor::new(
        "hair_clear_dom",
        &BindGroupLayoutEntries::sequential(ShaderStages::COMPUTE, (rw(4),)),
    );
    let lod_layout = BindGroupLayoutDescriptor::new(
        "hair_lod",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (params_uniform(), ro(16), ro(32), rw(16)),
        ),
    );
    let scan_layout = BindGroupLayoutDescriptor::new(
        "hair_scan",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (params_uniform(), ro(16), rw(16), rw(4)),
        ),
    );
    let raster_layout = BindGroupLayoutDescriptor::new(
        "hair_raster",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                params_uniform(),
                ro(16),
                ro(32),
                ro(16),
                ro(16),
                rw(8),
                rw(8),
                rw(4),
                rw(4),
                texture_depth_2d(),
            ),
        ),
    );
    let shade_layout = BindGroupLayoutDescriptor::new(
        "hair_shade",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (params_uniform(), ro(8), ro(4), rw(16), rw(4), ro(4)),
        ),
    );
    let filter_layout = BindGroupLayoutDescriptor::new(
        "hair_filter",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (params_uniform(), ro(8), ro(8), ro(16), ro(4), rw(16), rw(4)),
        ),
    );
    let composite_layout = BindGroupLayoutDescriptor::new(
        "hair_composite",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (params_uniform(), ro(16), ro(4)),
        ),
    );

    let queue =
        |label: &'static str, layout: &BindGroupLayoutDescriptor, shader: &Handle<Shader>| {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(label.into()),
                layout: vec![layout.clone()],
                shader: shader.clone(),
                ..default()
            })
        };

    let pipelines = HairPipelines {
        clear: queue("hair_clear", &clear_layout, &clear),
        clear_layout,
        clear_dom: queue("hair_clear_dom", &clear_dom_layout, &clear_dom),
        clear_dom_layout,
        lod: queue("hair_lod", &lod_layout, &lod),
        lod_layout,
        scan: queue("hair_scan", &scan_layout, &scan),
        scan_layout,
        raster: queue("hair_raster", &raster_layout, &raster),
        raster_layout,
        filter: queue("hair_filter", &filter_layout, &filter),
        filter_layout,
        composite_layout,
        composite_shader: composite,
    };
    commands.insert_resource(pipelines);
    commands.insert_resource(HairShadePipeline {
        mesh_view_layouts: mesh_view_layouts.clone(),
        binding_arrays_are_usable: mesh_pipeline.binding_arrays_are_usable,
        shader: shade,
        layout: shade_layout,
    });
    commands.init_resource::<SpecializedRenderPipelines<HairShadePipeline>>();
}

/// Fullscreen shade pass. Group 0 and 1 are the mesh view layouts so the
/// fragment shader can bind [`bevy::pbr::MeshViewBindGroup`].
#[derive(Resource)]
pub struct HairShadePipeline {
    mesh_view_layouts: MeshPipelineViewLayouts,
    binding_arrays_are_usable: bool,
    shader: Handle<Shader>,
    pub(crate) layout: BindGroupLayoutDescriptor,
}

impl SpecializedRenderPipeline for HairShadePipeline {
    type Key = MeshPipelineViewLayoutKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = vec!["SHADOW_FILTER_METHOD_HARDWARE_2X2".into()];
        if key.contains(MeshPipelineViewLayoutKey::ENVIRONMENT_MAP) {
            shader_defs.push("ENVIRONMENT_MAP".into());
        }
        if key.contains(MeshPipelineViewLayoutKey::AREA_LIGHT_LUTS) {
            shader_defs.push("AREA_LIGHT_LUTS".into());
        }
        if self.binding_arrays_are_usable {
            shader_defs.push("MULTIPLE_LIGHT_PROBES_IN_ARRAY".into());
        }

        let view_layout = self.mesh_view_layouts.get_view_layout(key);
        let probe_layout = if key.contains(MeshPipelineViewLayoutKey::ENVIRONMENT_MAP) {
            view_layout.binding_array_layout
        } else {
            view_layout.empty_layout
        };

        RenderPipelineDescriptor {
            label: Some("hair_shade".into()),
            layout: vec![view_layout.main_layout, probe_layout, self.layout.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                entry_point: Some("vs".into()),
                buffers: Vec::new(),
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                entry_point: Some("fs".into()),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            ..default()
        }
    }
}

fn shader(shaders: &mut Assets<Shader>, name: &str, body: &str, common: bool) -> Handle<Shader> {
    let source = if common {
        format!("{COMMON}\n{body}")
    } else {
        body.to_string()
    };
    shaders.add(Shader::from_wgsl(source, name))
}

fn params_uniform() -> bevy::render::render_resource::BindGroupLayoutEntryBuilder {
    uniform_buffer_sized(false, NonZeroU64::new(HAIR_PARAMS_SIZE))
}

fn shade_source() -> String {
    let body = include_str!("shaders/shade.wgsl");
    let (imports, rest) = body
        .split_once("@group")
        .expect("shade.wgsl bindings follow the imports");
    // `HairParams` has to exist before the bindings, and the `#import` block has to stay first.
    format!("{imports}\n{COMMON}\n@group{rest}")
}

fn ro(size: u64) -> bevy::render::render_resource::BindGroupLayoutEntryBuilder {
    storage_buffer_read_only_sized(false, NonZeroU64::new(size))
}

fn rw(size: u64) -> bevy::render::render_resource::BindGroupLayoutEntryBuilder {
    storage_buffer_sized(false, NonZeroU64::new(size))
}

pub fn queue_composite(
    cache: &PipelineCache,
    shader: Handle<Shader>,
    layout: &BindGroupLayoutDescriptor,
    format: TextureFormat,
    depth_format: TextureFormat,
    compare: CompareFunction,
) -> CachedRenderPipelineId {
    cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("hair_composite".into()),
        layout: vec![layout.clone()],
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vs".into()),
            shader_defs: Vec::new(),
            buffers: Vec::new(),
        },
        fragment: Some(FragmentState {
            shader,
            entry_point: Some("fs".into()),
            shader_defs: Vec::new(),
            targets: vec![Some(ColorTargetState {
                format,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: Some(DepthStencilState {
            format: depth_format,
            depth_write_enabled: Some(true),
            depth_compare: Some(compare),
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState::default(),
        ..default()
    })
}

#[cfg(test)]
mod shader_parse {
    fn parse(name: &str, source: &str) {
        naga::front::wgsl::parse_str(source).unwrap_or_else(|err| {
            panic!("{name} failed to parse:\n{err:?}\n{err}");
        });
    }

    fn with_common(body: &str) -> String {
        format!("{}\n{body}", include_str!("shaders/common.wgsl"))
    }

    #[test]
    fn shaders_parse() {
        parse("clear", include_str!("shaders/clear.wgsl"));
        parse("clear_dom", include_str!("shaders/clear_dom.wgsl"));
        parse("lod", &with_common(include_str!("shaders/lod.wgsl")));
        parse("scan", &with_common(include_str!("shaders/scan.wgsl")));
        parse("raster", &with_common(include_str!("shaders/raster.wgsl")));
        parse("filter", &with_common(include_str!("shaders/filter.wgsl")));
        parse(
            "composite",
            &with_common(include_str!("shaders/composite.wgsl")),
        );
    }
}
