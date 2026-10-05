//! Bake Cem Yuksel's woman grooms into `target/groom-cache/`.
//!
//! ```text
//! cargo run --example bake
//! ```
//!
//! A cached mesh is reused while the hair file's length and modification time
//! match and [`bevy_hair::BAKE_FINGERPRINT`] is unchanged. The groom example
//! includes this file and loads the same cache.

use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::Instant;

use bevy::app::App;
use bevy::log::{info, warn, LogPlugin};
use bevy::math::Vec3;
use bevy_hair::{
    BAKE_FINGERPRINT, BundleDesc, CageCorner, HairMesh, Scalp, bake_hair_mesh_with_scalp,
    load_hair_path, load_obj_path,
};

pub const GROOM_FILES: [&str; 3] = ["wStraight.hair", "wWavy.hair", "wCurly.hair"];

static LOGGING: Once = Once::new();

pub fn ensure_logging() {
    LOGGING.call_once(|| {
        App::new().add_plugins(LogPlugin::default());
    });
}

// Entry point for `cargo run --example bake`. The groom example includes this
// file as a module and does not call it.
#[allow(dead_code)]
fn main() {
    let scalp = scalp_from_asset();
    for file in GROOM_FILES {
        load(&asset_path(file), &scalp);
    }
}

pub fn scalp_from_asset() -> Scalp {
    let path = asset_path("woman.obj");
    let mesh = load_obj_path(&path).unwrap_or_else(|err| {
        panic!("failed to read {}: {err}", path.display());
    });
    Scalp::from_mesh(&mesh)
}

pub fn asset_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets/hair")
        .join(name)
}

pub fn load(path: &Path, scalp: &Scalp) -> HairMesh {
    ensure_logging();
    info!("loading {path:?}");
    if let Some(mesh) = read_cache(path) {
        info!(
            "cached {} bundles / {} strands",
            mesh.bundles.len(),
            mesh.strand_count()
        );
        return mesh;
    }

    let started = Instant::now();
    let strands = load_hair_path(path).unwrap_or_else(|err| {
        panic!("failed to read {}: {err}", path.display());
    });
    info!(
        "{} strands, {} points",
        strands.strands.len(),
        strands.points.len()
    );
    let mesh = bake_hair_mesh_with_scalp(&strands, Some(scalp));
    info!(
        "baked {} bundles / {} strands in {:.1}s",
        mesh.bundles.len(),
        mesh.strand_count(),
        started.elapsed().as_secs_f32()
    );
    if let Err(err) = write_cache(path, &mesh) {
        warn!("cache write failed: {err}");
    }
    mesh
}

const CACHE_MAGIC: &[u8; 8] = b"HAIRBAK1";

struct SourceStamp {
    len: u64,
    secs: u64,
    nanos: u32,
    scalp_len: u64,
    scalp_secs: u64,
    scalp_nanos: u32,
}

fn cache_path(source: &Path) -> PathBuf {
    let name = source.file_name().unwrap_or_default();
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/groom-cache")
        .join(name)
        .with_extension("bake")
}

fn file_stamp(path: &Path) -> Option<(u64, u64, u32)> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta.modified().ok()?;
    let elapsed = modified.duration_since(std::time::UNIX_EPOCH).ok()?;
    Some((meta.len(), elapsed.as_secs(), elapsed.subsec_nanos()))
}

fn source_stamp(path: &Path) -> Option<SourceStamp> {
    let (len, secs, nanos) = file_stamp(path)?;
    let (scalp_len, scalp_secs, scalp_nanos) = file_stamp(&asset_path("woman.obj"))?;
    Some(SourceStamp {
        len,
        secs,
        nanos,
        scalp_len,
        scalp_secs,
        scalp_nanos,
    })
}

fn read_cache(source: &Path) -> Option<HairMesh> {
    let stamp = source_stamp(source)?;
    let bytes = std::fs::read(cache_path(source)).ok()?;
    decode(&bytes, &stamp)
}

fn write_cache(source: &Path, mesh: &HairMesh) -> std::io::Result<()> {
    let stamp = source_stamp(source).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "hair file metadata is unavailable",
        )
    })?;
    let path = cache_path(source);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("bake.tmp");
    std::fs::write(&tmp, encode(&stamp, mesh))?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

fn encode(stamp: &SourceStamp, mesh: &HairMesh) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        72 + mesh.bundles.len() * 12 + mesh.corners.len() * 28 + mesh.style.len() * 16,
    );
    out.extend_from_slice(CACHE_MAGIC);
    push_u64(&mut out, BAKE_FINGERPRINT);
    push_u64(&mut out, stamp.len);
    push_u64(&mut out, stamp.secs);
    push_u32(&mut out, stamp.nanos);
    push_u64(&mut out, stamp.scalp_len);
    push_u64(&mut out, stamp.scalp_secs);
    push_u32(&mut out, stamp.scalp_nanos);
    push_u32(&mut out, mesh.bundles.len() as u32);
    push_u32(&mut out, mesh.corners.len() as u32);
    push_u32(&mut out, mesh.style.len() as u32);
    push_vec3(&mut out, mesh.bounds_min);
    push_vec3(&mut out, mesh.bounds_max);
    for bundle in &mesh.bundles {
        push_u32(&mut out, bundle.layer_offset);
        push_u32(&mut out, bundle.style_offset);
        push_u32(&mut out, bundle.strand_count);
    }
    for corner in &mesh.corners {
        push_vec3(&mut out, corner.position);
        push_vec3(&mut out, corner.tangent);
        push_f32(&mut out, corner.ao);
    }
    for texel in &mesh.style {
        for channel in texel {
            push_f32(&mut out, *channel);
        }
    }
    out
}

fn decode(bytes: &[u8], stamp: &SourceStamp) -> Option<HairMesh> {
    let mut cursor = Cursor { bytes, at: 0 };
    let magic = cursor.bytes(8)?;
    if magic != CACHE_MAGIC {
        return None;
    }
    if cursor.u64()? != BAKE_FINGERPRINT
        || cursor.u64()? != stamp.len
        || cursor.u64()? != stamp.secs
        || cursor.u32()? != stamp.nanos
        || cursor.u64()? != stamp.scalp_len
        || cursor.u64()? != stamp.scalp_secs
        || cursor.u32()? != stamp.scalp_nanos
    {
        return None;
    }
    let bundle_count = cursor.u32()? as usize;
    let corner_count = cursor.u32()? as usize;
    let style_count = cursor.u32()? as usize;
    let bounds_min = cursor.vec3()?;
    let bounds_max = cursor.vec3()?;
    let payload = bundle_count * 12 + corner_count * 28 + style_count * 16;
    if bytes.len() - cursor.at != payload {
        return None;
    }

    let mut bundles = Vec::with_capacity(bundle_count);
    for _ in 0..bundle_count {
        bundles.push(BundleDesc {
            layer_offset: cursor.u32()?,
            style_offset: cursor.u32()?,
            strand_count: cursor.u32()?,
        });
    }
    let mut corners = Vec::with_capacity(corner_count);
    for _ in 0..corner_count {
        corners.push(CageCorner {
            position: cursor.vec3()?,
            tangent: cursor.vec3()?,
            ao: cursor.f32()?,
        });
    }
    let mut style = Vec::with_capacity(style_count);
    for _ in 0..style_count {
        style.push([cursor.f32()?, cursor.f32()?, cursor.f32()?, cursor.f32()?]);
    }
    Some(HairMesh {
        bundles,
        corners,
        style,
        bounds_min,
        bounds_max,
    })
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(out: &mut Vec<u8>, value: f32) {
    push_u32(out, value.to_bits());
}

fn push_vec3(out: &mut Vec<u8>, value: Vec3) {
    push_f32(out, value.x);
    push_f32(out, value.y);
    push_f32(out, value.z);
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn bytes(&mut self, len: usize) -> Option<&[u8]> {
        let end = self.at.checked_add(len)?;
        let slice = self.bytes.get(self.at..end)?;
        self.at = end;
        Some(slice)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.bytes(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.bytes(8)?.try_into().ok()?))
    }

    fn f32(&mut self) -> Option<f32> {
        Some(f32::from_bits(self.u32()?))
    }

    fn vec3(&mut self) -> Option<Vec3> {
        Some(Vec3::new(self.f32()?, self.f32()?, self.f32()?))
    }
}
