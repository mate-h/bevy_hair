//! Minimal Wavefront OBJ loader for the example head mesh.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::math::Vec3;

pub fn load_obj_path(path: impl AsRef<std::path::Path>) -> Result<Mesh, std::io::Error> {
    let text = std::fs::read_to_string(path)?;
    Ok(parse_obj(&text))
}

pub fn parse_obj(text: &str) -> Mesh {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("v ") {
            let mut parts = rest.split_whitespace();
            let x: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let y: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let z: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            positions.push([x, y, z]);
        } else if let Some(rest) = line.strip_prefix("f ") {
            let verts: Vec<u32> = rest
                .split_whitespace()
                .filter_map(|corner| {
                    let index = corner.split('/').next()?;
                    let parsed: i32 = index.parse().ok()?;
                    if parsed > 0 {
                        Some((parsed as u32) - 1)
                    } else if parsed < 0 {
                        Some((positions.len() as i32 + parsed) as u32)
                    } else {
                        None
                    }
                })
                .collect();
            if verts.len() >= 3 {
                for i in 1..verts.len() - 1 {
                    indices.push(verts[0]);
                    indices.push(verts[i]);
                    indices.push(verts[i + 1]);
                }
            }
        }
    }

    let mut normals = vec![[0.0, 0.0, 0.0]; positions.len()];
    for tri in indices.chunks_exact(3) {
        let a = Vec3::from(positions[tri[0] as usize]);
        let b = Vec3::from(positions[tri[1] as usize]);
        let c = Vec3::from(positions[tri[2] as usize]);
        let n = (b - a).cross(c - a);
        for index in tri {
            let slot = &mut normals[*index as usize];
            slot[0] += n.x;
            slot[1] += n.y;
            slot[2] += n.z;
        }
    }
    for normal in &mut normals {
        let v = Vec3::from(*normal);
        let v = if v.length_squared() > 1e-12 {
            v.normalize()
        } else {
            Vec3::Y
        };
        *normal = v.to_array();
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
