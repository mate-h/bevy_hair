//! Convert explicit strands into the hair-mesh cages the rasterizer generates from.
//!
//! Roots are clustered on the scalp when a head mesh is available, otherwise in a
//! 3D grid. Each cluster becomes a quad prism whose faces contain the strands.
//! The rasterizer draws `uv = rand.next2()` uniformly inside each quad, so a quad
//! that spans empty space is split. The styling function is the mean residual from
//! the bilinear cage; a texel whose strands disagree is split as well.

use std::collections::{HashMap, VecDeque};

use bevy::math::{Mat3, Vec2, Vec3};
use bevy::mesh::{Indices, Mesh, VertexAttributeValues};

use crate::hair_file::HairStrands;
use crate::mesh::{BundleDesc, CageCorner, HairMesh, LAYER_COUNT, STYLE_TEXELS, STYLE_U, STYLE_V};

const MIN_STRANDS_PER_BUNDLE: usize = 8;
const TARGET_STRANDS_PER_CELL: f32 = 180.0;
/// Split a bundle when a sample of its quad is farther than this from every strand.
/// The uniform uv draw fills the whole quad, not just the strands.
const MAX_QUAD_GAP: f32 = 5.0;
/// Split when strands that share a style texel disagree by more than this.
const MAX_STYLE_STDDEV: f32 = 2.0;
/// Split until each member strand is this close to `f(uvw)` at its root uv.
const MAX_MEMBER_ERROR: f32 = 4.0;
/// Roots farther than this from every scalp triangle fall back to the 3D grid.
const SCALP_ASSIGN_DIST: f32 = 25.0;
const AO_RAYS: usize = 8;
const AO_REACH: f32 = 8.0;
const AO_RADIUS: f32 = 0.45;
const AO_MIN_T: f32 = 0.75;

/// Identity of this baker. On-disk groom caches should drop a mesh when it changes.
pub const BAKE_FINGERPRINT: u64 = fnv1a_64(include_str!("bake.rs").as_bytes())
    ^ (LAYER_COUNT as u64).wrapping_shl(32)
    ^ STYLE_TEXELS as u64;

const fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        i += 1;
    }
    hash
}

/// Triangle mesh the root layer is fit to. Faces are the scalp the hairs grow from.
#[derive(Clone)]
pub struct Scalp {
    pub positions: Vec<Vec3>,
    pub indices: Vec<u32>,
}

impl Scalp {
    pub fn from_mesh(mesh: &Mesh) -> Self {
        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(values)) => {
                values.iter().copied().map(Vec3::from).collect()
            }
            _ => Vec::new(),
        };
        let indices = match mesh.indices() {
            Some(Indices::U32(indices)) => indices.clone(),
            Some(Indices::U16(indices)) => indices.iter().map(|index| *index as u32).collect(),
            None => Vec::new(),
        };
        Self { positions, indices }
    }

    fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    fn triangle(&self, index: usize) -> Option<[Vec3; 3]> {
        let base = index * 3;
        let i0 = *self.indices.get(base)? as usize;
        let i1 = *self.indices.get(base + 1)? as usize;
        let i2 = *self.indices.get(base + 2)? as usize;
        Some([
            *self.positions.get(i0)?,
            *self.positions.get(i1)?,
            *self.positions.get(i2)?,
        ])
    }
}

pub fn bake_hair_mesh(strands: &HairStrands) -> HairMesh {
    bake_hair_mesh_with_scalp(strands, None)
}

pub fn bake_hair_mesh_with_scalp(strands: &HairStrands, scalp: Option<&Scalp>) -> HairMesh {
    let strand_count = strands.strands.len();
    if strand_count == 0 {
        return empty_mesh();
    }

    let roots: Vec<Vec3> = strands
        .strands
        .iter()
        .map(|&(start, _)| strands.points[start as usize])
        .collect();
    let groups = cluster_roots(&roots, scalp);

    let mut coherent = Vec::new();
    for group in groups {
        if group.is_empty() {
            continue;
        }
        coherent.extend(split_until_coherent(strands, &group));
    }

    let mut bundles = Vec::new();
    let mut corners = Vec::new();
    let mut style = Vec::new();
    let mut bounds_min = Vec3::splat(f32::MAX);
    let mut bounds_max = Vec3::splat(f32::MIN);

    for group in &coherent {
        let indexed = resample_indexed(strands, group);
        if indexed.is_empty() {
            continue;
        }
        let resampled: Vec<Vec<Vec3>> =
            indexed.iter().map(|(_, samples)| samples.clone()).collect();
        let frames = frames_for(&resampled);
        let root_frame = frames[0];
        let mut kept = Vec::new();
        for (strand, samples) in indexed {
            if samples_inside(&frames, &samples) {
                kept.push((strand, samples));
            }
        }
        if kept.is_empty() {
            continue;
        }

        let mut layer_corners = Vec::with_capacity(LAYER_COUNT as usize);
        for frame in &frames {
            let quad = frame.quad();
            for corner in quad {
                bounds_min = bounds_min.min(corner);
                bounds_max = bounds_max.max(corner);
            }
            layer_corners.push(quad);
        }

        let layer_offset = corners.len() as u32;
        let style_offset = style.len() as u32;
        let mut style_sum = vec![Vec3::ZERO; STYLE_TEXELS as usize];
        let mut style_weight = vec![0.0f32; STYLE_TEXELS as usize];

        for (_, samples) in &kept {
            let uv = root_frame.uv(samples[0]);
            if !uv_in_unit(uv) {
                continue;
            }
            let uv = uv.clamp(Vec2::ZERO, Vec2::ONE);
            for (layer, point) in samples.iter().enumerate() {
                let residual = *point - bilinear(layer_corners[layer], uv);
                splat_residual(&mut style_sum, &mut style_weight, uv, layer, residual);
            }
        }

        for layer in 0..LAYER_COUNT as usize {
            let quad = layer_corners[layer];
            let prev = layer_corners[layer.saturating_sub(1)];
            let next = layer_corners[(layer + 1).min(LAYER_COUNT as usize - 1)];
            for corner in 0..4 {
                let tangent = if layer == 0 {
                    next[corner] - quad[corner]
                } else if layer + 1 == LAYER_COUNT as usize {
                    quad[corner] - prev[corner]
                } else {
                    (next[corner] - prev[corner]) * 0.5
                };
                corners.push(CageCorner {
                    position: quad[corner],
                    tangent,
                    ao: 1.0,
                });
            }
        }

        for i in 0..STYLE_TEXELS as usize {
            let residual = if style_weight[i] > 0.0 {
                style_sum[i] / style_weight[i]
            } else {
                Vec3::ZERO
            };
            style.push([residual.x, residual.y, residual.z, 0.0]);
        }

        bundles.push(BundleDesc {
            layer_offset,
            style_offset,
            strand_count: kept.len() as u32,
        });
    }

    if bundles.is_empty() {
        return empty_mesh();
    }
    bake_ambient_occlusion(&mut corners, &bundles, strands, scalp);
    HairMesh {
        bundles,
        corners,
        style,
        bounds_min,
        bounds_max,
    }
}

fn empty_mesh() -> HairMesh {
    HairMesh {
        bundles: Vec::new(),
        corners: Vec::new(),
        style: Vec::new(),
        bounds_min: Vec3::ZERO,
        bounds_max: Vec3::ZERO,
    }
}

fn cluster_roots(roots: &[Vec3], scalp: Option<&Scalp>) -> Vec<Vec<usize>> {
    let all: Vec<usize> = (0..roots.len()).collect();
    let groups = match scalp {
        Some(scalp) if scalp.triangle_count() > 0 => cluster_on_scalp(roots, scalp),
        _ => cluster_roots_3d(roots, &all),
    };
    absorb_sparse(groups, roots)
}

fn cluster_on_scalp(roots: &[Vec3], scalp: &Scalp) -> Vec<Vec<usize>> {
    let tri_count = scalp.triangle_count();
    let accel = TriGrid::build(scalp);
    let mut members = vec![Vec::new(); tri_count];
    let mut missed = Vec::new();
    for (strand, root) in roots.iter().enumerate() {
        match accel.nearest(scalp, *root) {
            Some((tri, dist)) if dist <= SCALP_ASSIGN_DIST => {
                members[tri as usize].push(strand);
            }
            _ => missed.push(strand),
        }
    }

    let adjacency = triangle_adjacency(scalp, tri_count);
    let mut visited = vec![false; tri_count];
    let mut seeds: Vec<usize> = (0..tri_count)
        .filter(|tri| !members[*tri].is_empty())
        .collect();
    seeds.sort_by_key(|tri| std::cmp::Reverse(members[*tri].len()));

    let mut groups = Vec::new();
    let target = TARGET_STRANDS_PER_CELL as usize;
    for seed in seeds {
        if visited[seed] {
            continue;
        }
        let mut count = 0usize;
        let mut patch = Vec::new();
        let mut frontier = VecDeque::new();
        frontier.push_back(seed);
        while let Some(tri) = frontier.pop_front() {
            if visited[tri] {
                continue;
            }
            if count >= target && !patch.is_empty() {
                continue;
            }
            visited[tri] = true;
            count += members[tri].len();
            patch.push(tri);
            if count >= target {
                continue;
            }
            for &neighbor in &adjacency[tri] {
                let neighbor = neighbor as usize;
                if !visited[neighbor] && !members[neighbor].is_empty() {
                    frontier.push_back(neighbor);
                }
            }
        }
        let mut group = Vec::new();
        for tri in patch {
            group.extend_from_slice(&members[tri]);
        }
        if group.is_empty() {
            continue;
        }
        if group.len() > target * 2 {
            groups.extend(cluster_roots_3d(roots, &group));
        } else {
            groups.push(group);
        }
    }
    if !missed.is_empty() {
        groups.extend(cluster_roots_3d(roots, &missed));
    }
    groups
}

fn triangle_adjacency(scalp: &Scalp, tri_count: usize) -> Vec<Vec<u32>> {
    let mut edges: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for tri in 0..tri_count {
        let base = tri * 3;
        let idx = [
            scalp.indices[base],
            scalp.indices[base + 1],
            scalp.indices[base + 2],
        ];
        for edge in 0..3 {
            let a = idx[edge];
            let b = idx[(edge + 1) % 3];
            let key = if a < b { (a, b) } else { (b, a) };
            edges.entry(key).or_default().push(tri as u32);
        }
    }
    let mut adjacency = vec![Vec::new(); tri_count];
    for tris in edges.values() {
        if tris.len() < 2 {
            continue;
        }
        for &a in tris {
            for &b in tris {
                if a != b {
                    adjacency[a as usize].push(b);
                }
            }
        }
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    adjacency
}

fn cluster_roots_3d(roots: &[Vec3], which: &[usize]) -> Vec<Vec<usize>> {
    if which.is_empty() {
        return Vec::new();
    }
    if which.len() <= TARGET_STRANDS_PER_CELL as usize {
        return vec![which.to_vec()];
    }
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for &index in which {
        min = min.min(roots[index]);
        max = max.max(roots[index]);
    }
    let extent = (max - min).max(Vec3::splat(1e-3));
    let cells_axis = ((which.len() as f32 / TARGET_STRANDS_PER_CELL).cbrt())
        .ceil()
        .clamp(2.0, 32.0) as i32;
    let cell = extent / cells_axis as f32;
    let mut buckets: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for &index in which {
        let p = (roots[index] - min) / cell;
        let key = (
            (p.x as i32).clamp(0, cells_axis - 1),
            (p.y as i32).clamp(0, cells_axis - 1),
            (p.z as i32).clamp(0, cells_axis - 1),
        );
        buckets.entry(key).or_default().push(index);
    }
    let mut groups = Vec::new();
    let mut buckets: Vec<_> = buckets.into_iter().collect();
    buckets.sort_by_key(|(key, _)| *key);
    for (_, group) in buckets {
        if group.len() > TARGET_STRANDS_PER_CELL as usize * 2 {
            groups.extend(split_along_axis(roots, &group));
        } else {
            groups.push(group);
        }
    }
    groups
}

fn split_along_axis(roots: &[Vec3], group: &[usize]) -> Vec<Vec<usize>> {
    let points: Vec<Vec3> = group.iter().map(|&index| roots[index]).collect();
    let (mean, axis, _) = plane_axes(&points);
    let mut order: Vec<(f32, usize)> = group
        .iter()
        .map(|&index| ((roots[index] - mean).dot(axis), index))
        .collect();
    order.sort_by(|a, b| a.0.total_cmp(&b.0));
    if order[order.len() - 1].0 - order[0].0 < 1e-3 {
        return vec![group.to_vec()];
    }
    let chunk = TARGET_STRANDS_PER_CELL as usize;
    order
        .chunks(chunk.max(1))
        .map(|piece| piece.iter().map(|entry| entry.1).collect())
        .collect()
}

fn absorb_sparse(mut groups: Vec<Vec<usize>>, roots: &[Vec3]) -> Vec<Vec<usize>> {
    groups.retain(|group| !group.is_empty());
    if groups.len() <= 1 {
        return groups;
    }
    let mut centroids: Vec<Vec3> = groups.iter().map(|group| centroid(group, roots)).collect();
    let mut alive = vec![true; groups.len()];
    let merge_dist_sq = MAX_QUAD_GAP * MAX_QUAD_GAP;
    for i in 0..groups.len() {
        if !alive[i] || groups[i].len() >= MIN_STRANDS_PER_BUNDLE {
            continue;
        }
        let mut best = None;
        let mut best_d = f32::MAX;
        for (j, other_alive) in alive.iter().enumerate() {
            if i == j || !other_alive {
                continue;
            }
            let dist = centroids[i].distance_squared(centroids[j]);
            if dist < best_d {
                best_d = dist;
                best = Some(j);
            }
        }
        let Some(j) = best else {
            continue;
        };
        if best_d > merge_dist_sq {
            continue;
        }
        let taken = std::mem::take(&mut groups[i]);
        groups[j].extend(taken);
        centroids[j] = centroid(&groups[j], roots);
        alive[i] = false;
    }
    groups
        .into_iter()
        .zip(alive)
        .filter_map(|(group, keep)| keep.then_some(group))
        .filter(|group| !group.is_empty())
        .collect()
}

fn centroid(group: &[usize], roots: &[Vec3]) -> Vec3 {
    let mut acc = Vec3::ZERO;
    for &index in group {
        acc += roots[index];
    }
    acc / group.len().max(1) as f32
}

struct TriGrid {
    origin: Vec3,
    cell: f32,
    dims: (i32, i32, i32),
    buckets: Vec<Vec<u32>>,
}

impl TriGrid {
    fn build(scalp: &Scalp) -> Self {
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for position in &scalp.positions {
            min = min.min(*position);
            max = max.max(*position);
        }
        if !min.is_finite() {
            min = Vec3::ZERO;
            max = Vec3::ONE;
        }
        let mut cell = 4.0f32;
        let mut dims = grid_dims(min, max, cell);
        while dims.0 * dims.1 * dims.2 > 250_000 {
            cell *= 2.0;
            dims = grid_dims(min, max, cell);
        }
        let mut buckets = vec![Vec::new(); (dims.0 * dims.1 * dims.2) as usize];
        for tri in 0..scalp.triangle_count() {
            let Some(verts) = scalp.triangle(tri) else {
                continue;
            };
            let tri_min = verts[0].min(verts[1]).min(verts[2]);
            let tri_max = verts[0].max(verts[1]).max(verts[2]);
            let i0 = cell_coords(tri_min, min, cell, dims);
            let i1 = cell_coords(tri_max, min, cell, dims);
            let volume = ((i1.0 - i0.0 + 1) * (i1.1 - i0.1 + 1) * (i1.2 - i0.2 + 1)).max(1);
            if volume > 64 {
                if let Some(slot) =
                    bucket_index(cell_coords(average(&verts), min, cell, dims), dims)
                {
                    buckets[slot].push(tri as u32);
                }
                continue;
            }
            for z in i0.2..=i1.2 {
                for y in i0.1..=i1.1 {
                    for x in i0.0..=i1.0 {
                        if let Some(slot) = bucket_index((x, y, z), dims) {
                            buckets[slot].push(tri as u32);
                        }
                    }
                }
            }
        }
        Self {
            origin: min,
            cell,
            dims,
            buckets,
        }
    }

    fn nearest(&self, scalp: &Scalp, point: Vec3) -> Option<(u32, f32)> {
        let center = cell_coords(point, self.origin, self.cell, self.dims);
        let mut best: Option<(u32, f32)> = None;
        for radius in 0i32..=6 {
            for dz in -radius..=radius {
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        if dx.abs().max(dy.abs()).max(dz.abs()) != radius {
                            continue;
                        }
                        let coords = (center.0 + dx, center.1 + dy, center.2 + dz);
                        let Some(slot) = bucket_index(coords, self.dims) else {
                            continue;
                        };
                        for &tri in &self.buckets[slot] {
                            let Some(verts) = scalp.triangle(tri as usize) else {
                                continue;
                            };
                            let closest = closest_on_triangle(point, verts[0], verts[1], verts[2]);
                            let dist = closest.distance(point);
                            if best.is_none_or(|(_, best_dist)| dist < best_dist) {
                                best = Some((tri, dist));
                            }
                        }
                    }
                }
            }
            if let Some((_, dist)) = best
                && dist <= radius as f32 * self.cell
            {
                break;
            }
        }
        best
    }
}

fn grid_dims(min: Vec3, max: Vec3, cell: f32) -> (i32, i32, i32) {
    let extent = (max - min) / cell + Vec3::ONE;
    (
        extent.x.ceil().clamp(1.0, 128.0) as i32,
        extent.y.ceil().clamp(1.0, 128.0) as i32,
        extent.z.ceil().clamp(1.0, 128.0) as i32,
    )
}

fn cell_coords(point: Vec3, origin: Vec3, cell: f32, dims: (i32, i32, i32)) -> (i32, i32, i32) {
    let p = (point - origin) / cell;
    (
        (p.x.floor() as i32).clamp(0, dims.0 - 1),
        (p.y.floor() as i32).clamp(0, dims.1 - 1),
        (p.z.floor() as i32).clamp(0, dims.2 - 1),
    )
}

fn bucket_index(coords: (i32, i32, i32), dims: (i32, i32, i32)) -> Option<usize> {
    if coords.0 < 0
        || coords.1 < 0
        || coords.2 < 0
        || coords.0 >= dims.0
        || coords.1 >= dims.1
        || coords.2 >= dims.2
    {
        return None;
    }
    Some(((coords.2 * dims.1 + coords.1) * dims.0 + coords.0) as usize)
}

fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    let denom = va + vb + vc;
    if denom.abs() < 1e-12 {
        return a;
    }
    let v = vb / denom;
    let w = vc / denom;
    a + ab * v + ac * w
}

#[derive(Clone, Copy)]
struct Frame {
    mean: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    min_u: f32,
    max_u: f32,
    min_v: f32,
    max_v: f32,
}

impl Frame {
    fn width_u(&self) -> f32 {
        self.max_u - self.min_u
    }

    fn width_v(&self) -> f32 {
        self.max_v - self.min_v
    }

    fn quad(&self) -> [Vec3; 4] {
        let u0 = self.axis_u * self.min_u;
        let u1 = self.axis_u * self.max_u;
        let v0 = self.axis_v * self.min_v;
        let v1 = self.axis_v * self.max_v;
        [
            self.mean + u0 + v0,
            self.mean + u1 + v0,
            self.mean + u0 + v1,
            self.mean + u1 + v1,
        ]
    }

    fn uv(&self, point: Vec3) -> Vec2 {
        let d = point - self.mean;
        let lu = d.dot(self.axis_u);
        let lv = d.dot(self.axis_v);
        let u = if self.max_u > self.min_u + 1e-8 {
            (lu - self.min_u) / (self.max_u - self.min_u)
        } else {
            0.5
        };
        let v = if self.max_v > self.min_v + 1e-8 {
            (lv - self.min_v) / (self.max_v - self.min_v)
        } else {
            0.5
        };
        Vec2::new(u, v)
    }

    fn longer_axis(&self) -> Vec3 {
        if self.width_u() >= self.width_v() {
            self.axis_u
        } else {
            self.axis_v
        }
    }
}

fn resample_indexed(strands: &HairStrands, group: &[usize]) -> Vec<(usize, Vec<Vec3>)> {
    let layers = LAYER_COUNT as usize;
    let mut out = Vec::with_capacity(group.len());
    for &strand in group {
        let (start, count) = strands.strands[strand];
        if count < 2 {
            continue;
        }
        let pts = &strands.points[start as usize..(start + count) as usize];
        let mut cumulative = vec![0.0f32; pts.len()];
        for i in 1..pts.len() {
            cumulative[i] = cumulative[i - 1] + pts[i].distance(pts[i - 1]);
        }
        let length = cumulative[pts.len() - 1];
        if length < 1e-5 {
            continue;
        }
        let mut samples = Vec::with_capacity(layers);
        for layer in 0..layers {
            let target = length * layer as f32 / (layers - 1) as f32;
            let mut index = 1;
            while index + 1 < pts.len() && cumulative[index] < target {
                index += 1;
            }
            let span = (cumulative[index] - cumulative[index - 1]).max(1e-6);
            let t = ((target - cumulative[index - 1]) / span).clamp(0.0, 1.0);
            samples.push(pts[index - 1].lerp(pts[index], t));
        }
        out.push((strand, samples));
    }
    out
}

fn root_frame(strands: &[Vec<Vec3>]) -> Frame {
    let points: Vec<Vec3> = strands.iter().map(|samples| samples[0]).collect();
    let (mean, axis_u, axis_v) = plane_axes(&points);
    let axis_u = axis_u.normalize_or(Vec3::X);
    let axis_v = (axis_v - axis_u * axis_v.dot(axis_u))
        .normalize_or(axis_u.cross(Vec3::Y).normalize_or(Vec3::Z));
    let (min_u, max_u, min_v, max_v) = tight_bounds(&points, mean, axis_u, axis_v);
    Frame {
        mean,
        axis_u,
        axis_v,
        min_u,
        max_u,
        min_v,
        max_v,
    }
}

fn oriented_frame(strands: &[Vec<Vec3>], layer: usize, ref_u: Vec3, ref_v: Vec3) -> Frame {
    let points: Vec<Vec3> = strands.iter().map(|samples| samples[layer]).collect();
    let mean = average(&points);
    let mut tangent = Vec3::ZERO;
    for strand in strands {
        let next = strand[(layer + 1).min(strand.len() - 1)];
        let prev = strand[layer.saturating_sub(1)];
        tangent += next - prev;
    }
    let normal = tangent.normalize_or(Vec3::Y);
    let (axis_u, axis_v) = transported_axes(ref_u, ref_v, normal);
    let (min_u, max_u, min_v, max_v) = tight_bounds(&points, mean, axis_u, axis_v);
    Frame {
        mean,
        axis_u,
        axis_v,
        min_u,
        max_u,
        min_v,
        max_v,
    }
}

fn frames_for(strands: &[Vec<Vec3>]) -> Vec<Frame> {
    let mut frames = Vec::with_capacity(LAYER_COUNT as usize);
    let root = root_frame(strands);
    frames.push(root);
    for layer in 1..LAYER_COUNT as usize {
        let prev = frames[layer - 1];
        frames.push(oriented_frame(strands, layer, prev.axis_u, prev.axis_v));
    }
    frames
}

fn transported_axes(prev_u: Vec3, prev_v: Vec3, normal: Vec3) -> (Vec3, Vec3) {
    let u_plane = prev_u - normal * prev_u.dot(normal);
    let v_plane = prev_v - normal * prev_v.dot(normal);
    let u_len2 = u_plane.length_squared();
    let v_len2 = v_plane.length_squared();
    if u_len2 >= v_len2 && u_len2 > 1e-8 {
        let axis_u = u_plane.normalize();
        let mut axis_v = normal.cross(axis_u);
        if axis_v.dot(prev_v) < 0.0 {
            axis_v = -axis_v;
        }
        (axis_u, axis_v.normalize_or(Vec3::X))
    } else if v_len2 > 1e-8 {
        let axis_v = v_plane.normalize();
        let mut axis_u = axis_v.cross(normal);
        if axis_u.dot(prev_u) < 0.0 {
            axis_u = -axis_u;
        }
        (axis_u.normalize_or(Vec3::X), axis_v)
    } else {
        let axis_u = normal.cross(Vec3::Y).normalize_or(Vec3::X);
        let axis_v = normal.cross(axis_u).normalize_or(Vec3::Z);
        (axis_u, axis_v)
    }
}

fn tight_bounds(points: &[Vec3], mean: Vec3, axis_u: Vec3, axis_v: Vec3) -> (f32, f32, f32, f32) {
    if points.is_empty() {
        return (-1e-3, 1e-3, -1e-3, 1e-3);
    }
    let mut min_u = f32::MAX;
    let mut max_u = f32::MIN;
    let mut min_v = f32::MAX;
    let mut max_v = f32::MIN;
    for point in points {
        let d = *point - mean;
        min_u = min_u.min(d.dot(axis_u));
        max_u = max_u.max(d.dot(axis_u));
        min_v = min_v.min(d.dot(axis_v));
        max_v = max_v.max(d.dot(axis_v));
    }
    let pad_u = (max_u - min_u).max(1e-3) * 1e-3 + 1e-4;
    let pad_v = (max_v - min_v).max(1e-3) * 1e-3 + 1e-4;
    let mut min_u = min_u - pad_u;
    let mut max_u = max_u + pad_u;
    let mut min_v = min_v - pad_v;
    let mut max_v = max_v + pad_v;
    if max_u - min_u < 1e-2 {
        let mid = (max_u + min_u) * 0.5;
        min_u = mid - 5e-3;
        max_u = mid + 5e-3;
    }
    if max_v - min_v < 1e-2 {
        let mid = (max_v + min_v) * 0.5;
        min_v = mid - 5e-3;
        max_v = mid + 5e-3;
    }
    (min_u, max_u, min_v, max_v)
}

fn samples_inside(frames: &[Frame], samples: &[Vec3]) -> bool {
    frames
        .iter()
        .zip(samples)
        .all(|(frame, point)| uv_in_unit(frame.uv(*point)))
}

fn uv_in_unit(uv: Vec2) -> bool {
    uv.x >= -1e-3 && uv.y >= -1e-3 && uv.x <= 1.0 + 1e-3 && uv.y <= 1.0 + 1e-3
}

struct StyleFit {
    sum: Vec<Vec3>,
    sum_sq: Vec<f32>,
    weight: Vec<f32>,
}

fn accumulate_style(strands: &[Vec<Vec3>], frames: &[Frame]) -> StyleFit {
    let mut sum = vec![Vec3::ZERO; STYLE_TEXELS as usize];
    let mut sum_sq = vec![0.0f32; STYLE_TEXELS as usize];
    let mut weight = vec![0.0f32; STYLE_TEXELS as usize];
    let root = frames[0];
    for strand in strands {
        let uv = root.uv(strand[0]);
        if !uv_in_unit(uv) {
            continue;
        }
        let uv = uv.clamp(Vec2::ZERO, Vec2::ONE);
        for (layer, point) in strand.iter().enumerate() {
            let residual = *point - bilinear(frames[layer].quad(), uv);
            splat_residual(&mut sum, &mut weight, uv, layer, residual);
            let u = uv.x * (STYLE_U - 1) as f32;
            let v = uv.y * (STYLE_V - 1) as f32;
            let u0 = u.floor() as u32;
            let v0 = v.floor() as u32;
            let u1 = (u0 + 1).min(STYLE_U - 1);
            let v1 = (v0 + 1).min(STYLE_V - 1);
            let tu = u - u0 as f32;
            let tv = v - v0 as f32;
            let w = layer as u32;
            let length_sq = residual.length_squared();
            let corners = [
                (u0, v0, (1.0 - tu) * (1.0 - tv)),
                (u1, v0, tu * (1.0 - tv)),
                (u0, v1, (1.0 - tu) * tv),
                (u1, v1, tu * tv),
            ];
            for (iu, iv, alpha) in corners {
                if alpha <= 0.0 {
                    continue;
                }
                let index = ((w * STYLE_V + iv) * STYLE_U + iu) as usize;
                sum_sq[index] += length_sq * alpha;
            }
        }
    }
    StyleFit {
        sum,
        sum_sq,
        weight,
    }
}

fn style_means(fit: &StyleFit) -> Vec<[f32; 4]> {
    let mut style = vec![[0.0f32; 4]; STYLE_TEXELS as usize];
    for (texel, ((accumulated, weight), sum_sq)) in style
        .iter_mut()
        .zip(fit.sum.iter().zip(&fit.weight).zip(&fit.sum_sq))
    {
        if *weight > 0.0 {
            let residual = *accumulated / *weight;
            *texel = [residual.x, residual.y, residual.z, 0.0];
            let _ = sum_sq;
        }
    }
    style
}

fn worst_stddev(fit: &StyleFit) -> (f32, usize) {
    let mut worst = 0.0f32;
    let mut layer = 0usize;
    for (index, weight) in fit.weight.iter().enumerate() {
        if *weight <= 1.0 {
            continue;
        }
        let mean = fit.sum[index] / *weight;
        let second = fit.sum_sq[index] / *weight;
        let variance = (second - mean.length_squared()).max(0.0);
        let stddev = variance.sqrt();
        if stddev > worst {
            worst = stddev;
            layer = index / (STYLE_U as usize * STYLE_V as usize);
        }
    }
    (worst, layer)
}

fn style_slice(style: &[[f32; 4]], uv: Vec2, layer: usize) -> Vec3 {
    let u = uv.x.clamp(0.0, 1.0) * (STYLE_U - 1) as f32;
    let v = uv.y.clamp(0.0, 1.0) * (STYLE_V - 1) as f32;
    let u0 = u.floor() as usize;
    let v0 = v.floor() as usize;
    let u1 = (u0 + 1).min(STYLE_U as usize - 1);
    let v1 = (v0 + 1).min(STYLE_V as usize - 1);
    let tu = u - u0 as f32;
    let tv = v - v0 as f32;
    let at = |iu: usize, iv: usize| {
        let texel = style[(layer * STYLE_V as usize + iv) * STYLE_U as usize + iu];
        Vec3::new(texel[0], texel[1], texel[2])
    };
    let c0 = at(u0, v0).lerp(at(u1, v0), tu);
    let c1 = at(u0, v1).lerp(at(u1, v1), tu);
    c0.lerp(c1, tv)
}

/// Distance from styled quad samples to the strands. The rasterizer draws these
/// samples, so a cage-only check would miss a styling residual that leaves the groom.
fn worst_quad_gap(
    strands: &[Vec<Vec3>],
    frames: &[Frame],
    style: &[[f32; 4]],
) -> (usize, Vec3, f32) {
    let mut worst_layer = 0usize;
    let mut worst_axis = frames[0].longer_axis();
    let mut worst = 0.0f32;
    for (layer, frame) in frames.iter().enumerate() {
        let quad = frame.quad();
        let mut gap = 0.0f32;
        for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for u in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let uv = Vec2::new(u, v);
                let pos = bilinear(quad, uv) + style_slice(style, uv, layer);
                let mut nearest = f32::MAX;
                for strand in strands {
                    nearest = nearest.min(pos.distance_squared(strand[layer]));
                }
                gap = gap.max(nearest);
            }
        }
        gap = gap.sqrt();
        if gap > worst {
            worst = gap;
            worst_layer = layer;
            worst_axis = frame.longer_axis();
        }
    }
    (worst_layer, worst_axis, worst)
}

fn split_until_coherent(strands: &HairStrands, group: &[usize]) -> Vec<Vec<usize>> {
    fn rec(strands: &HairStrands, group: &[usize], depth: u32) -> Vec<Vec<usize>> {
        if group.len() < 2 || depth >= 16 {
            return vec![group.to_vec()];
        }
        let indexed = resample_indexed(strands, group);
        if indexed.len() < 2 {
            return vec![indexed.into_iter().map(|(id, _)| id).collect()];
        }
        let resampled: Vec<Vec<Vec3>> =
            indexed.iter().map(|(_, samples)| samples.clone()).collect();
        let frames = frames_for(&resampled);
        let mut inside = Vec::new();
        let mut outside = Vec::new();
        for (strand, samples) in &indexed {
            if samples_inside(&frames, samples) {
                inside.push(*strand);
            } else {
                outside.push(*strand);
            }
        }
        if !outside.is_empty() && !inside.is_empty() {
            let mut out = rec(strands, &inside, depth + 1);
            out.extend(rec(strands, &outside, depth + 1));
            return out;
        }

        let fit = accumulate_style(&resampled, &frames);
        let style = style_means(&fit);
        let (gap_layer, gap_axis, gap) = worst_quad_gap(&resampled, &frames, &style);
        let (stddev, var_layer) = worst_stddev(&fit);
        let (member_error, member_layer, member_axis) =
            max_member_error(&resampled, &frames, &style);
        let gap_bad = gap > MAX_QUAD_GAP;
        let var_bad = stddev > MAX_STYLE_STDDEV;
        let member_bad = member_error > MAX_MEMBER_ERROR;
        if !gap_bad && !var_bad && !member_bad {
            return vec![indexed.into_iter().map(|(id, _)| id).collect()];
        }
        let (layer, axis) = if member_bad
            && member_error / MAX_MEMBER_ERROR >= gap / MAX_QUAD_GAP
            && member_error / MAX_MEMBER_ERROR >= stddev / MAX_STYLE_STDDEV.max(1e-3)
        {
            (member_layer, member_axis)
        } else if var_bad && (!gap_bad || stddev / MAX_STYLE_STDDEV > gap / MAX_QUAD_GAP) {
            (
                var_layer.min(frames.len() - 1),
                frames[var_layer.min(frames.len() - 1)].longer_axis(),
            )
        } else {
            (gap_layer, gap_axis)
        };
        let mean = average(
            &resampled
                .iter()
                .map(|samples| samples[layer])
                .collect::<Vec<_>>(),
        );
        let mut order: Vec<(f32, usize)> = indexed
            .iter()
            .map(|(strand, samples)| ((samples[layer] - mean).dot(axis), *strand))
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let span = order[order.len() - 1].0 - order[0].0;
        let mid = order.len() / 2;
        if mid == 0 {
            return vec![group.to_vec()];
        }
        // Strands that share a uv but disagree cannot be split geometrically.
        // Give each half its own bundle so `f(uvw)` can follow them.
        if span < 1e-3 {
            let left: Vec<usize> = group[..mid].to_vec();
            let right: Vec<usize> = group[mid..].to_vec();
            let mut out = rec(strands, &left, depth + 1);
            out.extend(rec(strands, &right, depth + 1));
            return out;
        }
        let left: Vec<usize> = order[..mid].iter().map(|entry| entry.1).collect();
        let right: Vec<usize> = order[mid..].iter().map(|entry| entry.1).collect();
        let mut out = rec(strands, &left, depth + 1);
        out.extend(rec(strands, &right, depth + 1));
        out
    }
    rec(strands, group, 0)
}

fn max_member_error(
    strands: &[Vec<Vec3>],
    frames: &[Frame],
    style: &[[f32; 4]],
) -> (f32, usize, Vec3) {
    let mut worst = 0.0f32;
    let mut layer = 0usize;
    let mut axis = frames[0].longer_axis();
    for strand in strands {
        let uv = frames[0].uv(strand[0]);
        if !uv_in_unit(uv) {
            continue;
        }
        let uv = uv.clamp(Vec2::ZERO, Vec2::ONE);
        for (index, point) in strand.iter().enumerate() {
            let predicted = bilinear(frames[index].quad(), uv) + style_slice(style, uv, index);
            let error = predicted.distance(*point);
            if error > worst {
                worst = error;
                layer = index;
                axis = frames[index].longer_axis();
            }
        }
    }
    (worst, layer, axis)
}

fn bilinear(quad: [Vec3; 4], uv: Vec2) -> Vec3 {
    let u = uv.x;
    let v = uv.y;
    quad[0] * (1.0 - u) * (1.0 - v)
        + quad[1] * u * (1.0 - v)
        + quad[2] * (1.0 - u) * v
        + quad[3] * u * v
}

fn splat_residual(sum: &mut [Vec3], weight: &mut [f32], uv: Vec2, layer: usize, residual: Vec3) {
    let u = uv.x.clamp(0.0, 1.0) * (STYLE_U - 1) as f32;
    let v = uv.y.clamp(0.0, 1.0) * (STYLE_V - 1) as f32;
    let u0 = u.floor() as u32;
    let v0 = v.floor() as u32;
    let u1 = (u0 + 1).min(STYLE_U - 1);
    let v1 = (v0 + 1).min(STYLE_V - 1);
    let tu = u - u0 as f32;
    let tv = v - v0 as f32;
    let w = layer as u32;
    let corners = [
        (u0, v0, (1.0 - tu) * (1.0 - tv)),
        (u1, v0, tu * (1.0 - tv)),
        (u0, v1, (1.0 - tu) * tv),
        (u1, v1, tu * tv),
    ];
    for (iu, iv, alpha) in corners {
        if alpha <= 0.0 {
            continue;
        }
        let index = ((w * STYLE_V + iv) * STYLE_U + iu) as usize;
        sum[index] += residual * alpha;
        weight[index] += alpha;
    }
}

struct OccluderGrid {
    origin: Vec3,
    cell: f32,
    segments: Vec<(Vec3, Vec3)>,
    triangles: Vec<[Vec3; 3]>,
    buckets: HashMap<(i32, i32, i32), Bucket>,
}

#[derive(Default)]
struct Bucket {
    segments: Vec<u32>,
    triangles: Vec<u32>,
}

fn bake_ambient_occlusion(
    corners: &mut [CageCorner],
    bundles: &[BundleDesc],
    strands: &HairStrands,
    scalp: Option<&Scalp>,
) {
    if corners.is_empty() {
        return;
    }
    let grid = OccluderGrid::build(strands, scalp);
    let mut ao = vec![1.0f32; corners.len()];
    for bundle in bundles {
        let base = bundle.layer_offset as usize;
        for layer in 0..LAYER_COUNT as usize {
            let mut center = Vec3::ZERO;
            for corner in 0..4 {
                center += corners[base + layer * 4 + corner].position;
            }
            center *= 0.25;
            for corner in 0..4 {
                let index = base + layer * 4 + corner;
                let position = corners[index].position;
                let tangent = corners[index].tangent.normalize_or(Vec3::Y);
                let mut outward = position - center;
                outward -= tangent * outward.dot(tangent);
                let outward = outward.normalize_or(tangent.cross(Vec3::Y).normalize_or(Vec3::X));
                let mut occluded = 0.0f32;
                for ray in 0..AO_RAYS {
                    let dir = hemisphere_dir(ray, AO_RAYS, outward);
                    if grid.occluded(position, dir) {
                        occluded += 1.0;
                    }
                }
                ao[index] = (1.0 - occluded / AO_RAYS as f32).clamp(0.0, 1.0);
            }
        }
    }
    for (corner, value) in corners.iter_mut().zip(ao) {
        corner.ao = value;
    }
}

impl OccluderGrid {
    fn build(strands: &HairStrands, scalp: Option<&Scalp>) -> Self {
        let mut segments = Vec::new();
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for &(start, count) in &strands.strands {
            if count < 2 {
                continue;
            }
            let pts = &strands.points[start as usize..(start + count) as usize];
            let step = ((pts.len().saturating_sub(1)) as f32 / 16.0)
                .ceil()
                .max(1.0) as usize;
            let mut index = 0;
            while index + 1 < pts.len() {
                let next = (index + step).min(pts.len() - 1);
                if next > index {
                    let a = pts[index];
                    let b = pts[next];
                    min = min.min(a).min(b);
                    max = max.max(a).max(b);
                    segments.push((a, b));
                }
                if next + 1 >= pts.len() {
                    break;
                }
                index = next;
            }
        }
        let mut triangles = Vec::new();
        if let Some(scalp) = scalp {
            for tri in 0..scalp.triangle_count() {
                if let Some(verts) = scalp.triangle(tri) {
                    min = min.min(verts[0]).min(verts[1]).min(verts[2]);
                    max = max.max(verts[0]).max(verts[1]).max(verts[2]);
                    triangles.push(verts);
                }
            }
        }
        if !min.is_finite() {
            min = Vec3::ZERO;
            max = Vec3::ONE;
        }
        let cell = 4.0f32;
        let mut buckets: HashMap<(i32, i32, i32), Bucket> = HashMap::new();
        let key_of = |point: Vec3| {
            let p = (point - min) / cell;
            (p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32)
        };
        for (index, (a, b)) in segments.iter().enumerate() {
            for point in [*a, *b, (*a + *b) * 0.5] {
                buckets
                    .entry(key_of(point))
                    .or_default()
                    .segments
                    .push(index as u32);
            }
        }
        for (index, verts) in triangles.iter().enumerate() {
            let center = (verts[0] + verts[1] + verts[2]) / 3.0;
            buckets
                .entry(key_of(center))
                .or_default()
                .triangles
                .push(index as u32);
        }
        let _ = max;
        Self {
            origin: min,
            cell,
            segments,
            triangles,
            buckets,
        }
    }

    fn occluded(&self, origin: Vec3, dir: Vec3) -> bool {
        let steps = ((AO_REACH / (self.cell * 0.5)).ceil() as i32).clamp(1, 8);
        let dt = AO_REACH / steps as f32;
        for step in 0..=steps {
            let point = origin + dir * (step as f32 * dt);
            let p = (point - self.origin) / self.cell;
            let key = (p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32);
            let Some(bucket) = self.buckets.get(&key) else {
                continue;
            };
            for &index in &bucket.segments {
                let (a, b) = self.segments[index as usize];
                if ray_segment(origin, dir, a, b) {
                    return true;
                }
            }
            for &index in &bucket.triangles {
                let verts = self.triangles[index as usize];
                if ray_triangle(origin, dir, verts[0], verts[1], verts[2]) {
                    return true;
                }
            }
        }
        false
    }
}

fn hemisphere_dir(index: usize, count: usize, normal: Vec3) -> Vec3 {
    let z = ((index as f32 + 0.5) / count as f32).sqrt();
    let phi = index as f32 * 2.399_963_1;
    let radius = (1.0 - z * z).max(0.0).sqrt();
    let local = Vec3::new(radius * phi.cos(), radius * phi.sin(), z);
    let tangent = normal.cross(Vec3::Y).normalize_or(Vec3::X);
    let bitangent = normal.cross(tangent).normalize_or(Vec3::Z);
    (tangent * local.x + bitangent * local.y + normal * local.z).normalize_or(normal)
}

fn ray_segment(origin: Vec3, dir: Vec3, a: Vec3, b: Vec3) -> bool {
    let ab = b - a;
    let v = origin - a;
    let d2 = dir.dot(ab);
    let d3 = ab.length_squared().max(1e-12);
    let d4 = v.dot(dir);
    let d5 = v.dot(ab);
    let denom = d3 - d2 * d2;
    let mut s = if denom.abs() < 1e-8 {
        0.0
    } else {
        ((d5 - d4 * d2) / denom).clamp(0.0, 1.0)
    };
    let mut t = (a + ab * s - origin).dot(dir);
    if !(AO_MIN_T..=AO_REACH).contains(&t) {
        t = t.clamp(AO_MIN_T, AO_REACH);
        s = ((origin + dir * t - a).dot(ab) / d3).clamp(0.0, 1.0);
        t = (a + ab * s - origin).dot(dir);
        if !(AO_MIN_T..=AO_REACH).contains(&t) {
            return false;
        }
    }
    let closest_ray = origin + dir * t;
    let closest_seg = a + ab * s;
    closest_ray.distance_squared(closest_seg) <= AO_RADIUS * AO_RADIUS
}

fn ray_triangle(origin: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> bool {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-6 {
        return false;
    }
    let inv = 1.0 / det;
    let tvec = origin - v0;
    let u = tvec.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = tvec.cross(e1);
    let v = dir.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let t = e2.dot(q) * inv;
    (AO_MIN_T..=AO_REACH).contains(&t)
}

fn average(points: &[Vec3]) -> Vec3 {
    let mut acc = Vec3::ZERO;
    for point in points {
        acc += *point;
    }
    acc / points.len().max(1) as f32
}

fn plane_axes(points: &[Vec3]) -> (Vec3, Vec3, Vec3) {
    let mean = average(points);
    let mut cov = Mat3::ZERO;
    for point in points {
        let d = *point - mean;
        cov += Mat3::from_cols(d * d.x, d * d.y, d * d.z);
    }
    let axis_u = dominant_axis(cov, Vec3::X);
    let deflated = deflate(cov, axis_u);
    let mut axis_v = dominant_axis(deflated, Vec3::Y);
    if axis_u.dot(axis_v).abs() > 0.99 {
        axis_v = axis_u.cross(Vec3::Y).normalize_or(Vec3::X);
    }
    (mean, axis_u, axis_v.normalize())
}

fn dominant_axis(cov: Mat3, seed: Vec3) -> Vec3 {
    let mut v = seed.normalize_or(Vec3::X);
    for _ in 0..24 {
        let next = cov * v;
        if next.length_squared() < 1e-12 {
            break;
        }
        v = next.normalize();
    }
    v
}

fn deflate(cov: Mat3, axis: Vec3) -> Mat3 {
    let lambda = axis.dot(cov * axis);
    cov - Mat3::from_cols(axis * axis.x, axis * axis.y, axis * axis.z) * lambda
}

#[cfg(test)]
mod tests {
    use super::*;

    type SegmentBins = std::collections::HashMap<(i32, i32, i32), Vec<(Vec3, Vec3)>>;

    #[test]
    fn straight_cage_residuals_are_near_zero() {
        let hair = straight_bundle();
        let mesh = bake_hair_mesh(&hair);
        assert_eq!(mesh.bundles.len(), 1);
        assert!(
            mesh.corners
                .iter()
                .all(|c| c.position.is_finite() && c.tangent.is_finite())
        );
        let max_residual = mesh
            .style
            .iter()
            .map(|r| Vec3::new(r[0], r[1], r[2]).length())
            .fold(0.0f32, f32::max);
        assert!(
            max_residual < 1e-2,
            "expected the cage to match the corner strands, residual {max_residual}"
        );
    }

    #[test]
    fn isolated_bundle_ao_does_not_darken_toward_the_root() {
        let hair = straight_bundle();
        let mesh = bake_hair_mesh(&hair);
        assert_eq!(mesh.bundles.len(), 1);
        let mean_layer = |layer: usize| {
            let base = layer * 4;
            mesh.corners[base..base + 4]
                .iter()
                .map(|corner| corner.ao)
                .sum::<f32>()
                / 4.0
        };
        let root = mean_layer(0);
        let tip = mean_layer(LAYER_COUNT as usize - 1);
        assert!(
            root + 0.15 >= tip,
            "isolated root AO {root} should not fall off toward the scalp relative to tip {tip}"
        );
    }

    #[test]
    fn diverging_strands_are_not_bridged() {
        let mut points = Vec::new();
        let mut strands = Vec::new();
        for side in [-1.0f32, 1.0] {
            for i in 0..16 {
                let start = points.len() as u32;
                let tip = Vec3::new(side * 30.0, (i as f32 - 7.5) * 0.15, -40.0);
                for layer in 0..16 {
                    let t = layer as f32 / 15.0;
                    points.push(Vec3::ZERO.lerp(tip, t));
                }
                strands.push((start, 16));
            }
        }
        let hair = HairStrands {
            points: points.clone(),
            strands: strands.clone(),
            default_color: [1.0, 1.0, 1.0],
        };
        let mesh = bake_hair_mesh(&hair);
        assert!(
            mesh.bundles.len() > 1,
            "the two locks should not share one quad"
        );
        let stray = max_styled_stray(&mesh, &points, &strands);
        assert!(
            stray < 8.0,
            "uniform samples inside a bundle quad left the groom by {stray}"
        );
    }

    #[test]
    fn bent_bundle_keeps_its_cross_section() {
        let mut points = Vec::new();
        let mut strands = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let start = points.len() as u32;
                let root = Vec3::new(i as f32 * 0.4, j as f32 * 0.4, 0.0);
                for layer in 0..16 {
                    let angle = layer as f32 / 15.0 * std::f32::consts::FRAC_PI_2;
                    points.push(
                        root + Vec3::new(20.0 * (1.0 - angle.cos()), 0.0, 20.0 * angle.sin()),
                    );
                }
                strands.push((start, 16));
            }
        }
        let hair = HairStrands {
            points: points.clone(),
            strands: strands.clone(),
            default_color: [1.0, 1.0, 1.0],
        };
        let mesh = bake_hair_mesh(&hair);
        let stray = max_styled_stray(&mesh, &points, &strands);
        assert!(
            stray < 8.0,
            "a bending lock inflated its quad, stray {stray}"
        );
    }

    #[test]
    fn baked_grooms_stay_on_the_source_strands() {
        let scalp_mesh = crate::obj::load_obj_path(format!(
            "{}/assets/hair/woman.obj",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("woman.obj");
        let scalp = Scalp::from_mesh(&scalp_mesh);
        for file in ["wStraight.hair", "wWavy.hair", "wCurly.hair"] {
            let path = format!("{}/assets/hair/{file}", env!("CARGO_MANIFEST_DIR"));
            let hair = crate::hair_file::load_hair_path(&path).unwrap_or_else(|err| {
                panic!("failed to read {path}: {err}");
            });
            let mesh = bake_hair_mesh_with_scalp(&hair, Some(&scalp));
            let usable = hair.strands.iter().filter(|(_, count)| *count >= 2).count();
            assert_eq!(
                mesh.strand_count() as usize,
                usable,
                "{file} dropped strands"
            );
            assert!(
                first_uncovered_root(&mesh, &hair).is_none(),
                "{file} leaves a source root outside every root quad"
            );
            let stray = max_styled_stray(&mesh, &hair.points, &hair.strands);
            assert!(stray < 8.0, "{file} leaves the source groom by {stray:.1}");
            let error = max_reconstruction_error(&mesh, &hair);
            assert!(
                error < 6.0,
                "{file} reconstruction misses a strand by {error:.1}"
            );
        }
    }

    fn straight_bundle() -> HairStrands {
        let mut points = Vec::new();
        let mut strands = Vec::new();
        let corners = [
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(-1.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
        ];
        for _ in 0..3 {
            for corner in corners {
                let start = points.len() as u32;
                for layer in 0..8 {
                    points.push(corner + Vec3::Y * layer as f32);
                }
                strands.push((start, 8));
            }
        }
        HairStrands {
            points,
            strands,
            default_color: [1.0, 1.0, 1.0],
        }
    }

    fn max_styled_stray(mesh: &HairMesh, points: &[Vec3], strands: &[(u32, u32)]) -> f32 {
        let limit = 8.0f32;
        let cell = 4.0f32;
        let mut origin = Vec3::splat(f32::MAX);
        for point in points {
            origin = origin.min(*point);
        }
        let mut bins = SegmentBins::new();
        let voxel = |p: Vec3| {
            let q = (p - origin) / cell;
            (q.x.floor() as i32, q.y.floor() as i32, q.z.floor() as i32)
        };
        for &(start, count) in strands {
            let start = start as usize;
            let count = count as usize;
            for i in 1..count {
                let a = points[start + i - 1];
                let b = points[start + i];
                let ka = voxel(a);
                let kb = voxel(b);
                for z in ka.2.min(kb.2)..=ka.2.max(kb.2) {
                    for y in ka.1.min(kb.1)..=ka.1.max(kb.1) {
                        for x in ka.0.min(kb.0)..=ka.0.max(kb.0) {
                            bins.entry((x, y, z)).or_default().push((a, b));
                        }
                    }
                }
            }
        }

        let mut worst = 0.0f32;
        let uvs = [0.0f32, 0.25, 0.5, 0.75, 1.0];
        for (bundle_index, _) in mesh.bundles.iter().enumerate() {
            for &v in &uvs {
                for &u in &uvs {
                    let uv = Vec2::new(u, v);
                    for step in 0..(LAYER_COUNT as usize * 2 - 1) {
                        let w = step as f32 / (LAYER_COUNT as f32 * 2.0 - 2.0);
                        let pos = mesh.styled_position(bundle_index, uv, w);
                        let nearest = nearest_segment(&bins, voxel(pos), pos, limit, cell);
                        worst = worst.max(nearest);
                    }
                }
            }
        }
        worst
    }

    fn first_uncovered_root(mesh: &HairMesh, hair: &HairStrands) -> Option<usize> {
        let (origin, cell, grid) = root_bundle_grid(mesh);
        hair.strands
            .iter()
            .enumerate()
            .find_map(|(index, &(start, count))| {
                if count < 2 {
                    return None;
                }
                let root = hair.points[start as usize];
                let covered = containing_bundles(mesh, &grid, origin, cell, root).any(|bundle| {
                    let quad = mesh.layer_corners(bundle, 0);
                    let uv = HairMesh::quad_uv(quad, root);
                    uv_in_unit(uv) && HairMesh::quad_plane_distance(quad, root).abs() < 8.0
                });
                (!covered).then_some(index)
            })
    }

    fn max_reconstruction_error(mesh: &HairMesh, hair: &HairStrands) -> f32 {
        let (origin, cell, grid) = root_bundle_grid(mesh);
        let mut worst = 0.0f32;
        for (index, &(start, count)) in hair.strands.iter().enumerate() {
            if count < 2 {
                continue;
            }
            let root = hair.points[start as usize];
            let mut best = f32::MAX;
            for bundle in containing_bundles(mesh, &grid, origin, cell, root) {
                if let Some(distance) = bundle_reconstructs(mesh, hair, bundle, index, root) {
                    best = best.min(distance);
                }
            }
            if best == f32::MAX {
                best = 1.0e6;
            }
            worst = worst.max(best);
        }
        worst
    }

    fn bundle_reconstructs(
        mesh: &HairMesh,
        hair: &HairStrands,
        bundle: usize,
        strand: usize,
        root: Vec3,
    ) -> Option<f32> {
        let quad = mesh.layer_corners(bundle, 0);
        let uv = HairMesh::quad_uv(quad, root);
        if !uv_in_unit(uv) || HairMesh::quad_plane_distance(quad, root).abs() > 8.0 {
            return None;
        }
        let samples = resample_indexed(hair, &[strand]);
        let (_, samples) = samples.first()?;
        let mut worst = 0.0f32;
        for (layer, point) in samples.iter().enumerate() {
            let w = layer as f32 / (LAYER_COUNT - 1) as f32;
            worst = worst.max(mesh.styled_position(bundle, uv, w).distance(*point));
        }
        Some(worst)
    }

    type BundleGrid = HashMap<(i32, i32, i32), Vec<usize>>;

    fn containing_bundles<'a>(
        mesh: &'a HairMesh,
        grid: &'a BundleGrid,
        origin: Vec3,
        cell: f32,
        root: Vec3,
    ) -> impl Iterator<Item = usize> + 'a {
        let p = (root - origin) / cell;
        let key = (p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32);
        let mut seen = Vec::new();
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(list) = grid.get(&(key.0 + dx, key.1 + dy, key.2 + dz)) {
                        for &bundle in list {
                            if !seen.contains(&bundle) {
                                seen.push(bundle);
                            }
                        }
                    }
                }
            }
        }
        let _ = mesh;
        seen.into_iter()
    }

    fn root_bundle_grid(mesh: &HairMesh) -> (Vec3, f32, BundleGrid) {
        let cell = 8.0f32;
        let mut origin = Vec3::splat(f32::MAX);
        for bundle in 0..mesh.bundles.len() {
            for corner in mesh.layer_corners(bundle, 0) {
                origin = origin.min(corner);
            }
        }
        if !origin.is_finite() {
            origin = Vec3::ZERO;
        }
        origin -= Vec3::splat(cell);
        let mut grid = BundleGrid::new();
        for bundle in 0..mesh.bundles.len() {
            let quad = mesh.layer_corners(bundle, 0);
            let mut min = quad[0];
            let mut max = quad[0];
            for corner in quad {
                min = min.min(corner);
                max = max.max(corner);
            }
            min -= Vec3::splat(4.0);
            max += Vec3::splat(4.0);
            let a = (min - origin) / cell;
            let b = (max - origin) / cell;
            let i0 = (a.x.floor() as i32, a.y.floor() as i32, a.z.floor() as i32);
            let i1 = (b.x.floor() as i32, b.y.floor() as i32, b.z.floor() as i32);
            for z in i0.2..=i1.2 {
                for y in i0.1..=i1.1 {
                    for x in i0.0..=i1.0 {
                        grid.entry((x, y, z)).or_default().push(bundle);
                    }
                }
            }
        }
        (origin, cell, grid)
    }

    fn nearest_segment(
        bins: &SegmentBins,
        key: (i32, i32, i32),
        pos: Vec3,
        limit: f32,
        cell: f32,
    ) -> f32 {
        let (ix, iy, iz) = key;
        let mut nearest = f32::MAX;
        'nearby: for radius in 0i32..=2 {
            for dz in -radius..=radius {
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        if dx.abs().max(dy.abs()).max(dz.abs()) != radius {
                            continue;
                        }
                        let Some(list) = bins.get(&(ix + dx, iy + dy, iz + dz)) else {
                            continue;
                        };
                        for &(a, b) in list {
                            nearest = nearest.min(segment_distance(pos, a, b));
                            if nearest < limit {
                                break 'nearby;
                            }
                        }
                    }
                }
            }
        }
        if nearest < limit {
            return nearest;
        }
        for dz in -8i32..=8 {
            for dy in -8i32..=8 {
                for dx in -8i32..=8 {
                    let Some(list) = bins.get(&(ix + dx, iy + dy, iz + dz)) else {
                        continue;
                    };
                    for &(a, b) in list {
                        nearest = nearest.min(segment_distance(pos, a, b));
                    }
                }
            }
        }
        if nearest == f32::MAX {
            cell * 12.0
        } else {
            nearest
        }
    }

    fn segment_distance(point: Vec3, a: Vec3, b: Vec3) -> f32 {
        let ab = b - a;
        let denom = ab.length_squared().max(1e-8);
        let t = ((point - a).dot(ab) / denom).clamp(0.0, 1.0);
        point.distance(a + ab * t)
    }
}
