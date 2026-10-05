//! Convert explicit strands into the hair-mesh cages the rasterizer generates from.
//!
//! Roots are clustered in the scalp PCA plane, then a cluster is split until its quad
//! stays on those strands. The rasterizer draws `uv = rand.next2()` uniformly inside
//! each quad (Lipp et al. 2026, Appendix B), so a quad that spans empty space grows
//! hair there. The styling function `S(uvw)` is the mean residual from the cage,
//! stored as a trilinear volume.

use bevy::math::{Mat3, Vec2, Vec3};

use crate::hair_file::HairStrands;
use crate::mesh::{
    BundleDesc, CageCorner, HairMesh, LAYER_COUNT, STYLE_TEXELS, STYLE_U, STYLE_V,
};

const MIN_STRANDS_PER_BUNDLE: usize = 8;
const TARGET_STRANDS_PER_CELL: f32 = 180.0;
/// Split a bundle when a sample of its quad is farther than this from every strand.
/// The uniform uv draw in Appendix B fills the whole quad, not just the strands.
const MAX_QUAD_GAP: f32 = 5.0;

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

pub fn bake_hair_mesh(strands: &HairStrands) -> HairMesh {
    let strand_count = strands.strands.len();
    if strand_count == 0 {
        return empty_mesh();
    }

    let roots: Vec<Vec3> = strands
        .strands
        .iter()
        .map(|&(start, _)| strands.points[start as usize])
        .collect();
    let (mean, axis_u, axis_v) = plane_axes(&roots);
    let mut coords = Vec::with_capacity(roots.len());
    let mut min_c = Vec2::splat(f32::MAX);
    let mut max_c = Vec2::splat(f32::MIN);
    for root in &roots {
        let d = *root - mean;
        let c = Vec2::new(d.dot(axis_u), d.dot(axis_v));
        min_c = min_c.min(c);
        max_c = max_c.max(c);
        coords.push(c);
    }
    let extent = (max_c - min_c).max(Vec2::splat(1e-3));
    let cells_side = ((strand_count as f32 / TARGET_STRANDS_PER_CELL).sqrt())
        .ceil()
        .clamp(4.0, 40.0) as i32;
    let cell_size = extent / cells_side as f32;

    let mut cell_of = vec![0i32; strand_count];
    let mut occupancy = vec![0u32; (cells_side * cells_side) as usize];
    for (i, c) in coords.iter().enumerate() {
        let gx = (((c.x - min_c.x) / cell_size.x) as i32).clamp(0, cells_side - 1);
        let gy = (((c.y - min_c.y) / cell_size.y) as i32).clamp(0, cells_side - 1);
        let id = gy * cells_side + gx;
        cell_of[i] = id;
        occupancy[id as usize] += 1;
    }

    // Fold sparse cells into the nearest cell that already has enough strands.
    let mut remap = vec![-1i32; occupancy.len()];
    for id in 0..occupancy.len() {
        if occupancy[id] >= MIN_STRANDS_PER_BUNDLE as u32 {
            remap[id] = id as i32;
        }
    }
    for id in 0..occupancy.len() {
        if remap[id] >= 0 || occupancy[id] == 0 {
            continue;
        }
        let x = (id as i32) % cells_side;
        let y = (id as i32) / cells_side;
        let mut best = -1i32;
        let mut best_d = i32::MAX;
        for (other, mapped) in remap.iter().enumerate() {
            if *mapped != other as i32 {
                continue;
            }
            let other = other as i32;
            let ox = other % cells_side;
            let oy = other / cells_side;
            let d = (ox - x).abs() + (oy - y).abs();
            if d < best_d {
                best_d = d;
                best = other;
            }
        }
        remap[id] = best;
    }
    for cell in &mut cell_of {
        if *cell >= 0 {
            *cell = remap[*cell as usize];
        }
    }

    let mut bundle_ids: Vec<i32> = remap.iter().copied().filter(|id| *id >= 0).collect();
    bundle_ids.sort_unstable();
    bundle_ids.dedup();

    let members: Vec<Vec<usize>> = if bundle_ids.is_empty() {
        vec![(0..strand_count).collect()]
    } else {
        let mut members = vec![Vec::new(); bundle_ids.len()];
        for (strand, cell) in cell_of.iter().enumerate() {
            if let Some(index) = bundle_ids.iter().position(|b| *b == *cell) {
                members[index].push(strand);
            }
        }
        members
    };

    let mut coherent = Vec::new();
    for group in members.iter().filter(|g| g.len() >= MIN_STRANDS_PER_BUNDLE) {
        coherent.extend(split_until_coherent(strands, group));
    }

    let mut bundles = Vec::new();
    let mut corners = Vec::new();
    let mut style = Vec::new();
    let mut bounds_min = Vec3::splat(f32::MAX);
    let mut bounds_max = Vec3::splat(f32::MIN);

    for group in &coherent {
        let resampled = resample_group(strands, group);
        if resampled.is_empty() {
            continue;
        }
        let frames = frames_for(&resampled);
        let root_frame = frames[0];
        let mut layer_corners = Vec::with_capacity(LAYER_COUNT as usize);
        for frame in &frames {
            let quad = frame.quad();
            for corner in quad {
                bounds_min = bounds_min.min(corner);
                bounds_max = bounds_max.max(corner);
            }
            layer_corners.push((*frame, quad));
        }

        let layer_offset = corners.len() as u32;
        let style_offset = style.len() as u32;
        let mut style_sum = vec![Vec3::ZERO; STYLE_TEXELS as usize];
        let mut style_weight = vec![0.0f32; STYLE_TEXELS as usize];

        for strand in &resampled {
            let uv = root_frame.uv(strand[0]);
            for (layer, point) in strand.iter().enumerate() {
                let base = bilinear(layer_corners[layer].1, uv);
                let residual = *point - base;
                splat_residual(&mut style_sum, &mut style_weight, uv, layer, residual);
            }
        }

        for layer in 0..LAYER_COUNT as usize {
            let quad = layer_corners[layer].1;
            let prev = layer_corners[layer.saturating_sub(1)].1;
            let next = layer_corners[(layer + 1).min(LAYER_COUNT as usize - 1)].1;
            for corner in 0..4 {
                let tangent = if layer == 0 {
                    quad[corner] - next[corner]
                } else if layer + 1 == LAYER_COUNT as usize {
                    quad[corner] - prev[corner]
                } else {
                    (next[corner] - prev[corner]) * 0.5
                };
                // Tip difference above was reversed for the root; root should point toward the next layer.
                let tangent = if layer == 0 {
                    next[corner] - quad[corner]
                } else if layer + 1 == LAYER_COUNT as usize {
                    quad[corner] - prev[corner]
                } else {
                    tangent
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
            strand_count: group.len() as u32,
        });
    }

    if bundles.is_empty() {
        return empty_mesh();
    }
    bake_ambient_occlusion(&mut corners, &bundles);
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

#[derive(Clone, Copy)]
struct Frame {
    mean: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    ext_u: f32,
    ext_v: f32,
}

impl Frame {
    fn quad(&self) -> [Vec3; 4] {
        let u = self.axis_u * self.ext_u;
        let v = self.axis_v * self.ext_v;
        [
            self.mean - u - v,
            self.mean + u - v,
            self.mean - u + v,
            self.mean + u + v,
        ]
    }

    fn uv(&self, point: Vec3) -> Vec2 {
        let d = point - self.mean;
        let lu = d.dot(self.axis_u);
        let lv = d.dot(self.axis_v);
        Vec2::new(lu / (2.0 * self.ext_u) + 0.5, lv / (2.0 * self.ext_v) + 0.5)
    }
}

fn resample_group(strands: &HairStrands, group: &[usize]) -> Vec<Vec<Vec3>> {
    resample_indexed(strands, group)
        .into_iter()
        .map(|(_, samples)| samples)
        .collect()
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

fn cross_section_frame(strands: &[Vec<Vec3>], layer: usize) -> Frame {
    let points: Vec<Vec3> = strands.iter().map(|s| s[layer]).collect();
    let mean = average(&points);
    let mut tangent = Vec3::ZERO;
    for strand in strands {
        let next = strand[(layer + 1).min(strand.len() - 1)];
        let prev = strand[layer.saturating_sub(1)];
        tangent += next - prev;
    }
    let normal = tangent.normalize_or(Vec3::Y);
    let (axis_u, axis_v, ext_u, ext_v) = planar_extents(&points, mean, normal);
    Frame {
        mean,
        axis_u,
        axis_v,
        ext_u,
        ext_v,
    }
}

fn oriented_frame(strands: &[Vec<Vec3>], layer: usize, ref_u: Vec3, ref_v: Vec3) -> Frame {
    let points: Vec<Vec3> = strands.iter().map(|s| s[layer]).collect();
    let mean = average(&points);
    let mut tangent = Vec3::ZERO;
    for strand in strands {
        let next = strand[(layer + 1).min(strand.len() - 1)];
        let prev = strand[layer.saturating_sub(1)];
        tangent += next - prev;
    }
    let normal = tangent.normalize_or(Vec3::Y);
    // Carry the previous layer's axes into this plane. Re-using an axis that has
    // turned parallel to the strand would tilt the quad out of the cross-section.
    let (axis_u, axis_v) = transported_axes(ref_u, ref_v, normal);
    let ext_u = robust_extent(points.iter().map(|point| (*point - mean).dot(axis_u)));
    let ext_v = robust_extent(points.iter().map(|point| (*point - mean).dot(axis_v)));
    Frame {
        mean,
        axis_u,
        axis_v,
        ext_u,
        ext_v,
    }
}

fn planar_extents(points: &[Vec3], mean: Vec3, normal: Vec3) -> (Vec3, Vec3, f32, f32) {
    let mut helper = Vec3::X;
    if helper.cross(normal).length_squared() < 1e-4 {
        helper = Vec3::Y;
    }
    let axis_u0 = helper.cross(normal).normalize();
    let axis_v0 = normal.cross(axis_u0).normalize();
    let mut cxx = 0.0;
    let mut cyy = 0.0;
    let mut cxy = 0.0;
    for point in points {
        let d = *point - mean;
        let x = d.dot(axis_u0);
        let y = d.dot(axis_v0);
        cxx += x * x;
        cyy += y * y;
        cxy += x * y;
    }
    let angle = 0.5 * (2.0 * cxy).atan2(cxx - cyy);
    let (s, c) = angle.sin_cos();
    let axis_u = (axis_u0 * c + axis_v0 * s).normalize();
    let axis_v = normal.cross(axis_u).normalize();
    let ext_u = robust_extent(points.iter().map(|point| (*point - mean).dot(axis_u)));
    let ext_v = robust_extent(points.iter().map(|point| (*point - mean).dot(axis_v)));
    (axis_u, axis_v, ext_u, ext_v)
}

fn frames_for(strands: &[Vec<Vec3>]) -> Vec<Frame> {
    let mut frames = Vec::with_capacity(LAYER_COUNT as usize);
    let root = cross_section_frame(strands, 0);
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

/// High percentile of `|projection|`, so one stray strand cannot inflate the quad.
fn robust_extent(values: impl Iterator<Item = f32>) -> f32 {
    let mut values: Vec<f32> = values.map(|value| value.abs()).collect();
    if values.is_empty() {
        return 1e-3;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let rank = ((values.len() - 1) as f32 * 0.9).round() as usize;
    values[rank].max(1e-3)
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
fn worst_quad_gap(strands: &[Vec<Vec3>]) -> (usize, Vec3, f32) {
    let frames = frames_for(strands);
    let mut sum = vec![Vec3::ZERO; STYLE_TEXELS as usize];
    let mut weight = vec![0.0f32; STYLE_TEXELS as usize];
    for strand in strands {
        let uv = frames[0].uv(strand[0]);
        for (layer, point) in strand.iter().enumerate() {
            let residual = *point - bilinear(frames[layer].quad(), uv);
            splat_residual(&mut sum, &mut weight, uv, layer, residual);
        }
    }
    let mut style = vec![[0.0f32; 4]; STYLE_TEXELS as usize];
    for (texel, (accumulated, weight)) in style.iter_mut().zip(sum.iter().zip(&weight)) {
        if *weight > 0.0 {
            let residual = *accumulated / *weight;
            *texel = [residual.x, residual.y, residual.z, 0.0];
        }
    }

    let mut worst_layer = 0usize;
    let mut worst_axis = frames[0].axis_u;
    let mut worst = 0.0f32;
    for (layer, frame) in frames.iter().enumerate() {
        let quad = frame.quad();
        let mut gap = 0.0f32;
        for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for u in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let uv = Vec2::new(u, v);
                let pos = bilinear(quad, uv) + style_slice(&style, uv, layer);
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
            worst_axis = if frame.ext_u >= frame.ext_v {
                frame.axis_u
            } else {
                frame.axis_v
            };
        }
    }
    (worst_layer, worst_axis, worst)
}

fn split_until_coherent(strands: &HairStrands, group: &[usize]) -> Vec<Vec<usize>> {
    fn rec(strands: &HairStrands, group: &[usize], depth: u32) -> Vec<Vec<usize>> {
        let indexed = resample_indexed(strands, group);
        if indexed.is_empty() {
            return Vec::new();
        }
        if indexed.len() < 2 || depth >= 16 {
            return vec![indexed.into_iter().map(|(id, _)| id).collect()];
        }
        let resampled: Vec<Vec<Vec3>> =
            indexed.iter().map(|(_, samples)| samples.clone()).collect();
        let (layer, axis, gap) = worst_quad_gap(&resampled);
        if gap <= MAX_QUAD_GAP {
            return vec![indexed.into_iter().map(|(id, _)| id).collect()];
        }
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
        if span < 1e-3 || mid == 0 {
            return vec![group.to_vec()];
        }
        let left: Vec<usize> = order[..mid].iter().map(|entry| entry.1).collect();
        let right: Vec<usize> = order[mid..].iter().map(|entry| entry.1).collect();
        let mut out = rec(strands, &left, depth + 1);
        out.extend(rec(strands, &right, depth + 1));
        out
    }
    rec(strands, group, 0)
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

fn bake_ambient_occlusion(corners: &mut [CageCorner], bundles: &[BundleDesc]) {
    if corners.is_empty() {
        return;
    }
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for corner in corners.iter() {
        min = min.min(corner.position);
        max = max.max(corner.position);
    }
    let extent = (max - min).max(Vec3::splat(1.0));
    let cell = extent.max_element() / 24.0;
    let mut grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
        std::collections::HashMap::new();
    for (index, corner) in corners.iter().enumerate() {
        grid.entry(voxel(corner.position, min, cell))
            .or_default()
            .push(index);
    }

    let mut ao = vec![1.0f32; corners.len()];
    for (bundle_i, bundle) in bundles.iter().enumerate() {
        let base = bundle.layer_offset as usize;
        for layer in 0..LAYER_COUNT as usize {
            let mut center = Vec3::ZERO;
            for corner in 0..4 {
                center += corners[base + layer * 4 + corner].position;
            }
            center *= 0.25;
            let w = layer as f32 / (LAYER_COUNT - 1) as f32;
            for corner in 0..4 {
                let index = base + layer * 4 + corner;
                let position = corners[index].position;
                let inward = (center - position).normalize_or(Vec3::Y);
                let mut occupied = 0.0;
                let mut samples = 0.0;
                let key = voxel(position, min, cell);
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let neighbor = (key.0 + dx, key.1 + dy, key.2 + dz);
                            let Some(list) = grid.get(&neighbor) else {
                                continue;
                            };
                            for &other in list {
                                if other / (LAYER_COUNT as usize * 4) == bundle_i
                                    && (other - base) / 4 == layer
                                {
                                    continue;
                                }
                                let delta = corners[other].position - position;
                                let dist = delta.length();
                                if dist < 1e-4 || dist > cell * 2.5 {
                                    continue;
                                }
                                samples += 1.0;
                                if delta.dot(inward) > 0.0 {
                                    occupied += 1.0 - dist / (cell * 2.5);
                                }
                            }
                        }
                    }
                }
                let local = if samples > 0.0 {
                    1.0 - (occupied / samples).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                // Darken toward the scalp even when the voxel neighborhood is empty.
                let scalp = 0.45 + 0.55 * w;
                ao[index] = (local * scalp).clamp(0.05, 1.0);
            }
        }
    }
    for (corner, value) in corners.iter_mut().zip(ao) {
        corner.ao = value;
    }
}

fn voxel(position: Vec3, min: Vec3, cell: f32) -> (i32, i32, i32) {
    let p = (position - min) / cell;
    (p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32)
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
    use crate::hair_file::HairStrands;
    use crate::mesh::STYLE_W;

    type SegmentBins = std::collections::HashMap<(i32, i32, i32), Vec<(Vec3, Vec3)>>;

    #[test]
    fn straight_cage_residuals_are_near_zero() {
        let mut points = Vec::new();
        let mut strands = Vec::new();
        let corners = [
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(-1.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
        ];
        // Repeat the corner strands so the cell survives the occupancy threshold.
        for _ in 0..3 {
            for corner in corners {
                let start = points.len() as u32;
                for layer in 0..8 {
                    points.push(corner + Vec3::Y * layer as f32);
                }
                strands.push((start, 8));
            }
        }
        let hair = HairStrands {
            points,
            strands,
            default_color: [1.0, 1.0, 1.0],
        };
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
            stray.distance < 8.0,
            "uniform samples inside a bundle quad left the groom by {}",
            stray.distance
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
            stray.distance < 8.0,
            "a bending lock inflated its quad, stray {}",
            stray.distance
        );
    }

    #[test]
    fn baked_grooms_stay_on_the_source_strands() {
        for file in ["wStraight.hair", "wWavy.hair", "wCurly.hair"] {
            let path = format!("{}/assets/hair/{file}", env!("CARGO_MANIFEST_DIR"));
            let hair = crate::hair_file::load_hair_path(&path).unwrap_or_else(|err| {
                panic!("failed to read {path}: {err}");
            });
            let mesh = bake_hair_mesh(&hair);
            let stray = max_styled_stray(&mesh, &hair.points, &hair.strands);
            assert!(
                stray.distance < 8.0,
                "{file} leaves the source groom by {:.1} at bundle {} ({} strands) uv {:?} w {:.3} pos {:?} ({} bundles)",
                stray.distance,
                stray.bundle,
                mesh.bundles[stray.bundle].strand_count,
                stray.uv,
                stray.w,
                stray.position,
                mesh.bundles.len()
            );
        }
    }

    struct Stray {
        distance: f32,
        bundle: usize,
        uv: Vec2,
        w: f32,
        position: Vec3,
    }

    fn max_styled_stray(mesh: &HairMesh, points: &[Vec3], strands: &[(u32, u32)]) -> Stray {
        // Upper bound on the distance from styled samples to the source polylines.
        // A sample stops once it is within the acceptance distance: that bound is
        // enough to prove it did not grow a flyaway.
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

        let mut worst = Stray {
            distance: 0.0,
            bundle: 0,
            uv: Vec2::ZERO,
            w: 0.0,
            position: Vec3::ZERO,
        };
        let uvs = [0.0f32, 0.25, 0.5, 0.75, 1.0];
        for (bundle_index, bundle) in mesh.bundles.iter().enumerate() {
            for &v in &uvs {
                for &u in &uvs {
                    let uv = Vec2::new(u, v);
                    for step in 0..(LAYER_COUNT as usize * 2 - 1) {
                        let w = step as f32 / (LAYER_COUNT as f32 * 2.0 - 2.0);
                        let pos = styled_position(mesh, bundle, uv, w);
                        let nearest = nearest_segment(&bins, voxel(pos), pos, limit, cell);
                        if nearest > worst.distance {
                            worst = Stray {
                                distance: nearest,
                                bundle: bundle_index,
                                uv,
                                w,
                                position: pos,
                            };
                        }
                    }
                }
            }
        }
        worst
    }

    fn styled_position(mesh: &HairMesh, bundle: &BundleDesc, uv: Vec2, w: f32) -> Vec3 {
        let layers_f = w * (LAYER_COUNT - 1) as f32;
        let i0 = (layers_f.floor() as usize).min(LAYER_COUNT as usize - 1);
        let i1 = (i0 + 1).min(LAYER_COUNT as usize - 1);
        let t = layers_f - i0 as f32;
        let off = bundle.layer_offset as usize;
        let p0 = cage_point(mesh, off, i0, uv);
        let p1 = cage_point(mesh, off, i1, uv);
        let m0 = cage_tangent(mesh, off, i0, uv);
        let m1 = cage_tangent(mesh, off, i1, uv);
        hermite(p0, p1, m0, m1, t) + sample_style(mesh, bundle.style_offset as usize, uv, w)
    }

    fn cage_point(mesh: &HairMesh, offset: usize, layer: usize, uv: Vec2) -> Vec3 {
        let i = offset + layer * 4;
        bilinear(
            [
                mesh.corners[i].position,
                mesh.corners[i + 1].position,
                mesh.corners[i + 2].position,
                mesh.corners[i + 3].position,
            ],
            uv,
        )
    }

    fn cage_tangent(mesh: &HairMesh, offset: usize, layer: usize, uv: Vec2) -> Vec3 {
        let curr = cage_point(mesh, offset, layer, uv);
        let prev = cage_point(mesh, offset, layer.saturating_sub(1), uv);
        let next = cage_point(mesh, offset, (layer + 1).min(LAYER_COUNT as usize - 1), uv);
        let count = LAYER_COUNT as usize;
        if layer == 0 {
            let delta = next - curr;
            delta.normalize_or(Vec3::Y) * delta.length().max(1e-4)
        } else if layer + 1 >= count {
            let delta = curr - prev;
            delta.normalize_or(Vec3::Y) * delta.length().max(1e-4)
        } else {
            let delta = next - prev;
            delta.normalize_or(Vec3::Y) * (delta.length() * 0.5).max(1e-4)
        }
    }

    fn hermite(p0: Vec3, p1: Vec3, m0: Vec3, m1: Vec3, t: f32) -> Vec3 {
        let t2 = t * t;
        let t3 = t2 * t;
        p0 * (2.0 * t3 - 3.0 * t2 + 1.0)
            + m0 * (t3 - 2.0 * t2 + t)
            + p1 * (-2.0 * t3 + 3.0 * t2)
            + m1 * (t3 - t2)
    }

    fn sample_style(mesh: &HairMesh, offset: usize, uv: Vec2, w: f32) -> Vec3 {
        let u = uv.x.clamp(0.0, 1.0) * (STYLE_U - 1) as f32;
        let v = uv.y.clamp(0.0, 1.0) * (STYLE_V - 1) as f32;
        let ww = w.clamp(0.0, 1.0) * (STYLE_W - 1) as f32;
        let u0 = u.floor() as usize;
        let v0 = v.floor() as usize;
        let w0 = ww.floor() as usize;
        let u1 = (u0 + 1).min(STYLE_U as usize - 1);
        let v1 = (v0 + 1).min(STYLE_V as usize - 1);
        let w1 = (w0 + 1).min(STYLE_W as usize - 1);
        let tu = u - u0 as f32;
        let tv = v - v0 as f32;
        let tw = ww - w0 as f32;
        let at = |iu: usize, iv: usize, iw: usize| {
            let texel = mesh.style[offset + (iw * STYLE_V as usize + iv) * STYLE_U as usize + iu];
            Vec3::new(texel[0], texel[1], texel[2])
        };
        let c00 = at(u0, v0, w0).lerp(at(u1, v0, w0), tu);
        let c01 = at(u0, v1, w0).lerp(at(u1, v1, w0), tu);
        let c10 = at(u0, v0, w1).lerp(at(u1, v0, w1), tu);
        let c11 = at(u0, v1, w1).lerp(at(u1, v1, w1), tu);
        c00.lerp(c01, tv).lerp(c10.lerp(c11, tv), tw)
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
