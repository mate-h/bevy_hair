//! Procedural studio probe kept for unit tests.
//! The renderer samples the camera's Bevy environment cubemaps instead.

use bevy::math::{Vec2, Vec3};

pub const ENV_WIDTH: usize = 64;
pub const ENV_HEIGHT: usize = 32;
pub const ENV_MIPS: usize = 4;

#[derive(Clone, Debug)]
pub struct Probe {
    /// Irradiance spherical harmonics, RGB in xyz.
    pub sh: [Vec3; 9],
    /// Mip 0 is the sharpest lat-long. Each texel is rgba (a = 1).
    pub latlong: Vec<[f32; 4]>,
    pub mip_sizes: [(u32, u32); ENV_MIPS],
}

pub fn studio_probe() -> Probe {
    let mut mips = Vec::new();
    let mut mip_sizes = [(0u32, 0u32); ENV_MIPS];
    let mut width = ENV_WIDTH;
    let mut height = ENV_HEIGHT;
    let mut current = sample_latlong(width, height);
    for (mip, size) in mip_sizes.iter_mut().enumerate() {
        *size = (width as u32, height as u32);
        mips.extend_from_slice(&current);
        if mip + 1 == ENV_MIPS {
            break;
        }
        current = box_downsample(&current, width, height);
        width = (width / 2).max(1);
        height = (height / 2).max(1);
    }

    Probe {
        sh: project_irradiance_sh(),
        latlong: mips,
        mip_sizes,
    }
}

fn sample_latlong(width: usize, height: usize) -> Vec<[f32; 4]> {
    let mut texels = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;
            let dir = latlong_dir(u, v);
            let color = studio_env(dir);
            texels.push([color.x, color.y, color.z, 1.0]);
        }
    }
    texels
}

fn box_downsample(src: &[[f32; 4]], width: usize, height: usize) -> Vec<[f32; 4]> {
    let dst_w = (width / 2).max(1);
    let dst_h = (height / 2).max(1);
    let mut dst = vec![[0.0; 4]; dst_w * dst_h];
    for y in 0..dst_h {
        for x in 0..dst_w {
            let mut acc = [0.0; 4];
            let mut n = 0.0;
            for oy in 0..2 {
                for ox in 0..2 {
                    let sx = (x * 2 + ox).min(width - 1);
                    let sy = (y * 2 + oy).min(height - 1);
                    let s = src[sy * width + sx];
                    for c in 0..4 {
                        acc[c] += s[c];
                    }
                    n += 1.0;
                }
            }
            for channel in &mut acc {
                *channel /= n;
            }
            dst[y * dst_w + x] = acc;
        }
    }
    dst
}

pub fn studio_env(dir: Vec3) -> Vec3 {
    let dir = dir.normalize_or_zero();
    let up = dir.y.clamp(0.0, 1.0);
    let sky =
        Vec3::new(0.45, 0.55, 0.7) * (0.35 + 0.65 * up) + Vec3::new(0.18, 0.14, 0.11) * (1.0 - up);
    let sun_dir = Vec3::new(0.35, 0.82, 0.45).normalize();
    let sun = dir.dot(sun_dir).max(0.0).powf(64.0);
    let fill = dir
        .dot(Vec3::new(-0.6, 0.2, 0.4).normalize())
        .max(0.0)
        .powf(8.0);
    sky + Vec3::new(1.15, 0.95, 0.75) * sun * 6.0 + Vec3::new(0.45, 0.55, 0.8) * fill * 0.8
}

fn latlong_dir(u: f32, v: f32) -> Vec3 {
    let phi = (u - 0.5) * std::f32::consts::TAU;
    let theta = v * std::f32::consts::PI;
    let y = theta.cos();
    let r = theta.sin();
    Vec3::new(r * phi.cos(), y, r * phi.sin())
}

fn project_irradiance_sh() -> [Vec3; 9] {
    // Monte Carlo over the sphere, then apply the Lambertian cosine-kernel weights.
    const N: usize = 48;
    let mut coeff = [Vec3::ZERO; 9];
    let mut count = 0.0;
    for y in 0..N {
        for x in 0..N * 2 {
            let u = (x as f32 + 0.5) / (N * 2) as f32;
            let v = (y as f32 + 0.5) / N as f32;
            let dir = latlong_dir(u, v);
            // lat-long area element is proportional to sin(theta) = horizontal radius
            let weight = Vec2::new(dir.x, dir.z).length().max(1e-3);
            let color = studio_env(dir);
            let basis = sh_basis(dir);
            for i in 0..9 {
                coeff[i] += color * basis[i] * weight;
            }
            count += weight;
        }
    }
    let scale = std::f32::consts::TAU * 2.0 / count;
    for c in &mut coeff {
        *c *= scale;
    }
    // Ramamoorthi irradiance convolution: A0 = pi, A1 = 2pi/3, A2 = pi/4.
    let a0 = std::f32::consts::PI;
    let a1 = 2.0 * std::f32::consts::PI / 3.0;
    let a2 = std::f32::consts::PI / 4.0;
    coeff[0] *= a0;
    for c in &mut coeff[1..4] {
        *c *= a1;
    }
    for c in &mut coeff[4..9] {
        *c *= a2;
    }
    coeff
}

pub fn sh_basis(dir: Vec3) -> [f32; 9] {
    let n = dir.normalize_or_zero();
    let x = n.x;
    let y = n.y;
    let z = n.z;
    [
        0.282_095,
        0.488_603 * y,
        0.488_603 * z,
        0.488_603 * x,
        1.092_548 * x * y,
        1.092_548 * y * z,
        0.315_392 * (3.0 * z * z - 1.0),
        1.092_548 * x * z,
        0.546_274 * (x * x - y * y),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_has_finite_positive_sky() {
        let probe = studio_probe();
        assert!(probe.sh[0].x.is_finite());
        assert!(probe.sh[0].length() > 0.1);
        assert_eq!(probe.mip_sizes[0], (ENV_WIDTH as u32, ENV_HEIGHT as u32));
        assert!(probe.latlong.len() > ENV_WIDTH * ENV_HEIGHT);
    }
}
