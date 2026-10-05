//! Cem Yuksel's binary HAIR strand format.
//!
//! <https://www.cemyuksel.com/research/hairmodels/>

use std::io;

use bevy::math::Vec3;

#[derive(Clone, Debug)]
pub struct HairStrands {
    pub points: Vec<Vec3>,
    /// `(first point index, point count)` per strand.
    pub strands: Vec<(u32, u32)>,
    pub default_color: [f32; 3],
}

#[derive(Debug)]
pub enum HairError {
    Io(io::Error),
    Magic,
    Truncated,
    MissingPoints,
}

impl std::fmt::Display for HairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HairError::Io(err) => write!(f, "{err}"),
            HairError::Magic => write!(f, "not a HAIR file"),
            HairError::Truncated => write!(f, "HAIR file ended early"),
            HairError::MissingPoints => write!(f, "HAIR file has no point array"),
        }
    }
}

impl std::error::Error for HairError {}

impl From<io::Error> for HairError {
    fn from(value: io::Error) -> Self {
        HairError::Io(value)
    }
}

pub fn load_hair_path(path: impl AsRef<std::path::Path>) -> Result<HairStrands, HairError> {
    let bytes = std::fs::read(path)?;
    parse_hair(&bytes)
}

pub fn parse_hair(bytes: &[u8]) -> Result<HairStrands, HairError> {
    if bytes.len() < 128 || &bytes[0..4] != b"HAIR" {
        return Err(HairError::Magic);
    }
    let strand_count = read_u32(bytes, 4) as usize;
    let point_count = read_u32(bytes, 8) as usize;
    let flags = read_u32(bytes, 12);
    let default_segments = read_u32(bytes, 16);
    let default_color = [
        read_f32(bytes, 28),
        read_f32(bytes, 32),
        read_f32(bytes, 36),
    ];
    let has_segments = flags & 1 != 0;
    let has_points = flags & 2 != 0;
    let has_thickness = flags & 4 != 0;
    let has_transparency = flags & 8 != 0;
    let has_color = flags & 16 != 0;
    if !has_points {
        return Err(HairError::MissingPoints);
    }

    let mut cursor = 128usize;
    let mut segment_counts = vec![default_segments; strand_count];
    if has_segments {
        let bytes_needed = strand_count * 2;
        let raw = read_slice(bytes, &mut cursor, bytes_needed)?;
        for (i, chunk) in raw.chunks_exact(2).enumerate() {
            segment_counts[i] = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
        }
    }

    let point_bytes = read_slice(bytes, &mut cursor, point_count * 12)?;
    let mut points = Vec::with_capacity(point_count);
    for chunk in point_bytes.chunks_exact(12) {
        points.push(Vec3::new(
            f32::from_le_bytes(chunk[0..4].try_into().unwrap()),
            f32::from_le_bytes(chunk[4..8].try_into().unwrap()),
            f32::from_le_bytes(chunk[8..12].try_into().unwrap()),
        ));
    }

    // Optional arrays are skipped; appearance comes from the groom component.
    let mut skip = 0usize;
    if has_thickness {
        skip += point_count * 4;
    }
    if has_transparency {
        skip += point_count * 4;
    }
    if has_color {
        skip += point_count * 12;
    }
    let _ = skip;

    let mut strands = Vec::with_capacity(strand_count);
    let mut start = 0u32;
    for segments in segment_counts {
        let count = segments.saturating_add(1);
        if start as usize + count as usize > points.len() {
            return Err(HairError::Truncated);
        }
        strands.push((start, count));
        start += count;
    }

    Ok(HairStrands {
        points,
        strands,
        default_color,
    })
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_slice<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], HairError> {
    let end = cursor.checked_add(len).ok_or(HairError::Truncated)?;
    if end > bytes.len() {
        return Err(HairError::Truncated);
    }
    let slice = &bytes[*cursor..end];
    *cursor = end;
    Ok(slice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_header_and_default_segments() {
        let mut bytes = vec![0u8; 128 + 6 * 4];
        bytes[0..4].copy_from_slice(b"HAIR");
        bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&2u32.to_le_bytes()); // points bit
        bytes[16..20].copy_from_slice(&1u32.to_le_bytes()); // 1 segment => 2 points
        write_f32(&mut bytes, 28, 0.2);
        write_f32(&mut bytes, 32, 0.1);
        write_f32(&mut bytes, 36, 0.05);
        write_f32(&mut bytes, 128, 0.0);
        write_f32(&mut bytes, 132, 1.0);
        write_f32(&mut bytes, 136, 2.0);
        write_f32(&mut bytes, 140, 0.0);
        write_f32(&mut bytes, 144, 2.0);
        write_f32(&mut bytes, 148, 2.0);
        let hair = parse_hair(&bytes).unwrap();
        assert_eq!(hair.strands, vec![(0, 2)]);
        assert_eq!(hair.points[1], Vec3::new(0.0, 2.0, 2.0));
        assert!((hair.default_color[0] - 0.2).abs() < 1e-6);
    }

    fn write_f32(bytes: &mut [u8], offset: usize, value: f32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
}
