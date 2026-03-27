//! An impl of nf4 in Rust.
//! See: https://ngrok.com/blog/quantization
//!
//! Some key take-aways of the article is that NaN and Infinity are useless when you're quantizing.
//! NF4 is an attempt at making a small 4-bit data type that contains values that are *close* enough to the real model data.
//!

/// The 16 values below are the CDF of N(0,1) divided by 16.
/// CDF^-1(i/16) to CDF^-1((i+1)/16)
///
/// The midpoint being CDF^-1((i + 0.5) / 16).
const NF4_LUT: [f32; 16] = [
    -1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0000, 0.0796, 0.1609, 0.2461,
    0.3379, 0.4407, 0.5626, 0.7230, 1.0000,
];

/// a single nf4 value
/// only the low 4 bits are used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct NF4(u8);

impl NF4 {
    pub fn from_index(index: u8) -> Self {
        assert!(index < 16, "NF4 index must be 0..15, got {index}");
        Self(index)
    }

    /// Dequantize to f32
    pub fn to_f32(self) -> f32 {
        NF4_LUT[self.0 as usize]
    }

    /// Quantize a normalized f32
    pub fn quantize(value: f32) -> Self {
        let clamped = value.clamp(-1.0, 1.0);
        let mut best = 0u8;
        let mut best_dist = f32::MAX;
        for (i, &lut_val) in NF4_LUT.iter().enumerate() {
            let dist = (clamped - lut_val).abs();
            if dist < best_dist {
                best_dist = dist;
                best = i as u8;
            }
        }
        Self(best)
    }
}

// each block has a shared scale factor (absmax of the original weights) and packs two NF4 values per byte.
#[derive(Debug, Clone)]
pub struct NF4Block {
    // absmax of the original f32 weights in this block.
    pub scale: f32,
    // data[i] holds two nf4 codes (low nibble + high nibble).
    pub data: Vec<u8>, // nibbles
    // there's a chance block_size is odd, so we need to know the number of logical values.
    pub len: usize,
}

impl NF4Block {
    pub fn quantize(weights: &[f32]) -> Self {
        let absmax = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);

        // avoid division by zero for all-zero blocks.
        let scale = if absmax == 0.0 { 1.0 } else { absmax };

        let codes: Vec<NF4> = weights.iter().map(|&w| NF4::quantize(w / scale)).collect();

        // since the data type is 4 bits, we can pack two together into a byte
        let mut data = Vec::with_capacity(codes.len().div_ceil(2));
        for pair in codes.chunks(2) {
            let lo = pair[0].0;
            let hi = if pair.len() > 1 { pair[1].0 } else { 0 };
            data.push(lo | (hi << 4));
        }

        Self {
            scale,
            data,
            len: weights.len(),
        }
    }

    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len);
        for &byte in &self.data {
            let lo = NF4(byte & 0x0F);
            out.push(lo.to_f32() * self.scale);
            if out.len() < self.len {
                let hi = NF4(byte >> 4);
                out.push(hi.to_f32() * self.scale);
            }
        }
        out
    }

    pub fn size_bytes(&self) -> usize {
        4 + self.data.len()
    }
}

pub struct ErrorStats {
    pub mean_abs: f32,
    pub max_abs: f32,
    pub p99_abs: f32,
}

impl ErrorStats {
    pub fn compute(original: &[f32], restored: &[f32]) -> Self {
        let mut errors: Vec<f32> = original
            .iter()
            .zip(restored.iter())
            .map(|(a, b)| (a - b).abs())
            .collect();
        let n = errors.len();
        let mean_abs = errors.iter().sum::<f32>() / n as f32;
        errors.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let max_abs = errors[n - 1];
        let p99_abs = errors[(n as f64 * 0.99) as usize];
        Self {
            mean_abs,
            max_abs,
            p99_abs,
        }
    }

    /// relative error: |orig - restored| / |orig|
    // skipping near-zero original
    pub fn compute_relative(original: &[f32], restored: &[f32]) -> Self {
        let eps = 1e-10;
        let mut errors: Vec<f32> = original
            .iter()
            .zip(restored.iter())
            .filter(|(a, _)| a.abs() > eps)
            .map(|(a, b)| ((a - b) / a).abs())
            .collect();
        let n = errors.len();
        if n == 0 {
            return Self {
                mean_abs: 0.0,
                max_abs: 0.0,
                p99_abs: 0.0,
            };
        }
        let mean_abs = errors.iter().sum::<f32>() / n as f32;
        errors.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let max_abs = errors[n - 1];
        let p99_abs = errors[((n as f64 * 0.99) as usize).min(n - 1)];
        Self {
            mean_abs,
            max_abs,
            p99_abs,
        }
    }
}

impl std::fmt::Display for ErrorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mean={:.6} max={:.6} p99={:.6}",
            self.mean_abs, self.max_abs, self.p99_abs
        )
    }
}

// the paper talks about double-quantizing, using larger blocks.
// i want to know which block size has the lowest error rate
// and even that doesnt really tell me much...
pub struct NF4Tensor {
    pub block_size: usize,
    pub blocks: Vec<NF4Block>,
    pub len: usize,
}

impl NF4Tensor {
    pub fn quantize(weights: &[f32], block_size: usize) -> Self {
        let blocks = weights.chunks(block_size).map(NF4Block::quantize).collect();

        Self {
            block_size,
            blocks,
            len: weights.len(),
        }
    }

    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len);
        for block in &self.blocks {
            out.extend(block.dequantize());
        }
        out
    }

    pub fn size_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.size_bytes()).sum()
    }

    pub fn compression_ratio(&self, original_bytes: usize) -> f32 {
        (original_bytes as f32) / (self.size_bytes() as f32)
    }
}
