//! Example of quantizing Qwen2-0.5B at 4 different block sizes.

use std::path::PathBuf;
use std::time::Instant;

use memmap2::Mmap;
use safetensors::SafeTensors;

use nf4_rs::nf4::{ErrorStats, NF4Tensor};

const BLOCK_SIZES: &[usize] = &[32, 64, 128, 256];

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

fn f16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect()
}

fn f32_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

struct TensorResult {
    name: String,
    elements: usize,
    block_size: usize,
    stats: ErrorStats,
    rel_stats: ErrorStats,
    compression: f32,
    is_bias: bool,
}

fn main() {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./models/qwen2-0.5b".to_string());
    let model_dir = PathBuf::from(&model_dir);

    if !model_dir.exists() {
        panic!("Model directory not found: {}", model_dir.display());
    }

    let mut st_files: Vec<PathBuf> = std::fs::read_dir(&model_dir)
        .expect("failed to read model directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    st_files.sort();

    if st_files.is_empty() {
        panic!("No .safetensors files found in {}", model_dir.display());
    }

    println!(
        "found {} safetensors file(s) in {}",
        st_files.len(),
        model_dir.display()
    );

    let mut results: Vec<TensorResult> = Vec::new();

    for path in &st_files {
        println!("Loading {}", path.file_name().unwrap().to_string_lossy());
        let file = std::fs::File::open(path).expect("failed to open safetensors file");
        let mmap = unsafe { Mmap::map(&file) }.expect("failed to mmap file");
        let tensors = SafeTensors::deserialize(&mmap).expect("failed to parse safetensors");

        for (name, view) in tensors.tensors() {
            let weights: Vec<f32> = match view.dtype() {
                safetensors::Dtype::BF16 => bf16_to_f32(view.data()),
                safetensors::Dtype::F16 => f16_to_f32(view.data()),
                safetensors::Dtype::F32 => f32_from_bytes(view.data()),
                other => {
                    println!("Skipping {name} (dtype: {other:?})");
                    continue;
                }
            };

            if weights.len() < 32 {
                continue;
            }

            let original_bytes = weights.len() * 4;

            let is_bias = name.ends_with(".bias");

            for &bs in BLOCK_SIZES {
                let t = Instant::now();
                let tensor = NF4Tensor::quantize(&weights, bs);
                let restored = tensor.dequantize();
                let elapsed = t.elapsed();
                let stats = ErrorStats::compute(&weights, &restored);
                let rel_stats = ErrorStats::compute_relative(&weights, &restored);
                let compression = tensor.compression_ratio(original_bytes);

                if bs == BLOCK_SIZES[0] {
                    println!(
                        "  {name}: {} elements, quantized in {:.1}ms",
                        weights.len(),
                        elapsed.as_secs_f64() * 1000.0
                    );
                }

                results.push(TensorResult {
                    name: name.clone(),
                    elements: weights.len(),
                    block_size: bs,
                    stats,
                    rel_stats,
                    compression,
                    is_bias,
                });
            }
        }
    }

    // per-layer table at block_size=64
    println!(
        "{:<50} {:>10} {:>10} {:>10} {:>10} {:>10} {:>7}",
        "tensor", "elements", "mean abs", "p99 abs", "mean rel%", "p99 rel%", "ratio"
    );
    for r in results.iter().filter(|r| r.block_size == 64) {
        println!(
            "{:<50} {:>10} {:>10.6} {:>10.6} {:>9.2}% {:>9.2}% {:>6.1}x",
            truncate(&r.name, 50),
            r.elements,
            r.stats.mean_abs,
            r.stats.p99_abs,
            r.rel_stats.mean_abs * 100.0,
            r.rel_stats.p99_abs * 100.0,
            r.compression,
        );
    }

    // block size comparison, weights vs biases
    for (label, filter_bias) in [("WEIGHTS", false), ("BIASES", true)] {
        let filtered: Vec<&TensorResult> = results
            .iter()
            .filter(|r| r.is_bias == filter_bias)
            .collect();
        if filtered.is_empty() {
            continue;
        }
        println!("\n{label}");
        println!(
            "{:>10} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "block size", "mean abs", "p99 abs", "mean rel%", "p99 rel%", "compression"
        );

        for &bs in BLOCK_SIZES {
            let bs_results: Vec<&&TensorResult> =
                filtered.iter().filter(|r| r.block_size == bs).collect();
            let n = bs_results.len() as f32;
            if n == 0.0 {
                continue;
            }
            let avg_mean = bs_results.iter().map(|r| r.stats.mean_abs).sum::<f32>() / n;
            let avg_p99 = bs_results.iter().map(|r| r.stats.p99_abs).sum::<f32>() / n;
            let avg_rel_mean = bs_results.iter().map(|r| r.rel_stats.mean_abs).sum::<f32>() / n;
            let avg_rel_p99 = bs_results.iter().map(|r| r.rel_stats.p99_abs).sum::<f32>() / n;
            let avg_comp = bs_results.iter().map(|r| r.compression).sum::<f32>() / n;
            println!(
                "{:>10} {:>12.6} {:>12.6} {:>11.2}% {:>11.2}% {:>11.1}x",
                bs,
                avg_mean,
                avg_p99,
                avg_rel_mean * 100.0,
                avg_rel_p99 * 100.0,
                avg_comp
            );
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}", &s[s.len() - max + 3..])
    }
}
