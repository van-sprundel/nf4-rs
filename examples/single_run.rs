use nf4_rs::nf4::*;

fn main() {
    let weights = vec![-0.89, 0.16, 0.08, -0.13, 0.16, -0.54];

    let block = NF4Block::quantize(&weights);
    let restored = block.dequantize();

    let fmt_vec = |v: &[f32]| -> String {
        let parts: Vec<String> = v.iter().map(|x| format!("{x:+.4}")).collect();
        format!("[{}]", parts.join(", "))
    };
    println!("Original: {}", fmt_vec(&weights));
    println!("Restored: {}", fmt_vec(&restored));
    println!(
        "Storage: {} bytes (vs {} for f32)",
        block.size_bytes(),
        weights.len() * 4,
    );
    println!();

    for (orig, deq) in weights.iter().zip(restored.iter()) {
        let err = ((deq - orig) / orig).abs() * 100.0;
        println!("{orig:+.4} -> {deq:+.4}  (error {err:5.1}%)");
    }
}
