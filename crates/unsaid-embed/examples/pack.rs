//! Pack a Hugging Face model directory into the Unsaid model format.
//!
//!     cargo run --release -p unsaid-embed --example pack -- DIR OUT NAME cls|mean f32|q16|q8 [QUERY_PREFIX]
use unsaid_embed::{PackOptions, Pooling, WeightFormat, pack_safetensors};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (dir, out, name, pooling) = (&args[1], &args[2], &args[3], &args[4]);
    let safetensors = std::fs::read(format!("{dir}/model.safetensors"))?;
    let config = std::fs::read_to_string(format!("{dir}/config.json"))?;
    let vocab = std::fs::read_to_string(format!("{dir}/vocab.txt"))?;
    let options = PackOptions {
        name: name.clone(),
        pooling: if pooling == "cls" {
            Pooling::Cls
        } else {
            Pooling::Mean
        },
        query_prefix: args.get(6).cloned().unwrap_or_default(),
        document_prefix: String::new(),
        max_tokens: 512,
        weights: match args[5].as_str() {
            "f32" => WeightFormat::F32,
            "q16" => WeightFormat::Q16,
            _ => WeightFormat::Q8,
        },
    };
    let packed = pack_safetensors(&safetensors, &config, &vocab, &options)?;
    std::fs::write(out, &packed)?;
    println!("{out}: {:.1} MB", packed.len() as f64 / 1e6);
    Ok(())
}
