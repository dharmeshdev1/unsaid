//! Print token ids and embeddings for lines on stdin, as JSON lines.
use std::io::BufRead;
use unsaid_embed::Embedder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: embed MODEL < lines");
    let t = std::time::Instant::now();
    let model = Embedder::load(&std::fs::read(path)?)?;
    eprintln!(
        "loaded {} in {:.0} ms",
        model.name(),
        t.elapsed().as_secs_f64() * 1e3
    );
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let text: String = serde_json::from_str(&line)?;
        let t = std::time::Instant::now();
        let doc = model.embed_document(&text, 128);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!(
            "{}",
            serde_json::json!({"ids": model.tokenize(&text, 128), "doc": doc, "query": model.embed_query(&text), "ms": ms})
        );
    }
    Ok(())
}
