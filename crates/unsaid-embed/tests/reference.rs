//! Compare with the reference tokenizer and model, on inputs chosen to be awkward.
//!
//! `fixtures/reference.json` was produced with Hugging Face `tokenizers` and PyTorch. These
//! tests need a packed model, found through `UNSAID_MODEL_DIR` or the user cache, and are
//! skipped when there is none, so a plain checkout can still run the rest of the suite.

use std::path::PathBuf;

use unsaid_embed::Embedder;

fn fixtures() -> serde_json::Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/reference.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn load(name: &str) -> Option<Embedder> {
    let dirs = [
        std::env::var_os("UNSAID_MODEL_DIR").map(PathBuf::from),
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/unsaid/models")),
    ];
    for dir in dirs.into_iter().flatten() {
        let path = dir.join(format!("{name}.usm"));
        if let Ok(bytes) = std::fs::read(&path) {
            return Some(Embedder::load(&bytes).unwrap());
        }
    }
    eprintln!("skipping: no packed model {name} found");
    None
}

fn cosine(a: &[f32], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| f64::from(*x) * y).sum()
}

fn check(name: &str) {
    let Some(model) = load(name) else { return };
    let all = fixtures();
    let texts: Vec<&str> = all["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    let reference = &all["models"][name];
    let numbers = |v: &serde_json::Value| -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    };
    for (i, text) in texts.iter().enumerate() {
        let want: Vec<u32> = reference["ids"][i]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            model.tokenize(text, 128),
            want,
            "tokens differ for {text:?}"
        );

        let doc = model.embed_document(text, 128);
        let c = cosine(&doc, &numbers(&reference["doc"][i]));
        // Packed weights are 8-bit, so this is close to the reference rather than equal.
        assert!(c > 0.99, "document cosine {c} for {text:?}");

        let query = model.embed_query(text);
        let c = cosine(&query, &numbers(&reference["query"][i]));
        assert!(c > 0.99, "query cosine {c} for {text:?}");
        let norm: f32 = query.iter().map(|v| v * v).sum();
        assert!((norm - 1.0).abs() < 1e-4);
    }
}

#[test]
fn matches_the_reference_for_all_minilm_l6_v2() {
    check("all-MiniLM-L6-v2");
}

#[test]
fn matches_the_reference_for_bge_small_en_v1_5() {
    check("bge-small-en-v1.5");
}

#[test]
fn related_text_scores_higher_than_unrelated_text() {
    let Some(model) = load("all-MiniLM-L6-v2") else {
        return;
    };
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let q = model.embed_query("symptoms of the flu");
    let related = model.embed_document(
        "Influenza causes fever, chills, cough and muscle aches.",
        128,
    );
    let unrelated = model.embed_document("The stock exchange closed higher on Tuesday.", 128);
    assert!(dot(&q, &related) > dot(&q, &unrelated) + 0.2);
}

#[test]
fn embeddings_are_deterministic() {
    let Some(model) = load("all-MiniLM-L6-v2") else {
        return;
    };
    assert_eq!(
        model.embed_query("same input"),
        model.embed_query("same input")
    );
}
