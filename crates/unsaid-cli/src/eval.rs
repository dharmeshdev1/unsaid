//! Measure retrieval quality on a BEIR dataset, against exact search, through the real
//! protocol with real encryption. No step is simulated.

use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;
use unsaid_build::{BuildOptions, build};
use unsaid_core::Document;
use unsaid_core::client::SearchPlan;
use unsaid_embed::Embedder;

use crate::engine::Engine;

/// Options for an evaluation run.
pub struct EvalOptions {
    /// Directory holding `corpus.jsonl`, `queries.jsonl` and `qrels/test.tsv`.
    pub dataset: std::path::PathBuf,
    /// Where to write the index and cached embeddings.
    pub out: std::path::PathBuf,
    /// Model to use.
    pub model: Vec<u8>,
    /// Index options.
    pub build: BuildOptions,
    /// Probe counts to try.
    pub probes: Vec<usize>,
    /// Page counts to try.
    pub pages: Vec<usize>,
    /// Longest document input, in tokens.
    pub max_tokens: usize,
    /// Evaluate at most this many queries, for very large datasets.
    pub max_queries: Option<usize>,
}

#[derive(serde::Deserialize)]
struct CorpusRow {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    text: String,
}

#[derive(serde::Deserialize)]
struct QueryRow {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    std::io::BufReader::new(file)
        .lines()
        .map(|line| {
            let line = line.map_err(|e| e.to_string())?;
            serde_json::from_str(&line).map_err(|e| format!("{}: {e}", path.display()))
        })
        .collect()
}

/// BEIR's convention: a document that is the query itself does not count, which matters for
/// datasets such as ArguAna where queries are also documents.
fn ndcg_at(ranked: &[&str], relevant: &HashMap<String, u32>, query: &str, k: usize) -> f64 {
    let ranked = ranked.iter().filter(|d| **d != query).take(k);
    let dcg: f64 = ranked
        .enumerate()
        .map(|(i, d)| f64::from(*relevant.get(*d).unwrap_or(&0)) / ((i + 2) as f64).log2())
        .sum();
    let mut gains: Vec<u32> = relevant.values().copied().collect();
    gains.sort_unstable_by(|a, b| b.cmp(a));
    let ideal: f64 = gains
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, g)| f64::from(*g) / ((i + 2) as f64).log2())
        .sum();
    if ideal > 0.0 { dcg / ideal } else { 0.0 }
}

fn recall_at(ranked: &[&str], relevant: &HashMap<String, u32>, query: &str, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let ranked = ranked.iter().filter(|d| **d != query).take(k);
    ranked.filter(|d| relevant.contains_key(**d)).count() as f64 / relevant.len() as f64
}

fn percentile(values: &mut [f64], q: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return 0.0;
    }
    let rank = ((q / 100.0 * values.len() as f64).ceil() as usize).clamp(1, values.len());
    values[rank - 1]
}

/// What one search through the protocol scored and cost.
struct Measured {
    ndcg: f64,
    recall: f64,
    server_ms: f64,
    uploaded: u64,
    downloaded: u64,
}

/// Run the evaluation and print a table.
pub fn run(options: &EvalOptions) -> Result<(), String> {
    let corpus: Vec<CorpusRow> = read_jsonl(&options.dataset.join("corpus.jsonl"))?;
    let queries: Vec<QueryRow> = read_jsonl(&options.dataset.join("queries.jsonl"))?;
    let mut qrels: HashMap<String, HashMap<String, u32>> = HashMap::new();
    let qrel_path = options.dataset.join("qrels/test.tsv");
    let qrel_text =
        std::fs::read_to_string(&qrel_path).map_err(|e| format!("{}: {e}", qrel_path.display()))?;
    for line in qrel_text.lines().skip(1) {
        let mut parts = line.split('\t');
        if let (Some(q), Some(d), Some(s)) = (parts.next(), parts.next(), parts.next()) {
            let score: u32 = s.trim().parse().unwrap_or(0);
            if score > 0 {
                qrels
                    .entry(q.to_owned())
                    .or_default()
                    .insert(d.to_owned(), score);
            }
        }
    }
    let mut queries: Vec<QueryRow> = queries
        .into_iter()
        .filter(|q| qrels.contains_key(&q.id))
        .collect();
    if let Some(limit) = options.max_queries {
        queries.truncate(limit);
    }
    eprintln!(
        "{} documents, {} queries with judgements",
        corpus.len(),
        queries.len()
    );

    let embedder = Embedder::load(&options.model).map_err(|e| e.to_string())?;
    let dim = embedder.dim();
    let texts: Vec<String> = corpus
        .iter()
        .map(|d| {
            if d.title.is_empty() {
                d.text.clone()
            } else {
                format!("{}. {}", d.title, d.text)
            }
        })
        .collect();
    std::fs::create_dir_all(&options.out).map_err(|e| e.to_string())?;
    let cache = options
        .out
        .join(format!("embeddings.{}.f32", embedder.name()));
    let embeddings = crate::embed_corpus(&embedder, &texts, options.max_tokens, Some(&cache))?;
    let documents: Vec<Document> = corpus
        .iter()
        .map(|d| Document {
            title: d.title.clone(),
            text: d.text.clone(),
            url: d.id.clone(),
        })
        .collect();

    let started = Instant::now();
    let report = build(
        &documents,
        &embeddings,
        embedder.name(),
        dim,
        &options.model,
        &options.build,
        &options.out,
    )?;
    eprintln!(
        "index built in {:.1} s: {}",
        started.elapsed().as_secs_f64(),
        serde_json::to_string(&report).unwrap_or_default()
    );

    // Very large datasets need the memory for the rest of the run.
    drop(documents);
    drop(texts);

    let query_vectors: Vec<Vec<f32>> = queries
        .par_iter()
        .map(|q| embedder.embed_query(&q.text))
        .collect();

    // Exact search over the original embeddings is the baseline.
    let ids: Vec<&str> = corpus.iter().map(|d| d.id.as_str()).collect();
    let exact: Vec<(f64, f64)> = queries
        .par_iter()
        .zip(&query_vectors)
        .map(|(q, v)| {
            let mut scored: Vec<(f32, u32)> = embeddings
                .chunks_exact(dim)
                .enumerate()
                .map(|(i, e)| (e.iter().zip(v).map(|(a, b)| a * b).sum(), i as u32))
                .collect();
            scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let ranked: Vec<&str> = scored
                .iter()
                .take(100)
                .map(|(_, i)| ids[*i as usize])
                .collect();
            (
                ndcg_at(&ranked, &qrels[&q.id], &q.id, 10),
                recall_at(&ranked, &qrels[&q.id], &q.id, 10),
            )
        })
        .collect();
    let mean =
        |f: &dyn Fn(&(f64, f64)) -> f64| exact.iter().map(f).sum::<f64>() / exact.len() as f64;
    let (base_ndcg, base_recall) = (mean(&|x| x.0), mean(&|x| x.1));
    println!("exact search: nDCG@10 = {base_ndcg:.4}, Recall@10 = {base_recall:.4}\n");
    println!(
        "| probes | pages | nDCG@10 | share of exact | Recall@10 | upload | download | server ms (p50) |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- |");

    let engine = Engine::local(&options.out)?;
    for &probes in &options.probes {
        for &pages in &options.pages {
            let plan = SearchPlan {
                probes: probes.min(report.clusters),
                pages,
            };
            let results = queries
                .iter()
                .zip(&query_vectors)
                .map(|(q, v)| {
                    let (outcome, trace) = engine.search_embedding(v, 100, plan)?;
                    if outcome.corrupted_pages != 0 {
                        return Err(format!("{} corrupted pages", outcome.corrupted_pages));
                    }
                    let ranked: Vec<&str> = outcome
                        .hits
                        .iter()
                        .map(|h| h.document.url.as_str())
                        .collect();
                    Ok(Measured {
                        ndcg: ndcg_at(&ranked, &qrels[&q.id], &q.id, 10),
                        recall: recall_at(&ranked, &qrels[&q.id], &q.id, 10),
                        server_ms: trace.rank_ms + trace.fetch_ms,
                        uploaded: trace.uploaded,
                        downloaded: trace.downloaded,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let n = results.len() as f64;
            let ndcg = results.iter().map(|r| r.ndcg).sum::<f64>() / n;
            let recall = results.iter().map(|r| r.recall).sum::<f64>() / n;
            let mut server: Vec<f64> = results.iter().map(|r| r.server_ms).collect();
            println!(
                "| {} | {} | {:.4} | {:.1}% | {:.4} | {:.0} KB | {:.0} KB | {:.1} |",
                plan.probes,
                pages,
                ndcg,
                ndcg / base_ndcg * 100.0,
                recall,
                results[0].uploaded as f64 / 1024.0,
                results[0].downloaded as f64 / 1024.0,
                percentile(&mut server, 50.0)
            );
        }
    }
    Ok(())
}
