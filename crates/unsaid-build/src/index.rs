//! The build pipeline: from documents and embeddings to the files a server publishes.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use rayon::prelude::*;
use sha2::{Digest, Sha256};
use unsaid_core::Document;
use unsaid_core::client::PAGE_DIGEST_BYTES;
use unsaid_core::manifest::{BlobInfo, FORMAT, LweParams, MatrixInfo, index_id, sha256_hex};
use unsaid_core::{Manifest, pages};
use unsaid_lwe::noise::failure_log2;
use unsaid_lwe::params::MAX_FAILURE_LOG2;
use unsaid_lwe::{LWE_DIM, Matrix};

use crate::cluster::{balanced_assign, kmeans, neighbour_chain, prototypes};
use crate::linalg::{Pca, dot};

/// How an index is built. `None` means choose automatically from the corpus size.
#[derive(Clone, Debug)]
pub struct BuildOptions {
    /// Display name.
    pub name: String,
    /// Length of vectors after projection. Higher is more accurate and costs more traffic.
    pub dim: usize,
    /// Largest number of documents in a cluster, which sets the ranking hint's size.
    pub rows: Option<usize>,
    /// Bytes in a page, which sets the document hint's size.
    pub page_bytes: Option<usize>,
    /// Clusters scored per search by default.
    pub probes: usize,
    /// Pages retrieved per search by default.
    pub pages: usize,
    /// Routing prototypes per cluster.
    pub prototypes: usize,
    /// Longest text stored for one document, in bytes.
    pub max_text_bytes: usize,
    /// Seed for clustering. Changing it changes the index.
    pub seed: u64,
}

impl BuildOptions {
    /// Defaults measured to give close to exact search quality.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dim: 256,
            rows: None,
            page_bytes: None,
            probes: 8,
            pages: 4,
            prototypes: 32,
            max_text_bytes: 600,
            seed: 0x756e_7361_6964,
        }
    }
}

/// What a build produced.
#[derive(Clone, Debug, serde::Serialize)]
pub struct BuildReport {
    /// Documents indexed.
    pub documents: usize,
    /// Clusters created.
    pub clusters: usize,
    /// Rows of the ranking matrix.
    pub rank_rows: usize,
    /// Columns of the ranking matrix.
    pub rank_cols: usize,
    /// Pages in the document matrix.
    pub pages: usize,
    /// Bytes per page.
    pub page_bytes: usize,
    /// Share of embedding variance kept by the projection.
    pub explained_variance: f64,
    /// Log of the ranking plaintext modulus.
    pub rank_log_p: u32,
    /// Scale applied to vectors before rounding.
    pub scale: f32,
    /// Log2 bound on a ranking answer containing an error.
    pub rank_failure_log2: f64,
    /// Log2 bound on a page answer containing an error.
    pub payload_failure_log2: f64,
    /// Files a client downloads once, in bytes.
    pub public_bytes: u64,
    /// Files only the server holds, in bytes.
    pub private_bytes: u64,
    /// Bytes uploaded per search with the default plan.
    pub upload_per_search: u64,
    /// Bytes downloaded per search with the default plan.
    pub download_per_search: u64,
    /// Identifier of the whole index.
    pub index_id: String,
}

/// Smallest page allowed. Below a few bytes not even an empty document fits.
const MIN_PAGE_BYTES: usize = 64;

/// Nearest power of two, within limits.
fn nearest_power_of_two(value: f64, low: usize, high: usize) -> usize {
    let exponent = value.max(1.0).log2().round() as u32;
    (1usize << exponent).clamp(low, high)
}

/// Balances the one-time hint against per-search traffic, assuming a session of ten
/// searches. Ranking rows only pay for the hint, while more rows mean fewer clusters.
fn automatic_rows(documents: usize, dim: usize, probes: usize) -> usize {
    let per_doc_upload = (dim * probes * 4) as f64;
    let optimum = (documents as f64 * per_doc_upload * 10.0 / (LWE_DIM * 4) as f64 / 0.8).sqrt();
    nearest_power_of_two(optimum, 64, 16384)
}

/// Same balance for pages, with a floor. A larger page holds several neighbouring documents, so
/// one fetch returns more of the good results: on BEIR SciFact, four pages of 2048 bytes matched
/// eight pages of 512 bytes while sending about a third less per search.
fn automatic_page_bytes(total_compressed: f64, pages: usize) -> usize {
    let optimum = (total_compressed * (pages * 4 * 10) as f64 / (LWE_DIM * 4) as f64).sqrt();
    nearest_power_of_two(optimum, 2048, 16384)
}

fn truncate_utf8(text: &str, limit: usize) -> &str {
    if text.len() <= limit {
        return text;
    }
    let mut end = limit;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Shorten a passage for display so that it ends on a whole word and says that it was cut.
fn shorten_passage(text: &str, limit: usize) -> String {
    const ELLIPSIS: &str = "\u{2026}";
    if text.len() <= limit {
        return text.to_owned();
    }
    let cut = truncate_utf8(text, limit.saturating_sub(ELLIPSIS.len()));
    let ends_on_word = text[cut.len()..].starts_with(char::is_whitespace);
    let kept = match cut.rfind(char::is_whitespace) {
        Some(at) if !ends_on_word && at * 2 > cut.len() => &cut[..at],
        _ => cut,
    };
    format!("{}{ELLIPSIS}", kept.trim_end())
}

/// Pack documents into pages of `size` bytes, in order. A document too large for a page on
/// its own has its text shortened until it fits, so every document is always findable.
/// Returns each page with the number of documents it holds.
fn paginate(docs: &[Document], size: usize) -> Vec<(Vec<u8>, usize)> {
    let mut out = Vec::new();
    let mut current: Vec<Document> = Vec::new();
    for doc in docs {
        let mut doc = doc.clone();
        current.push(doc.clone());
        let refs: Vec<&Document> = current.iter().collect();
        if pages::encode(&refs, size).is_some() {
            continue;
        }
        current.pop();
        if !current.is_empty() {
            let refs: Vec<&Document> = current.iter().collect();
            let page = pages::encode(&refs, size).expect("this page already fitted");
            out.push((page, current.len()));
            current.clear();
        }
        while pages::encode(&[&doc], size).is_none() {
            if doc.text.is_empty() {
                doc.url.clear();
                let title_len = doc.title.len() * 3 / 4;
                doc.title = truncate_utf8(&doc.title, title_len).to_owned();
            } else {
                let keep = doc.text.len() * 3 / 4;
                doc.text = truncate_utf8(&doc.text, keep).to_owned();
            }
        }
        current.push(doc);
    }
    if !current.is_empty() {
        let refs: Vec<&Document> = current.iter().collect();
        let page = pages::encode(&refs, size).expect("this page already fitted");
        out.push((page, current.len()));
    }
    out
}

/// A failure bound that JSON can represent. A matrix of zeros cannot fail at all, and its
/// bound of minus infinity has no JSON form.
fn storable(bound: f64) -> f64 {
    bound.max(-1e9)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn blob(data: &[u8]) -> BlobInfo {
    BlobInfo {
        sha256: sha256_hex(data),
        bytes: data.len() as u64,
    }
}

/// Build an index and write it to `out`.
///
/// `embeddings` holds `documents.len()` rows of `model_dim` values from the model in
/// `model_file`. The output directory is created if needed.
pub fn build(
    documents: &[Document],
    embeddings: &[f32],
    model_name: &str,
    model_dim: usize,
    model_file: &[u8],
    options: &BuildOptions,
    out: &Path,
) -> Result<BuildReport, String> {
    let n = documents.len();
    if n == 0 {
        return Err("there are no documents to index".into());
    }
    if model_dim == 0 || embeddings.len() != n * model_dim {
        return Err(format!(
            "expected {} embedding values, found {}",
            n * model_dim,
            embeddings.len()
        ));
    }
    if embeddings.iter().any(|v| !v.is_finite()) {
        return Err("embeddings contain a non-finite value".into());
    }
    let dim = options.dim.min(model_dim).max(1);

    // Project into the smaller space.
    let pca = Pca::fit(embeddings, model_dim, dim, 50_000);
    let mut z = vec![0f32; n * dim];
    z.par_chunks_mut(dim)
        .zip(embeddings.par_chunks(model_dim))
        .for_each(|(o, e)| pca.project(e, o));

    // Cluster, with a cap on how many documents each cluster may take.
    let rows_cap = options
        .rows
        .unwrap_or_else(|| automatic_rows(n, dim, options.probes))
        .clamp(1, n.max(1));
    let clusters = (n as f64 / (rows_cap as f64 * 0.8)).ceil().max(1.0) as usize;
    let clusters = clusters.clamp(n.div_ceil(rows_cap), n);
    let centres = if clusters == 1 {
        let mut mean = vec![0f32; dim];
        for p in z.chunks_exact(dim) {
            mean.iter_mut().zip(p).for_each(|(a, b)| *a += b);
        }
        crate::linalg::normalize(&mut mean);
        mean
    } else {
        let step = (n / (clusters * 256)).max(1);
        let sample: Vec<f32> = z
            .chunks_exact(dim)
            .step_by(step)
            .flatten()
            .copied()
            .collect();
        kmeans(&sample, dim, clusters, 20, options.seed)
    };
    let assign = balanced_assign(&z, dim, &centres, rows_cap);
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); clusters];
    for (i, &c) in assign.iter().enumerate() {
        members[c as usize].push(i as u32);
    }

    // Order each cluster so similar documents share pages, and summarise it for routing.
    let results: Vec<(Vec<u32>, Vec<f32>)> = members
        .par_iter()
        .enumerate()
        .map(|(c, m)| {
            let centre = &centres[c * dim..(c + 1) * dim];
            let chain = neighbour_chain(&z, dim, m, centre);
            let protos = prototypes(&z, dim, m, options.prototypes, options.seed ^ c as u64);
            (chain, protos)
        })
        .collect();
    let mut order: Vec<u32> = Vec::with_capacity(n);
    let mut sizes: Vec<u32> = Vec::with_capacity(clusters);
    let mut proto_values: Vec<f32> = Vec::new();
    let mut owners: Vec<u32> = Vec::new();
    for (c, (chain, protos)) in results.iter().enumerate() {
        sizes.push(chain.len() as u32);
        order.extend_from_slice(chain);
        owners.extend(std::iter::repeat_n(c as u32, protos.len() / dim));
        proto_values.extend_from_slice(protos);
    }
    let rank_rows = sizes.iter().copied().max().unwrap_or(1).max(1) as usize;
    let rank_cols = clusters * dim;

    // Choose the largest quantisation scale whose noise budget works out. Retrieval quality
    // barely changes with scale, so this is about fitting the scores into the plaintext.
    let mut starts = vec![0usize; clusters + 1];
    for c in 0..clusters {
        starts[c + 1] = starts[c] + sizes[c] as usize;
    }
    let mut chosen = None;
    let mut last_error = String::new();
    for scale in [64f32, 48.0, 32.0, 24.0, 16.0, 12.0, 8.0] {
        let mut data = vec![0i8; rank_rows * rank_cols];
        data.par_chunks_mut(rank_cols.max(1))
            .enumerate()
            .for_each(|(row, row_data)| {
                for c in 0..clusters {
                    if row < sizes[c] as usize {
                        let doc = order[starts[c] + row] as usize;
                        let source = &z[doc * dim..(doc + 1) * dim];
                        let target = &mut row_data[c * dim..(c + 1) * dim];
                        for (t, &v) in target.iter_mut().zip(source) {
                            *t = (v * scale).round().clamp(-127.0, 127.0) as i8;
                        }
                    }
                }
            });
        // Largest possible score, by Cauchy-Schwarz, with rounding of the query included.
        let max_doc_norm = order
            .par_chunks(1024)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|&doc| {
                        let doc = doc as usize;
                        z[doc * dim..(doc + 1) * dim]
                            .iter()
                            .map(|v| {
                                let q = (v * scale).round().clamp(-127.0, 127.0);
                                f64::from(q * q)
                            })
                            .sum::<f64>()
                    })
                    .fold(0f64, f64::max)
            })
            .reduce(|| 0.0, f64::max)
            .sqrt();
        let max_query_norm = f64::from(scale) + (dim as f64).sqrt() / 2.0;
        let bound = max_doc_norm * max_query_norm;
        let log_p = ((2.0 * bound + 2.0).log2().ceil() as u32).max(2);
        let matrix = Matrix::new(rank_rows, rank_cols, data);
        let failure = storable(failure_log2(log_p, matrix.max_row_norm_sq(), rank_rows));
        if failure <= MAX_FAILURE_LOG2 && log_p <= 24 {
            chosen = Some((scale, log_p, failure, matrix));
            break;
        }
        last_error =
            format!("scale {scale} needs p = 2^{log_p} and has failure bound 2^{failure:.1}");
    }
    let (scale, log_p, rank_failure, rank_matrix) = chosen.ok_or_else(|| {
        format!(
            "no quantisation fits the noise budget ({last_error}). Use fewer clusters by \
             raising --rows, or lower --dim"
        )
    })?;

    // Store each document's text, in the same order as the ranking matrix.
    let ordered: Vec<Document> = order
        .par_iter()
        .map(|&i| {
            let d = &documents[i as usize];
            Document {
                title: d.title.clone(),
                text: shorten_passage(&d.text, options.max_text_bytes),
                url: d.url.clone(),
            }
        })
        .collect();
    let sample_compressed: f64 = {
        let step = (n / 2000).max(1);
        let sample: Vec<&Document> = ordered.iter().step_by(step).collect();
        let chunks: Vec<usize> = sample.chunks(8).map(miniz_estimate).collect();
        chunks.iter().sum::<usize>() as f64 * step as f64
    };
    let page_bytes = options
        .page_bytes
        .unwrap_or_else(|| automatic_page_bytes(sample_compressed, options.pages));
    if page_bytes < MIN_PAGE_BYTES {
        return Err(format!("a page must hold at least {MIN_PAGE_BYTES} bytes"));
    }

    // Pages never cross a cluster boundary, so a cluster's best documents stay together.
    let by_cluster: Vec<Vec<(Vec<u8>, usize)>> = (0..clusters)
        .into_par_iter()
        .map(|c| paginate(&ordered[starts[c]..starts[c + 1]], page_bytes))
        .collect();
    let mut page_first: Vec<u32> = Vec::new();
    let mut page_data: Vec<Vec<u8>> = Vec::new();
    let mut doc_cursor = 0u32;
    for pages_of_cluster in by_cluster {
        for (page, count) in pages_of_cluster {
            page_first.push(doc_cursor);
            page_data.push(page);
            doc_cursor += count as u32;
        }
    }
    debug_assert_eq!(doc_cursor as usize, n);
    let page_count = page_data.len();
    let mut page_digests = Vec::with_capacity(page_count * PAGE_DIGEST_BYTES);
    for page in &page_data {
        page_digests.extend_from_slice(&Sha256::digest(page)[..PAGE_DIGEST_BYTES]);
    }

    // A page is a column of the document matrix.
    let mut payload = vec![0i8; page_bytes * page_count];
    payload
        .par_chunks_mut(page_count)
        .enumerate()
        .for_each(|(row, values)| {
            for (v, page) in values.iter_mut().zip(&page_data) {
                *v = page[row] as i8;
            }
        });
    drop(page_data);
    let payload_matrix = Matrix::new(page_bytes, page_count, payload);
    let payload_failure = storable(failure_log2(
        8,
        payload_matrix.max_row_norm_sq(),
        page_bytes,
    ));
    if payload_failure > MAX_FAILURE_LOG2 {
        return Err(format!(
            "the document matrix has failure bound 2^{payload_failure:.1}, over the limit of \
             2^{MAX_FAILURE_LOG2}. Use larger pages with --page-bytes"
        ));
    }

    // Everything public, then the manifest, and only then the hints, because the public
    // matrices are derived from the hash of everything else.
    let mut routing = Vec::new();
    routing.extend(f32_bytes(&pca.mean));
    routing.extend(f32_bytes(&pca.components));
    routing.extend(f32_bytes(&proto_values));
    routing.extend(u32_bytes(&owners));
    let mut layout = u32_bytes(&sizes);
    layout.extend(u32_bytes(&page_first));

    let mut blobs = BTreeMap::new();
    blobs.insert("model".to_owned(), blob(model_file));
    blobs.insert("routing".to_owned(), blob(&routing));
    blobs.insert("layout".to_owned(), blob(&layout));
    blobs.insert("page_digests".to_owned(), blob(&page_digests));
    let placeholder = BlobInfo {
        sha256: String::new(),
        bytes: 0,
    };
    blobs.insert("rank_hint".to_owned(), placeholder.clone());
    blobs.insert("payload_hint".to_owned(), placeholder);

    let mut manifest = Manifest {
        format: FORMAT.to_owned(),
        name: options.name.clone(),
        documents: n as u64,
        model: model_name.to_owned(),
        model_dim,
        dim,
        clusters,
        prototypes: owners.len(),
        doc_scale: scale,
        query_scale: scale,
        probes: options.probes.clamp(1, clusters.min(64)),
        pages: options.pages.clamp(1, 64),
        rank: MatrixInfo {
            rows: rank_rows,
            cols: rank_cols,
            log_p,
            failure_log2: rank_failure,
        },
        payload: MatrixInfo {
            rows: page_bytes,
            cols: page_count,
            log_p: 8,
            failure_log2: payload_failure,
        },
        lwe: LweParams::current(),
        blobs,
    };

    let rank_hint = u32_bytes(&rank_matrix.hint(&manifest.rank_matrix()));
    let payload_hint = u32_bytes(&payload_matrix.hint(&manifest.payload_matrix()));
    manifest
        .blobs
        .insert("rank_hint".to_owned(), blob(&rank_hint));
    manifest
        .blobs
        .insert("payload_hint".to_owned(), blob(&payload_hint));
    manifest.validate().map_err(|e| e.to_string())?;

    fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let write = |name: &str, data: &[u8]| {
        let path = out.join(name);
        fs::write(&path, data).map_err(|e| format!("{}: {e}", path.display()))
    };
    write("routing.bin", &routing)?;
    write("layout.bin", &layout)?;
    write("page_digests.bin", &page_digests)?;
    write("rank.hint", &rank_hint)?;
    write("payload.hint", &payload_hint)?;
    write("model.usm", model_file)?;
    let to_bytes = |m: &Matrix| m.data().iter().map(|&v| v as u8).collect::<Vec<u8>>();
    write("rank.matrix", &to_bytes(&rank_matrix))?;
    write("payload.matrix", &to_bytes(&payload_matrix))?;
    let json = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    // Written last, so a partial build can never be mistaken for a finished one.
    write("manifest.json", &json)?;

    let public_bytes = (routing.len()
        + layout.len()
        + page_digests.len()
        + rank_hint.len()
        + payload_hint.len()
        + model_file.len()) as u64;
    let probes = manifest.probes as u64;
    let fetches = manifest.pages as u64;
    Ok(BuildReport {
        documents: n,
        clusters,
        rank_rows,
        rank_cols,
        pages: page_count,
        page_bytes,
        explained_variance: pca.explained,
        rank_log_p: log_p,
        scale,
        rank_failure_log2: rank_failure,
        payload_failure_log2: payload_failure,
        public_bytes,
        private_bytes: (rank_matrix.data().len() + payload_matrix.data().len()) as u64,
        upload_per_search: (probes * rank_cols as u64 + fetches * page_count as u64) * 4,
        download_per_search: (probes * rank_rows as u64 + fetches * page_bytes as u64) * 4,
        index_id: index_id(&json),
    })
}

/// Compressed size of a group of documents, used to estimate the size of the whole corpus.
fn miniz_estimate(docs: &[&Document]) -> usize {
    pages::encode(docs, 1 << 16).map_or_else(
        || pages::serialize(docs).len() / 2,
        |page| usize::from(u16::from_le_bytes([page[0], page[1]])) + 2,
    )
}

/// Cosine similarity, exposed so tools can compare against exact search.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(i: usize, len: usize) -> Document {
        Document {
            title: format!("Doc {i}"),
            text: format!("word{i} ").repeat(len),
            url: format!("u/{i}"),
        }
    }

    #[test]
    fn pagination_keeps_every_document_in_order() {
        let docs: Vec<Document> = (0..200).map(|i| doc(i, 20)).collect();
        let pages_out = paginate(&docs, 512);
        assert!(pages_out.len() > 1);
        let mut recovered = Vec::new();
        for (page, count) in &pages_out {
            assert_eq!(page.len(), 512);
            let d = pages::decode(page, usize::MAX).unwrap();
            assert_eq!(d.len(), *count);
            recovered.extend(d);
        }
        assert_eq!(recovered, docs);
    }

    #[test]
    fn a_document_too_big_for_a_page_is_shortened_not_lost() {
        // Long incompressible text, so that it cannot fit however well DEFLATE does.
        let mut state = 1u64;
        let noisy: String = (0..4000)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                char::from(b'a' + ((state >> 33) % 26) as u8)
            })
            .collect();
        let big = Document {
            title: "Huge".into(),
            text: noisy,
            url: "u".into(),
        };
        let pages_out = paginate(&[doc(1, 3), big, doc(2, 3)], 512);
        let all: Vec<Document> = pages_out
            .iter()
            .flat_map(|(p, _)| pages::decode(p, usize::MAX).unwrap())
            .collect();
        assert_eq!(all.len(), 3);
        assert_eq!(all[1].title, "Huge");
        assert!(all[1].text.len() < 4000 && !all[1].text.is_empty());
    }

    #[test]
    fn shortening_respects_character_boundaries() {
        let d = Document {
            title: "t".into(),
            text: "日本語🙂".repeat(400),
            url: String::new(),
        };
        let pages_out = paginate(&[d], 256);
        assert!(pages::decode(&pages_out[0].0, usize::MAX).is_some());
    }

    #[test]
    fn a_title_that_cannot_fit_is_shortened_too() {
        let mut state = 7u64;
        let title: String = (0..3000)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                char::from(b'a' + ((state >> 33) % 26) as u8)
            })
            .collect();
        let d = Document {
            title,
            text: String::new(),
            url: "x".repeat(50),
        };
        let pages_out = paginate(&[d], 128);
        assert_eq!(pages_out.len(), 1);
    }

    #[test]
    fn automatic_sizes_grow_with_the_corpus_and_stay_in_range() {
        assert!(automatic_rows(1_000_000, 256, 8) > automatic_rows(10_000, 256, 8));
        assert!(automatic_rows(10, 256, 8) >= 64);
        assert!(automatic_page_bytes(1e9, 4) > automatic_page_bytes(1e6, 4));
        assert_eq!(automatic_page_bytes(1.0, 4), 2048);
    }

    #[test]
    fn passages_are_shortened_on_a_word_boundary_within_the_limit() {
        assert_eq!(shorten_passage("short enough", 100), "short enough");
        let cut = shorten_passage("one two three four five six seven", 16);
        assert_eq!(cut, "one two three\u{2026}");
        assert!(cut.len() <= 16);
        // One enormous word is cut anyway rather than losing the whole passage.
        let long = "x".repeat(50);
        let cut = shorten_passage(&long, 20);
        assert!(cut.len() <= 20 && cut.ends_with('\u{2026}') && cut.starts_with("xxx"));
    }

    #[test]
    fn truncation_never_splits_a_character() {
        assert_eq!(truncate_utf8("héllo", 2), "h");
        assert_eq!(truncate_utf8("hi", 10), "hi");
    }
}
