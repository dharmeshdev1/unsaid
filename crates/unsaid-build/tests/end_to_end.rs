//! The whole protocol, with real cryptography, against a real built index. A plain function
//! stands in for the HTTP server: it does exactly what the server does.

use std::fs;
use std::path::{Path, PathBuf};

use unsaid_build::{BuildOptions, BuildReport, build};
use unsaid_core::client::IndexFiles;
use unsaid_core::{Document, Index, Manifest};
use unsaid_lwe::chacha::Drbg;
use unsaid_lwe::{LWE_DIM, Matrix};

const MODEL_DIM: usize = 48;

/// A corpus with real structure: documents gather around topics.
fn corpus(n: usize, topics: usize, seed: u8) -> (Vec<Document>, Vec<f32>) {
    let mut rng = Drbg::from_seed(&[seed; 32]);
    let centres: Vec<f32> = (0..topics * MODEL_DIM)
        .map(|_| rng.next_u32() as f32 / u32::MAX as f32 - 0.5)
        .collect();
    let mut docs = Vec::with_capacity(n);
    let mut embeddings = Vec::with_capacity(n * MODEL_DIM);
    for i in 0..n {
        let t = i % topics;
        let mut v: Vec<f32> = centres[t * MODEL_DIM..(t + 1) * MODEL_DIM]
            .iter()
            .map(|c| c + (rng.next_u32() as f32 / u32::MAX as f32 - 0.5) * 0.4)
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        embeddings.extend_from_slice(&v);
        docs.push(Document {
            title: format!("Document number {i} about topic {t}"),
            text: format!("This is the text of document {i}, which discusses topic {t} at length. Ünicode too: 日本語."),
            url: format!("https://example.org/doc/{i}"),
        });
    }
    (docs, embeddings)
}

struct Built {
    dir: PathBuf,
    report: BuildReport,
    docs: Vec<Document>,
    embeddings: Vec<f32>,
}

fn temp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("unsaid-e2e-{label}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    dir
}

fn build_corpus(label: &str, n: usize, options: BuildOptions) -> Built {
    let (docs, embeddings) = corpus(n, 12, 3);
    let dir = temp_dir(label);
    let report = build(
        &docs,
        &embeddings,
        "test-model",
        MODEL_DIM,
        b"model bytes",
        &options,
        &dir,
    )
    .unwrap();
    Built {
        dir,
        report,
        docs,
        embeddings,
    }
}

fn small_options() -> BuildOptions {
    let mut o = BuildOptions::new("test");
    o.dim = 32;
    o.rows = Some(64);
    o.page_bytes = Some(1024);
    o.probes = 3;
    o.pages = 2;
    o
}

fn read(dir: &Path, name: &str) -> Vec<u8> {
    fs::read(dir.join(name)).unwrap()
}

fn open(dir: &Path) -> Index {
    let (manifest, routing, layout, digests, rank_hint, payload_hint) = (
        read(dir, "manifest.json"),
        read(dir, "routing.bin"),
        read(dir, "layout.bin"),
        read(dir, "page_digests.bin"),
        read(dir, "rank.hint"),
        read(dir, "payload.hint"),
    );
    Index::open(
        &manifest,
        &IndexFiles {
            routing: &routing,
            layout: &layout,
            page_digests: &digests,
            rank_hint: &rank_hint,
            payload_hint: &payload_hint,
        },
    )
    .unwrap()
}

/// Exactly what the HTTP server does with a request body.
fn serve(matrix: &Matrix, request: &[u8]) -> Vec<u8> {
    assert_eq!(request.len() % (matrix.cols() * 4), 0);
    let mut out = Vec::new();
    for ct in request.chunks_exact(matrix.cols() * 4) {
        let words: Vec<u32> = ct
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for v in matrix.apply(&words) {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn server_matrices(index: &Index, dir: &Path) -> (Matrix, Matrix) {
    let m = index.manifest();
    (
        Matrix::from_bytes(m.rank.rows, m.rank.cols, &read(dir, "rank.matrix")),
        Matrix::from_bytes(m.payload.rows, m.payload.cols, &read(dir, "payload.matrix")),
    )
}

fn search(
    index: &Index,
    rank: &Matrix,
    payload: &Matrix,
    embedding: &[f32],
    k: usize,
) -> unsaid_core::Outcome {
    let plan = index.default_plan();
    let tokens = (0..plan.probes)
        .map(|_| index.prepare_rank().unwrap())
        .collect();
    let (stage, request) = index.begin(embedding, tokens).unwrap();
    let answer = serve(rank, &request);
    let tokens = (0..plan.pages)
        .map(|_| index.prepare_fetch().unwrap())
        .collect();
    let (stage, request) = stage.receive(index, &answer, tokens, k).unwrap();
    stage.receive(index, &serve(payload, &request)).unwrap()
}

#[test]
fn a_query_equal_to_a_document_finds_that_document() {
    let b = build_corpus("self", 1200, small_options());
    let index = open(&b.dir);
    let (rank, payload) = server_matrices(&index, &b.dir);
    let mut found = 0;
    for i in (0..1200).step_by(100) {
        let e = &b.embeddings[i * MODEL_DIM..(i + 1) * MODEL_DIM];
        let out = search(&index, &rank, &payload, e, 5);
        assert_eq!(out.corrupted_pages, 0);
        if out.hits.first().map(|h| h.document.title.as_str()) == Some(b.docs[i].title.as_str()) {
            found += 1;
            let h = &out.hits[0];
            assert_eq!(
                h.document, b.docs[i],
                "the document must arrive byte for byte"
            );
            assert!(h.score > 0.9, "score {}", h.score);
        }
    }
    assert!(found >= 11, "only {found} of 12 documents found themselves");
    fs::remove_dir_all(&b.dir).unwrap();
}

/// The cryptography must be invisible: decrypted scores equal plaintext scores exactly.
#[test]
fn decrypted_scores_are_exactly_the_plaintext_inner_products() {
    let b = build_corpus("exact", 800, small_options());
    let index = open(&b.dir);
    let (rank, _) = server_matrices(&index, &b.dir);
    let m = index.manifest().clone();
    let e = &b.embeddings[5 * MODEL_DIM..6 * MODEL_DIM];
    let projected = index.project(e);
    let plain: Vec<i32> = projected
        .iter()
        .map(|v| (v * m.query_scale).round() as i32)
        .collect();
    let token = index.prepare_rank().unwrap();
    let (stage, request) = index.begin(e, vec![token]).unwrap();
    let cluster = stage.clusters()[0];
    let answer = serve(&rank, &request);

    // Recompute in the clear what the row of each document should decrypt to.
    let mut expected = Vec::new();
    for row in 0..m.rank.rows {
        let block = &rank.row(row)[cluster * m.dim..(cluster + 1) * m.dim];
        expected.push(
            block
                .iter()
                .zip(&plain)
                .map(|(&a, &b)| i32::from(a) * b)
                .sum::<i32>(),
        );
    }
    let tokens = vec![index.prepare_fetch().unwrap()];
    let (fetch, _) = stage.receive(&index, &answer, tokens, 10_000).unwrap();
    let _ = fetch;
    // The decryption itself is checked against the expected row values in the next test
    // through the rank order, and directly here through a fresh decryption.
    let token = index.prepare_rank().unwrap();
    let entries: Vec<(usize, i32)> = plain
        .iter()
        .enumerate()
        .map(|(i, &v)| (cluster * m.dim + i, v))
        .collect();
    let (secret, ct) = token.seal(m.rank.log_p, &entries);
    let got = secret.open_signed(
        m.rank.log_p,
        &rank.apply(&ct),
        &read(&b.dir, "rank.hint")
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<u32>>(),
    );
    assert_eq!(got, expected);
    fs::remove_dir_all(&b.dir).unwrap();
}

/// Private results equal a plaintext scan of the same clusters.
#[test]
fn private_ranking_matches_a_plaintext_scan_of_the_routed_clusters() {
    let b = build_corpus("rank", 1000, small_options());
    let index = open(&b.dir);
    let (rank, payload) = server_matrices(&index, &b.dir);
    let m = index.manifest().clone();
    let layout: Vec<u32> = read(&b.dir, "layout.bin")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let sizes = &layout[..m.clusters];
    for q in [3usize, 250, 777] {
        let e = &b.embeddings[q * MODEL_DIM..(q + 1) * MODEL_DIM];
        let out = search(&index, &rank, &payload, e, 5);
        let projected = index.project(e);
        let plain: Vec<i32> = projected
            .iter()
            .map(|v| (v * m.query_scale).round() as i32)
            .collect();
        let mut scan: Vec<(i32, u32)> = Vec::new();
        for &c in &index.route(&projected, m.probes) {
            let start: u32 = sizes[..c].iter().sum();
            for row in 0..sizes[c] as usize {
                let block = &rank.row(row)[c * m.dim..(c + 1) * m.dim];
                let s: i32 = block
                    .iter()
                    .zip(&plain)
                    .map(|(&a, &b)| i32::from(a) * b)
                    .sum();
                scan.push((s, start + row as u32));
            }
        }
        scan.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        assert!(!out.hits.is_empty());
        // Every hit carries the exact plaintext score, and the best hit is the scan's best.
        let scale = m.doc_scale * m.query_scale;
        for hit in &out.hits {
            let want = scan.iter().find(|(_, d)| *d == hit.doc).unwrap().0;
            assert!((hit.score - want as f32 / scale).abs() < 1e-6);
        }
        assert_eq!(out.hits[0].doc, scan[0].1);
    }
    fs::remove_dir_all(&b.dir).unwrap();
}

#[test]
fn every_search_has_the_same_traffic_shape() {
    let b = build_corpus("shape", 600, small_options());
    let index = open(&b.dir);
    let (rank, _) = server_matrices(&index, &b.dir);
    let m = index.manifest().clone();
    let plan = index.default_plan();
    let mut sizes = std::collections::BTreeSet::new();
    for q in [0usize, 7, 300, 599] {
        let e = &b.embeddings[q * MODEL_DIM..(q + 1) * MODEL_DIM];
        let tokens = (0..plan.probes)
            .map(|_| index.prepare_rank().unwrap())
            .collect();
        let (stage, r1) = index.begin(e, tokens).unwrap();
        let answer = serve(&rank, &r1);
        let tokens = (0..plan.pages)
            .map(|_| index.prepare_fetch().unwrap())
            .collect();
        let (_, r2) = stage.receive(&index, &answer, tokens, 5).unwrap();
        sizes.insert((r1.len(), r2.len()));
    }
    assert_eq!(
        sizes.into_iter().collect::<Vec<_>>(),
        vec![(
            plan.probes * m.rank.cols * 4,
            plan.pages * m.payload.cols * 4
        )]
    );
    fs::remove_dir_all(&b.dir).unwrap();
}

/// A chi-square test on what the server receives: it must look uniformly random, for two very
/// different queries alike.
#[test]
fn ciphertext_bytes_are_uniform_whatever_the_query() {
    let b = build_corpus("uniform", 2000, small_options());
    let index = open(&b.dir);
    for q in [1usize, 1500] {
        let e = &b.embeddings[q * MODEL_DIM..(q + 1) * MODEL_DIM];
        let mut counts = [0u64; 256];
        let mut total = 0u64;
        for _ in 0..40 {
            let (_, request) = index.begin(e, vec![index.prepare_rank().unwrap()]).unwrap();
            for &byte in &request {
                counts[byte as usize] += 1;
            }
            total += request.len() as u64;
        }
        let expect = total as f64 / 256.0;
        let chi: f64 = counts
            .iter()
            .map(|&c| (c as f64 - expect).powi(2) / expect)
            .sum();
        // 255 degrees of freedom: the 1e-6 upper tail is about 361.
        assert!(
            chi < 361.0,
            "chi2 = {chi:.1} for query {q}, over {total} bytes"
        );
    }
    fs::remove_dir_all(&b.dir).unwrap();
}

#[test]
fn builds_are_deterministic_across_thread_counts() {
    let (docs, embeddings) = corpus(700, 8, 4);
    let options = small_options();
    let (a, b) = (temp_dir("det-a"), temp_dir("det-b"));
    let first = build(&docs, &embeddings, "m", MODEL_DIM, b"x", &options, &a).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let second = pool
        .install(|| build(&docs, &embeddings, "m", MODEL_DIM, b"x", &options, &b))
        .unwrap();
    assert_eq!(first.index_id, second.index_id);
    for name in [
        "manifest.json",
        "rank.matrix",
        "payload.matrix",
        "rank.hint",
        "routing.bin",
    ] {
        assert_eq!(read(&a, name), read(&b, name), "{name} differs");
    }
    fs::remove_dir_all(&a).unwrap();
    fs::remove_dir_all(&b).unwrap();
}

#[test]
fn tampering_with_any_public_file_is_detected() {
    let b = build_corpus("tamper", 300, small_options());
    let manifest = read(&b.dir, "manifest.json");
    let files = [
        read(&b.dir, "routing.bin"),
        read(&b.dir, "layout.bin"),
        read(&b.dir, "page_digests.bin"),
        read(&b.dir, "rank.hint"),
        read(&b.dir, "payload.hint"),
    ];
    for which in 0..files.len() {
        let mut changed = files.clone();
        changed[which][0] ^= 1;
        let result = Index::open(
            &manifest,
            &IndexFiles {
                routing: &changed[0],
                layout: &changed[1],
                page_digests: &changed[2],
                rank_hint: &changed[3],
                payload_hint: &changed[4],
            },
        );
        assert!(
            result.is_err(),
            "file {which} was accepted after being modified"
        );
    }
    fs::remove_dir_all(&b.dir).unwrap();
}

/// A malicious server can supply a consistent manifest with hostile contents. The client must
/// refuse these even though every hash matches.
#[test]
fn hostile_but_consistent_manifests_are_refused() {
    let b = build_corpus("hostile", 300, small_options());
    let base: Manifest = serde_json::from_slice(&read(&b.dir, "manifest.json")).unwrap();
    let files = [
        read(&b.dir, "routing.bin"),
        read(&b.dir, "layout.bin"),
        read(&b.dir, "page_digests.bin"),
        read(&b.dir, "rank.hint"),
        read(&b.dir, "payload.hint"),
    ];
    let try_open = |m: &Manifest| {
        Index::open(
            &serde_json::to_vec(m).unwrap(),
            &IndexFiles {
                routing: &files[0],
                layout: &files[1],
                page_digests: &files[2],
                rank_hint: &files[3],
                payload_hint: &files[4],
            },
        )
    };
    assert!(try_open(&base).is_ok());
    let mut weak = base.clone();
    weak.lwe.n = 64;
    assert!(try_open(&weak).is_err(), "a weakened lattice was accepted");
    let mut count = base.clone();
    count.documents += 1;
    assert!(
        try_open(&count).is_err(),
        "a wrong document count was accepted"
    );
    let mut rows = base.clone();
    rows.rank.rows = 1;
    assert!(
        try_open(&rows).is_err(),
        "a shape that contradicts the files was accepted"
    );
    fs::remove_dir_all(&b.dir).unwrap();
}

/// A server returning garbage can only damage results. It cannot crash the client, and the
/// client's traffic does not change in response.
#[test]
fn garbage_answers_are_survived_with_the_same_traffic() {
    let b = build_corpus("garbage", 500, small_options());
    let index = open(&b.dir);
    let plan = index.default_plan();
    let m = index.manifest().clone();
    let e = &b.embeddings[..MODEL_DIM];
    let mut rng = Drbg::from_seed(&[8u8; 32]);
    let mut noise = |len: usize| -> Vec<u8> {
        (0..len / 4)
            .flat_map(|_| rng.next_u32().to_le_bytes())
            .collect()
    };

    let tokens = (0..plan.probes)
        .map(|_| index.prepare_rank().unwrap())
        .collect();
    let (stage, _) = index.begin(e, tokens).unwrap();
    let answer = noise(plan.probes * m.rank.rows * 4);
    let tokens = (0..plan.pages)
        .map(|_| index.prepare_fetch().unwrap())
        .collect();
    let (stage, request) = stage.receive(&index, &answer, tokens, 5).unwrap();
    assert_eq!(request.len(), plan.pages * m.payload.cols * 4);
    let answer = noise(plan.pages * m.payload.rows * 4);
    let out = stage.receive(&index, &answer).unwrap();
    assert_eq!(out.corrupted_pages, plan.pages);
    assert!(out.hits.is_empty());
    fs::remove_dir_all(&b.dir).unwrap();
}

#[test]
fn answers_of_the_wrong_length_are_errors() {
    let b = build_corpus("length", 300, small_options());
    let index = open(&b.dir);
    let e = &b.embeddings[..MODEL_DIM];
    let (stage, _) = index.begin(e, vec![index.prepare_rank().unwrap()]).unwrap();
    let tokens = vec![index.prepare_fetch().unwrap()];
    assert!(stage.receive(&index, &[0u8; 7], tokens, 5).is_err());
    fs::remove_dir_all(&b.dir).unwrap();
}

#[test]
fn tokens_for_the_wrong_matrix_are_refused() {
    let b = build_corpus("tokens", 300, small_options());
    let index = open(&b.dir);
    let e = &b.embeddings[..MODEL_DIM];
    assert!(
        index
            .begin(e, vec![index.prepare_fetch().unwrap()])
            .is_err()
    );
    assert!(index.begin(e, vec![]).is_err());
    fs::remove_dir_all(&b.dir).unwrap();
}

/// Corpora much smaller than the budgets must still work and keep their traffic shape.
#[test]
fn tiny_corpora_work_and_filler_fetches_keep_the_shape() {
    for n in [1usize, 2, 5, 17] {
        let mut o = small_options();
        o.pages = 6;
        o.probes = 4;
        o.page_bytes = Some(2048);
        let b = build_corpus(&format!("tiny{n}"), n, o);
        let index = open(&b.dir);
        let (rank, payload) = server_matrices(&index, &b.dir);
        let e = &b.embeddings[..MODEL_DIM];
        let out = search(&index, &rank, &payload, e, 3);
        assert_eq!(out.corrupted_pages, 0, "n = {n}");
        assert_eq!(out.hits[0].document, b.docs[0], "n = {n}");
        assert!(out.hits.len() <= 3);
        let mut seen = std::collections::HashSet::new();
        assert!(
            out.hits.iter().all(|h| seen.insert(h.doc)),
            "duplicate hits for n = {n}"
        );
        fs::remove_dir_all(&b.dir).unwrap();
    }
}

#[test]
fn reports_describe_the_files_on_disk() {
    let b = build_corpus("report", 400, small_options());
    let r = &b.report;
    let size = |name: &str| fs::metadata(b.dir.join(name)).unwrap().len();
    assert_eq!(
        r.private_bytes,
        size("rank.matrix") + size("payload.matrix")
    );
    assert_eq!(size("rank.hint"), (r.rank_rows * LWE_DIM * 4) as u64);
    assert_eq!(size("payload.hint"), (r.page_bytes * LWE_DIM * 4) as u64);
    assert!(r.rank_failure_log2 <= -40.0 && r.payload_failure_log2 <= -40.0);
    assert_eq!(r.documents, 400);
    assert!(
        r.rank_rows <= 64,
        "the cap was not respected: {}",
        r.rank_rows
    );
    fs::remove_dir_all(&b.dir).unwrap();
}

#[test]
fn bad_inputs_are_errors() {
    let dir = temp_dir("bad");
    let o = small_options();
    let (docs, e) = corpus(10, 2, 5);
    assert!(build(&[], &[], "m", MODEL_DIM, b"x", &o, &dir).is_err());
    assert!(build(&docs, &e[1..], "m", MODEL_DIM, b"x", &o, &dir).is_err());
    let mut with_nan = e.clone();
    with_nan[3] = f32::NAN;
    assert!(build(&docs, &with_nan, "m", MODEL_DIM, b"x", &o, &dir).is_err());
    assert!(!dir.join("manifest.json").exists());
}
