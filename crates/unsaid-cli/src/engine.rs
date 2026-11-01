//! A complete private search client: embed locally, encrypt, talk to a server, decrypt.

use std::path::Path;
use std::time::Instant;

use sha2::{Digest, Sha256};
use unsaid_core::client::{IndexFiles, SearchPlan};
use unsaid_core::{Index, Manifest, Outcome};
use unsaid_embed::Embedder;
use unsaid_lwe::Matrix;

use crate::models::cache_dir;

/// Something that can answer encrypted queries.
pub trait Transport: Send + Sync {
    /// Send ranking queries, get the encrypted scores.
    fn rank(&self, request: &[u8]) -> Result<Vec<u8>, String>;
    /// Send page queries, get the encrypted pages.
    fn fetch(&self, request: &[u8]) -> Result<Vec<u8>, String>;
}

/// A server reached over HTTP.
pub struct Http {
    base: String,
}

fn post(url: &str, body: &[u8]) -> Result<Vec<u8>, String> {
    let response = ureq::post(url)
        .set("Content-Type", "application/octet-stream")
        .send_bytes(body)
        .map_err(|e| format!("{url}: {e}"))?;
    let mut out = Vec::new();
    std::io::Read::read_to_end(&mut response.into_reader(), &mut out)
        .map_err(|e| format!("{url}: {e}"))?;
    Ok(out)
}

fn get(url: &str) -> Result<Vec<u8>, String> {
    let response = ureq::get(url).call().map_err(|e| format!("{url}: {e}"))?;
    let mut out = Vec::new();
    std::io::Read::read_to_end(&mut response.into_reader(), &mut out)
        .map_err(|e| format!("{url}: {e}"))?;
    Ok(out)
}

impl Transport for Http {
    fn rank(&self, request: &[u8]) -> Result<Vec<u8>, String> {
        post(&format!("{}/v1/rank", self.base), request)
    }

    fn fetch(&self, request: &[u8]) -> Result<Vec<u8>, String> {
        post(&format!("{}/v1/fetch", self.base), request)
    }
}

/// The server's arithmetic, run in this process. Used for benchmarks and evaluation.
pub struct InProcess {
    rank: Matrix,
    payload: Matrix,
}

impl Transport for InProcess {
    fn rank(&self, request: &[u8]) -> Result<Vec<u8>, String> {
        self.rank
            .answer(request)
            .ok_or_else(|| "bad request size".to_owned())
    }

    fn fetch(&self, request: &[u8]) -> Result<Vec<u8>, String> {
        self.payload
            .answer(request)
            .ok_or_else(|| "bad request size".to_owned())
    }
}

/// Time and traffic for one search.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Trace {
    /// Embedding the query, in milliseconds.
    pub embed_ms: f64,
    /// Creating the encrypted queries, in milliseconds.
    pub encrypt_ms: f64,
    /// Waiting for the ranking answer, in milliseconds.
    pub rank_ms: f64,
    /// Waiting for the page answer, in milliseconds.
    pub fetch_ms: f64,
    /// Decrypting and assembling results, in milliseconds.
    pub decrypt_ms: f64,
    /// Bytes sent to the server.
    pub uploaded: u64,
    /// Bytes received from the server.
    pub downloaded: u64,
}

impl Trace {
    /// The whole search, in milliseconds.
    pub fn total_ms(&self) -> f64 {
        self.embed_ms + self.encrypt_ms + self.rank_ms + self.fetch_ms + self.decrypt_ms
    }
}

/// A connected client.
pub struct Engine {
    index: Index,
    embedder: Embedder,
    transport: Box<dyn Transport>,
}

fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Fetch a public file, using a local copy when one with the right hash exists.
fn cached_blob(base: &str, sha256: &str) -> Result<Vec<u8>, String> {
    let path = cache_dir().join("blobs").join(sha256);
    if let Ok(data) = std::fs::read(&path) {
        if sha256_hex(&data) == sha256 {
            return Ok(data);
        }
    }
    let data = get(&format!("{base}/v1/blob/{sha256}"))?;
    if sha256_hex(&data) != sha256 {
        return Err(format!(
            "the server sent a file that does not match hash {sha256}"
        ));
    }
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // A failed write only costs a download next time.
    let _ = std::fs::write(&path, &data);
    Ok(data)
}

impl Engine {
    /// Connect to a server, downloading and verifying whatever is not cached yet. If `pin` is
    /// given, the connection is refused unless the index has that identifier.
    pub fn connect(base: &str, pin: Option<&str>) -> Result<Self, String> {
        let base = base.trim_end_matches('/').to_owned();
        let manifest_json = get(&format!("{base}/v1/manifest"))?;
        let manifest = Manifest::parse(&manifest_json).map_err(|e| e.to_string())?;
        let id = manifest.index_id();
        if let Some(pin) = pin {
            if pin != id {
                return Err(format!("this server has index {id}, not the pinned {pin}"));
            }
        }
        let blob = |name: &str| -> Result<Vec<u8>, String> {
            let info = manifest
                .blobs
                .get(name)
                .ok_or_else(|| format!("no blob {name}"))?;
            cached_blob(&base, &info.sha256)
        };
        let (routing, layout, digests, rank_hint, payload_hint, model) = (
            blob("routing")?,
            blob("layout")?,
            blob("page_digests")?,
            blob("rank_hint")?,
            blob("payload_hint")?,
            blob("model")?,
        );
        let index = Index::open(
            &manifest_json,
            &IndexFiles {
                routing: &routing,
                layout: &layout,
                page_digests: &digests,
                rank_hint: &rank_hint,
                payload_hint: &payload_hint,
            },
        )
        .map_err(|e| e.to_string())?;
        let embedder = Embedder::load(&model).map_err(|e| e.to_string())?;
        Ok(Self {
            index,
            embedder,
            transport: Box::new(Http { base }),
        })
    }

    /// Open an index directory and answer queries in this process.
    pub fn local(dir: &Path) -> Result<Self, String> {
        let read = |name: &str| {
            let path = dir.join(name);
            std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))
        };
        let manifest_json = read("manifest.json")?;
        let index = Index::open(
            &manifest_json,
            &IndexFiles {
                routing: &read("routing.bin")?,
                layout: &read("layout.bin")?,
                page_digests: &read("page_digests.bin")?,
                rank_hint: &read("rank.hint")?,
                payload_hint: &read("payload.hint")?,
            },
        )
        .map_err(|e| e.to_string())?;
        let embedder = Embedder::load(&read("model.usm")?).map_err(|e| e.to_string())?;
        let m = index.manifest();
        let transport = InProcess {
            rank: Matrix::from_bytes(m.rank.rows, m.rank.cols, &read("rank.matrix")?),
            payload: Matrix::from_bytes(m.payload.rows, m.payload.cols, &read("payload.matrix")?),
        };
        Ok(Self {
            index,
            embedder,
            transport: Box::new(transport),
        })
    }

    /// The verified index.
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Embed a query on this machine.
    pub fn embed(&self, query: &str) -> Vec<f32> {
        self.embedder.embed_query(query)
    }

    /// Search with an embedding computed elsewhere.
    pub fn search_embedding(
        &self,
        embedding: &[f32],
        limit: usize,
        plan: SearchPlan,
    ) -> Result<(Outcome, Trace), String> {
        let mut trace = Trace::default();
        let t = Instant::now();
        let tokens = (0..plan.probes)
            .map(|_| self.index.prepare_rank())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        let fetch_tokens = (0..plan.pages)
            .map(|_| self.index.prepare_fetch())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        let (stage, request) = self
            .index
            .begin(embedding, tokens)
            .map_err(|e| e.to_string())?;
        trace.encrypt_ms = t.elapsed().as_secs_f64() * 1e3;
        trace.uploaded += request.len() as u64;

        let t = Instant::now();
        let answer = self.transport.rank(&request)?;
        trace.rank_ms = t.elapsed().as_secs_f64() * 1e3;
        trace.downloaded += answer.len() as u64;

        let t = Instant::now();
        let (stage, request) = stage
            .receive(&self.index, &answer, fetch_tokens, limit)
            .map_err(|e| e.to_string())?;
        trace.decrypt_ms = t.elapsed().as_secs_f64() * 1e3;
        trace.uploaded += request.len() as u64;

        let t = Instant::now();
        let answer = self.transport.fetch(&request)?;
        trace.fetch_ms = t.elapsed().as_secs_f64() * 1e3;
        trace.downloaded += answer.len() as u64;

        let t = Instant::now();
        let outcome = stage
            .receive(&self.index, &answer)
            .map_err(|e| e.to_string())?;
        trace.decrypt_ms += t.elapsed().as_secs_f64() * 1e3;
        Ok((outcome, trace))
    }

    /// Search for a query.
    pub fn search(
        &self,
        query: &str,
        limit: usize,
        plan: SearchPlan,
    ) -> Result<(Outcome, Trace), String> {
        let t = Instant::now();
        let embedding = self.embed(query);
        let embed_ms = t.elapsed().as_secs_f64() * 1e3;
        let (outcome, mut trace) = self.search_embedding(&embedding, limit, plan)?;
        trace.embed_ms = embed_ms;
        Ok((outcome, trace))
    }
}
