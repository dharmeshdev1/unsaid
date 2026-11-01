//! A complete private search client: embed locally, encrypt, talk to a server, decrypt.

use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use unsaid_core::client::{IndexFiles, SearchPlan};
use unsaid_core::manifest::{BlobInfo, MAX_MANIFEST_BYTES, MatrixInfo, index_id, sha256_hex};
use unsaid_core::{Index, Manifest, Outcome};
use unsaid_embed::{Embedder, is_known_model};
use unsaid_lwe::{Matrix, Prepared};

use crate::models::cache_dir;

/// A longer question takes longer to embed. If a server has any reference point, such as having
/// just answered this process, it can time that, so a query only leaves on a multiple of this
/// after the question arrived. That covers all the work that depends on the question.
const QUERY_WINDOW: Duration = Duration::from_millis(250);

/// Longest a query and its answer may take.
const POST_TIMEOUT: Duration = Duration::from_secs(300);
/// Longest a public file may take to download.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(3600);

/// Something that can answer encrypted queries.
pub trait Transport: Send + Sync {
    /// Send ranking queries, get the encrypted scores.
    fn rank(&self, request: &[u8]) -> Result<Vec<u8>, String>;
    /// Send page queries, get the encrypted pages.
    fn fetch(&self, request: &[u8]) -> Result<Vec<u8>, String>;
    /// Whether something outside this process can time the requests.
    fn observable(&self) -> bool;
}

/// A server reached over HTTP. It is treated as hostile: it cannot redirect a request, stall one
/// forever, or send more than the exact amount the protocol calls for.
pub struct Http {
    base: String,
    agent: ureq::Agent,
    rank: MatrixInfo,
    payload: MatrixInfo,
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .redirects(0)
        // Close each connection as soon as its answer is read. A connection left open would close
        // later, after the results were processed, and a server could time that.
        .max_idle_connections(0)
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(120))
        .timeout_write(Duration::from_secs(120))
        .build()
}

/// Read a response of at most `limit` bytes.
fn read_limited(response: ureq::Response, limit: u64, url: &str) -> Result<Vec<u8>, String> {
    if response.status() != 200 {
        return Err(format!("{url} answered {}", response.status()));
    }
    let mut out = Vec::new();
    response
        .into_reader()
        .take(limit + 1)
        .read_to_end(&mut out)
        .map_err(|e| format!("{url}: {e}"))?;
    if out.len() as u64 > limit {
        return Err(format!("{url} sent more than the {limit} bytes expected"));
    }
    Ok(out)
}

fn get(agent: &ureq::Agent, url: &str, limit: u64) -> Result<Vec<u8>, String> {
    let response = agent
        .get(url)
        .timeout(DOWNLOAD_TIMEOUT)
        .call()
        .map_err(|e| format!("{url}: {e}"))?;
    read_limited(response, limit, url)
}

impl Http {
    fn post(&self, path: &str, matrix: &MatrixInfo, request: &[u8]) -> Result<Vec<u8>, String> {
        let url = format!("{}{path}", self.base);
        let queries = (request.len() / (matrix.cols * 4)) as u64;
        let response = self
            .agent
            .post(&url)
            .timeout(POST_TIMEOUT)
            .set("Content-Type", "application/octet-stream")
            .send_bytes(request)
            .map_err(|e| format!("{url}: {e}"))?;
        read_limited(response, queries * matrix.rows as u64 * 4, &url)
    }
}

impl Transport for Http {
    fn rank(&self, request: &[u8]) -> Result<Vec<u8>, String> {
        self.post("/v1/rank", &self.rank, request)
    }

    fn fetch(&self, request: &[u8]) -> Result<Vec<u8>, String> {
        self.post("/v1/fetch", &self.payload, request)
    }

    fn observable(&self) -> bool {
        true
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

    fn observable(&self) -> bool {
        false
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
    /// Waiting on purpose so that timing reveals nothing, in milliseconds.
    pub held_ms: f64,
    /// Bytes sent to the server.
    pub uploaded: u64,
    /// Bytes received from the server.
    pub downloaded: u64,
}

impl Trace {
    /// The whole search, in milliseconds.
    pub fn total_ms(&self) -> f64 {
        self.embed_ms
            + self.encrypt_ms
            + self.rank_ms
            + self.fetch_ms
            + self.decrypt_ms
            + self.held_ms
    }
}

/// Encryption work done ahead of time. Each token is handed out once and then gone.
#[derive(Default)]
struct Pool {
    rank: Vec<Prepared>,
    fetch: Vec<Prepared>,
}

/// A connected client.
pub struct Engine {
    index: Index,
    embedder: Embedder,
    transport: Box<dyn Transport>,
    pool: Mutex<Pool>,
    taken: Condvar,
}

/// Fetch a public file, using a local copy when one with the right hash exists.
fn cached_blob(agent: &ureq::Agent, base: &str, info: &BlobInfo) -> Result<Vec<u8>, String> {
    // The manifest has already been checked, so this hash really is a hash and not a path.
    let path = cache_dir().join("blobs").join(&info.sha256);
    let right_size = std::fs::metadata(&path).is_ok_and(|m| m.is_file() && m.len() == info.bytes);
    if right_size {
        if let Ok(data) = std::fs::read(&path) {
            if sha256_hex(&data) == info.sha256 {
                return Ok(data);
            }
        }
    }
    let url = format!("{base}/v1/blob/{}", info.sha256);
    let data = get(agent, &url, info.bytes)?;
    if sha256_hex(&data) != info.sha256 {
        return Err(format!(
            "the server sent a file that does not match hash {}",
            info.sha256
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
    /// given, the connection is refused unless the index has that identifier. The index's model
    /// must be on the trusted list or named in `allowed_models`.
    pub fn connect(
        base: &str,
        pin: Option<&str>,
        allowed_models: &[String],
    ) -> Result<Self, String> {
        let base = base.trim_end_matches('/').to_owned();
        let agent = agent();
        let manifest_json = get(&agent, &format!("{base}/v1/manifest"), MAX_MANIFEST_BYTES)?;
        let manifest = Manifest::parse(&manifest_json).map_err(|e| e.to_string())?;
        let id = index_id(&manifest_json);
        if let Some(pin) = pin {
            if pin != id {
                return Err(format!("this server has index {id}, not the pinned {pin}"));
            }
        }
        let model_hash = &manifest.blobs["model"].sha256;
        if !is_known_model(model_hash) && !allowed_models.contains(model_hash) {
            return Err(format!(
                "this index uses a model this client does not know, with hash {model_hash}. A \
                 model shapes how your query is handled on this machine, so a hostile one could \
                 reveal words in it. To use it anyway, pass --allow-model {model_hash}"
            ));
        }
        let blob = |name: &str| cached_blob(&agent, &base, &manifest.blobs[name]);
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
        if embedder.dim() != manifest.model_dim {
            return Err("the model's vectors do not fit this index".to_owned());
        }
        let transport = Http {
            base,
            agent,
            rank: manifest.rank.clone(),
            payload: manifest.payload.clone(),
        };
        Ok(Self {
            index,
            embedder,
            transport: Box::new(transport),
            pool: Mutex::default(),
            taken: Condvar::new(),
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
        if embedder.dim() != m.model_dim {
            return Err("the model's vectors do not fit this index".to_owned());
        }
        let matrix = |name: &str, shape: &MatrixInfo| {
            let data = read(name)?;
            if data.len() != shape.rows * shape.cols {
                return Err(format!("{name} has the wrong size"));
            }
            Ok::<_, String>(Matrix::from_bytes(shape.rows, shape.cols, &data))
        };
        let transport = InProcess {
            rank: matrix("rank.matrix", &m.rank)?,
            payload: matrix("payload.matrix", &m.payload)?,
        };
        Ok(Self {
            index,
            embedder,
            transport: Box::new(transport),
            pool: Mutex::default(),
            taken: Condvar::new(),
        })
    }

    /// Keep the encryption work for `searches` searches ready in the background, so that a
    /// long-running process such as the gateway can start each search at once.
    pub fn prepare_in_background(self: &Arc<Self>, plan: SearchPlan, searches: usize) {
        let engine = Arc::clone(self);
        std::thread::spawn(move || {
            loop {
                {
                    let lock = engine.pool.lock().unwrap_or_else(PoisonError::into_inner);
                    let _idle = engine
                        .taken
                        .wait_while(lock, |pool| {
                            pool.rank.len() >= plan.probes * searches
                                && pool.fetch.len() >= plan.pages * searches
                        })
                        .unwrap_or_else(PoisonError::into_inner);
                }
                let Ok((rank, fetch)) = engine.prepare(plan) else {
                    // Searches prepare for themselves and will report the error.
                    return;
                };
                let mut pool = engine.pool.lock().unwrap_or_else(PoisonError::into_inner);
                pool.rank.extend(rank);
                pool.fetch.extend(fetch);
            }
        });
    }

    fn prepare(&self, plan: SearchPlan) -> Result<(Vec<Prepared>, Vec<Prepared>), String> {
        let rank = (0..plan.probes)
            .map(|_| self.index.prepare_rank())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        let fetch = (0..plan.pages)
            .map(|_| self.index.prepare_fetch())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        Ok((rank, fetch))
    }

    /// Tokens for one search, from the pool if it has enough.
    fn tokens(&self, plan: SearchPlan) -> Result<(Vec<Prepared>, Vec<Prepared>), String> {
        {
            let mut pool = self.pool.lock().unwrap_or_else(PoisonError::into_inner);
            if pool.rank.len() >= plan.probes && pool.fetch.len() >= plan.pages {
                let rank_from = pool.rank.len() - plan.probes;
                let fetch_from = pool.fetch.len() - plan.pages;
                let tokens = (
                    pool.rank.split_off(rank_from),
                    pool.fetch.split_off(fetch_from),
                );
                self.taken.notify_all();
                return Ok(tokens);
            }
        }
        self.prepare(plan)
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
        let tokens = self.tokens(plan)?;
        trace.encrypt_ms = t.elapsed().as_secs_f64() * 1e3;
        self.run(embedding, tokens, limit, None, trace)
    }

    /// Search for a query.
    pub fn search(
        &self,
        query: &str,
        limit: usize,
        plan: SearchPlan,
    ) -> Result<(Outcome, Trace), String> {
        let mut trace = Trace::default();
        let t = Instant::now();
        // This work is the same for any question, so it comes before the clock starts.
        let tokens = self.tokens(plan)?;
        trace.encrypt_ms = t.elapsed().as_secs_f64() * 1e3;
        let asked = Instant::now();
        let embedding = self.embed(query);
        trace.embed_ms = asked.elapsed().as_secs_f64() * 1e3;
        self.run(&embedding, tokens, limit, Some(asked), trace)
    }

    fn hold(&self, wait: Duration, trace: &mut Trace) {
        if self.transport.observable() {
            std::thread::sleep(wait);
            trace.held_ms += wait.as_secs_f64() * 1e3;
        }
    }

    fn run(
        &self,
        embedding: &[f32],
        (tokens, fetch_tokens): (Vec<Prepared>, Vec<Prepared>),
        limit: usize,
        asked: Option<Instant>,
        mut trace: Trace,
    ) -> Result<(Outcome, Trace), String> {
        let t = Instant::now();
        let (stage, request) = self
            .index
            .begin(embedding, tokens)
            .map_err(|e| e.to_string())?;
        trace.encrypt_ms += t.elapsed().as_secs_f64() * 1e3;
        if let Some(asked) = asked {
            let spent = asked.elapsed();
            let windows = spent.as_nanos().div_ceil(QUERY_WINDOW.as_nanos()).max(1) as u32;
            self.hold((QUERY_WINDOW * windows).saturating_sub(spent), &mut trace);
        }
        trace.uploaded += request.len() as u64;

        let t = Instant::now();
        let answer = self.transport.rank(&request)?;
        trace.rank_ms = t.elapsed().as_secs_f64() * 1e3;
        trace.downloaded += answer.len() as u64;

        let t = Instant::now();
        let scored = stage
            .decrypt(&self.index, &answer)
            .map_err(|e| e.to_string())?;
        let decrypting = t.elapsed();
        let picked = scored.pick(&self.index, fetch_tokens.len(), limit);
        trace.decrypt_ms = t.elapsed().as_secs_f64() * 1e3;
        // Picking pages takes a time that depends on the results. It is tiny next to the decryption
        // before it, which takes a fixed time, so waiting until twice that hides it.
        self.hold((decrypting * 2).saturating_sub(t.elapsed()), &mut trace);

        let t = Instant::now();
        let (stage, request) = picked
            .seal(&self.index, fetch_tokens)
            .map_err(|e| e.to_string())?;
        trace.encrypt_ms += t.elapsed().as_secs_f64() * 1e3;
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
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::net::TcpListener;

    use super::*;

    /// Answer one request with these exact bytes, and return the address to ask.
    fn serve_once(response: Vec<u8>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buffer = [0u8; 4096];
            let _ = stream.read(&mut buffer);
            let _ = stream.write_all(&response);
        });
        format!("http://{address}/")
    }

    fn http_response(status: &str, headers: &str, body: &[u8]) -> Vec<u8> {
        let mut out = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{headers}\r\n",
            body.len()
        )
        .into_bytes();
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn a_response_of_the_right_size_is_read() {
        let url = serve_once(http_response("200 OK", "", b"12345678"));
        assert_eq!(get(&agent(), &url, 8).unwrap(), b"12345678");
    }

    #[test]
    fn more_data_than_expected_is_refused() {
        let url = serve_once(http_response("200 OK", "", &[7u8; 1000]));
        let error = get(&agent(), &url, 16).unwrap_err();
        assert!(error.contains("more than"), "{error}");
    }

    /// A connection left open would close later, after the results were processed, and the server
    /// could time that. So the close must come when the answer is read, not when the agent goes.
    #[test]
    fn a_connection_closes_as_soon_as_its_answer_is_read() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let (closed, seen) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buffer = [0u8; 4096];
            let _ = stream.read(&mut buffer);
            stream
                .write_all(&http_response("200 OK", "", b"12345678"))
                .unwrap();
            // A read of nothing means that the client has closed its end.
            let _ = stream.read(&mut buffer);
            closed.send(()).unwrap();
        });
        let client = agent();
        assert_eq!(get(&client, &url, 8).unwrap(), b"12345678");
        // The agent is still alive here, and would be keeping an idle connection open.
        seen.recv_timeout(Duration::from_secs(5))
            .expect("the connection stayed open after its answer was read");
        drop(client);
    }

    /// A redirect could send a query anywhere, and turn a POST into a GET.
    #[test]
    fn a_redirect_is_not_followed() {
        let target = TcpListener::bind("127.0.0.1:0").unwrap();
        target.set_nonblocking(true).unwrap();
        let location = format!("Location: http://{}/\r\n", target.local_addr().unwrap());
        let url = serve_once(http_response("302 Found", &location, b""));
        let error = get(&agent(), &url, 16).unwrap_err();
        assert!(error.contains("302"), "{error}");
        assert!(target.accept().is_err(), "the client followed the redirect");
    }
}
