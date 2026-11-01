//! The Unsaid private search server.
//!
//! The server holds two matrices and multiplies them by whatever it receives. Everything it
//! gets from a client is ciphertext, every query reads the whole matrix, and the handlers keep
//! no record of requests. So there is no search history here to log, leak or hand over.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path as UrlPath, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use tokio::sync::Semaphore;
use tower_http::cors::{Any, CorsLayer};
use unsaid_core::Manifest;
use unsaid_core::manifest::sha256_hex;
use unsaid_lwe::Matrix;

/// Most encrypted queries accepted in one request.
pub const MAX_QUERIES_PER_REQUEST: usize = 64;

/// File name of each public blob inside an index directory.
pub const BLOB_FILES: [(&str, &str); 6] = [
    ("model", "model.usm"),
    ("routing", "routing.bin"),
    ("layout", "layout.bin"),
    ("page_digests", "page_digests.bin"),
    ("rank_hint", "rank.hint"),
    ("payload_hint", "payload.hint"),
];

/// A static file served with the web interface.
#[derive(Clone, Copy)]
pub struct Asset {
    /// Path under the site root, such as `/app.js`.
    pub path: &'static str,
    /// Media type.
    pub content_type: &'static str,
    /// The file.
    pub bytes: &'static [u8],
}

/// A loaded index, ready to answer queries.
pub struct Loaded {
    manifest_json: Vec<u8>,
    manifest: Manifest,
    rank: Matrix,
    payload: Matrix,
    blobs: HashMap<String, Bytes>,
    permits: Arc<Semaphore>,
    rank_requests: AtomicU64,
    fetch_requests: AtomicU64,
}

impl Loaded {
    /// Load an index directory, checking every file against the manifest.
    pub fn open(dir: &Path) -> Result<Self, String> {
        let read = |name: &str| {
            let path = dir.join(name);
            std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))
        };
        let manifest_json = read("manifest.json")?;
        let manifest = Manifest::parse(&manifest_json).map_err(|e| e.to_string())?;
        let mut blobs = HashMap::new();
        for (name, file) in BLOB_FILES {
            let data = read(file)?;
            let info = manifest
                .blobs
                .get(name)
                .ok_or_else(|| format!("no blob {name}"))?;
            if info.sha256 != sha256_hex(&data) {
                return Err(format!("{file} does not match the manifest"));
            }
            blobs.insert(info.sha256.clone(), Bytes::from(data));
        }
        let matrix = |name: &str, rows: usize, cols: usize| -> Result<Matrix, String> {
            let data = read(name)?;
            if data.len() != rows * cols {
                return Err(format!("{name} has the wrong size"));
            }
            Ok(Matrix::from_bytes(rows, cols, &data))
        };
        let rank = matrix("rank.matrix", manifest.rank.rows, manifest.rank.cols)?;
        let payload = matrix(
            "payload.matrix",
            manifest.payload.rows,
            manifest.payload.cols,
        )?;
        let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
        Ok(Self {
            manifest_json,
            manifest,
            rank,
            payload,
            blobs,
            permits: Arc::new(Semaphore::new(workers)),
            rank_requests: AtomicU64::new(0),
            fetch_requests: AtomicU64::new(0),
        })
    }

    /// The manifest this server publishes.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
}

#[derive(Clone)]
struct AppState {
    index: Arc<Loaded>,
    assets: &'static [Asset],
}

fn octets(body: Vec<u8>) -> Response {
    (
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/octet-stream"),
            ),
            (header::CACHE_CONTROL, HeaderValue::from_static("no-store")),
        ],
        body,
    )
        .into_response()
}

/// Multiply a request by one of the matrices.
async fn answer(state: AppState, payload: bool, body: Bytes) -> Response {
    let cols = if payload {
        state.index.payload.cols()
    } else {
        state.index.rank.cols()
    };
    let width = cols * 4;
    // Refuse a bad request before queueing for the arithmetic.
    if body.is_empty() || body.len() % width != 0 {
        return (
            StatusCode::BAD_REQUEST,
            "request size is not a whole number of queries",
        )
            .into_response();
    }
    if body.len() / width > MAX_QUERIES_PER_REQUEST {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            "too many queries in one request",
        )
            .into_response();
    }
    let Ok(permit) = state.index.permits.clone().acquire_owned().await else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    let index = state.index.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let matrix = if payload { &index.payload } else { &index.rank };
        matrix.answer(&body)
    })
    .await;
    match result {
        Ok(Some(out)) => {
            let counter = if payload {
                &state.index.fetch_requests
            } else {
                &state.index.rank_requests
            };
            counter.fetch_add(1, Ordering::Relaxed);
            octets(out)
        }
        Ok(None) | Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

async fn rank(State(state): State<AppState>, body: Bytes) -> Response {
    answer(state, false, body).await
}

async fn fetch(State(state): State<AppState>, body: Bytes) -> Response {
    answer(state, true, body).await
}

async fn manifest(State(state): State<AppState>) -> Response {
    (
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            ),
            (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")),
        ],
        state.index.manifest_json.clone(),
    )
        .into_response()
}

/// Public files are addressed by hash, so they can be cached forever.
async fn blob(State(state): State<AppState>, UrlPath(hash): UrlPath<String>) -> Response {
    match state.index.blobs.get(&hash) {
        Some(data) => (
            [
                (
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("application/octet-stream"),
                ),
                (
                    header::CACHE_CONTROL,
                    HeaderValue::from_static("public, max-age=31536000, immutable"),
                ),
            ],
            data.clone(),
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// All the server ever knows about its users, shown so that anyone can check.
async fn stats(State(state): State<AppState>) -> Response {
    axum::Json(serde_json::json!({
        "documents": state.index.manifest.documents,
        "rank_requests": state.index.rank_requests.load(Ordering::Relaxed),
        "fetch_requests": state.index.fetch_requests.load(Ordering::Relaxed),
    }))
    .into_response()
}

async fn health() -> &'static str {
    "ok"
}

/// Serve the web interface. The policy stops a script injected through a document's text from
/// reaching any other site. It cannot protect anyone from this server's own page, which is why
/// docs/security.md tells people who distrust the operator to use the command line client or a
/// separately hosted copy of the page.
async fn asset(State(state): State<AppState>, uri: axum::http::Uri) -> Response {
    let path = if uri.path() == "/" {
        "/index.html"
    } else {
        uri.path()
    };
    match state.assets.iter().find(|a| a.path == path) {
        Some(a) => (
            [
                (
                    header::CONTENT_TYPE,
                    HeaderValue::from_static(a.content_type),
                ),
                (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")),
                (
                    header::CONTENT_SECURITY_POLICY,
                    HeaderValue::from_static(
                        "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; \
                         worker-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' \
                         data:; font-src 'self'; base-uri 'none'; form-action 'none'; \
                         frame-ancestors 'none'",
                    ),
                ),
                (
                    header::REFERRER_POLICY,
                    HeaderValue::from_static("no-referrer"),
                ),
                (
                    header::X_CONTENT_TYPE_OPTIONS,
                    HeaderValue::from_static("nosniff"),
                ),
            ],
            a.bytes,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// The HTTP application. `assets` may be empty for an API-only server.
pub fn router(index: Arc<Loaded>, assets: &'static [Asset]) -> Router {
    let limit = |cols: usize| DefaultBodyLimit::max(MAX_QUERIES_PER_REQUEST * cols * 4);
    let rank_limit = limit(index.rank.cols());
    let fetch_limit = limit(index.payload.cols());
    let state = AppState { index, assets };
    Router::new()
        .route("/v1/manifest", get(manifest))
        .route("/v1/blob/{hash}", get(blob))
        .route("/v1/rank", post(rank).layer(rank_limit))
        .route("/v1/fetch", post(fetch).layer(fetch_limit))
        .route("/v1/stats", get(stats))
        .route("/v1/health", get(health))
        .fallback(get(asset))
        .with_state(state)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
}
