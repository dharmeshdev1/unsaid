//! The HTTP interface, against a real built index.

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use axum::response::Response;
use tower::ServiceExt;
use unsaid_build::{BuildOptions, build};
use unsaid_core::{Document, Manifest};
use unsaid_server::{Loaded, MAX_QUERIES_PER_REQUEST, router};

struct Fixture {
    app: axum::Router,
    manifest: Manifest,
    dir: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn fixture(label: &str) -> Fixture {
    let (n, dim) = (60usize, 8usize);
    let documents: Vec<Document> = (0..n)
        .map(|i| Document {
            title: format!("Document {i}"),
            text: format!("Some text about thing {i}."),
            url: String::new(),
        })
        .collect();
    let embeddings: Vec<f32> = (0..n * dim)
        .map(|i| ((i * 37 % 101) as f32 / 50.0) - 1.0)
        .collect();
    let dir = std::env::temp_dir().join(format!("unsaid-http-{label}-{}", std::process::id()));
    let mut options = BuildOptions::new("test");
    options.dim = 4;
    options.rows = Some(16);
    options.page_bytes = Some(512);
    options.probes = 2;
    options.pages = 2;
    options.prototypes = 2;
    build(&documents, &embeddings, "m", dim, b"model", &options, &dir).unwrap();
    let loaded = Arc::new(Loaded::open(&dir).unwrap());
    let manifest = loaded.manifest().clone();
    Fixture {
        app: router(loaded, &[], false),
        manifest,
        dir,
    }
}

async fn post(f: &Fixture, path: &str, length: Option<usize>, body: Vec<u8>) -> Response {
    let mut request = Request::post(path);
    if let Some(length) = length {
        request = request.header(header::CONTENT_LENGTH, length);
    }
    let request = request.body(Body::from(body)).unwrap();
    f.app.clone().oneshot(request).await.unwrap()
}

async fn send(f: &Fixture, path: &str, body: Vec<u8>) -> Response {
    let length = body.len();
    post(f, path, Some(length), body).await
}

#[tokio::test]
async fn a_whole_number_of_queries_gets_an_answer_of_the_right_size() {
    let f = fixture("sizes");
    for (path, matrix) in [
        ("/v1/rank", &f.manifest.rank),
        ("/v1/fetch", &f.manifest.payload),
    ] {
        for queries in [1, 3] {
            let body = vec![7u8; queries * matrix.cols * 4];
            let response = send(&f, path, body).await;
            assert_eq!(response.status(), StatusCode::OK, "{path}");
            assert_eq!(
                response.headers()[header::CACHE_CONTROL],
                "no-store",
                "an answer must never be cached by a proxy"
            );
            let answer = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            assert_eq!(answer.len(), queries * matrix.rows * 4, "{path}");
        }
    }
}

#[tokio::test]
async fn bad_requests_are_refused() {
    let f = fixture("bad");
    let width = f.manifest.rank.cols * 4;
    let path = "/v1/rank";
    let status = |response: Response| response.status();

    assert_eq!(
        status(send(&f, path, vec![]).await),
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        status(send(&f, path, vec![0; 3]).await),
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        status(send(&f, path, vec![0; width + 1]).await),
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        status(send(&f, path, vec![0; width * (MAX_QUERIES_PER_REQUEST + 1)]).await),
        StatusCode::PAYLOAD_TOO_LARGE
    );
    assert_eq!(
        status(post(&f, path, None, vec![0; width]).await),
        StatusCode::LENGTH_REQUIRED
    );
    // A body that does not match its stated length.
    assert_eq!(
        status(post(&f, path, Some(width), vec![0; width - 4]).await),
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        status(post(&f, path, Some(width), vec![0; width * 2]).await),
        StatusCode::BAD_REQUEST
    );
}

#[tokio::test]
async fn public_files_are_served_by_hash_and_nothing_else_exists() {
    let f = fixture("public");
    let get = |path: String| {
        let app = f.app.clone();
        async move {
            let request = Request::get(path).body(Body::empty()).unwrap();
            app.oneshot(request).await.unwrap().status()
        }
    };
    assert_eq!(get("/v1/manifest".into()).await, StatusCode::OK);
    assert_eq!(get("/v1/health".into()).await, StatusCode::OK);
    let hash = &f.manifest.blobs["routing"].sha256;
    assert_eq!(get(format!("/v1/blob/{hash}")).await, StatusCode::OK);
    assert_eq!(
        get(format!("/v1/blob/{}", "0".repeat(64))).await,
        StatusCode::NOT_FOUND
    );
    // There is no record of requests, so there is nothing to ask for.
    assert_eq!(get("/v1/stats".into()).await, StatusCode::NOT_FOUND);
}
