//! A local HTTP gateway, so programs in any language can search privately.
//!
//! The program sends plain JSON to this process on its own machine. This process embeds and
//! encrypts the query, and only ciphertext leaves.

use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use serde::Deserialize;
use serde_json::{Value, json};
use unsaid_core::client::SearchPlan;

use crate::engine::Engine;

#[derive(Clone)]
struct Gateway {
    engine: Arc<Engine>,
    plan: SearchPlan,
}

#[derive(Deserialize)]
struct SearchRequest {
    #[serde(alias = "q")]
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    5
}

async fn run_search(gateway: Gateway, request: SearchRequest) -> (StatusCode, Json<Value>) {
    let limit = request.limit.clamp(1, 100);
    let result = tokio::task::spawn_blocking(move || {
        gateway.engine.search(&request.query, limit, gateway.plan)
    })
    .await;
    match result {
        Ok(Ok((outcome, trace))) => (
            StatusCode::OK,
            Json(json!({
                "hits": outcome.hits,
                "corrupted_pages": outcome.corrupted_pages,
                "trace": trace
            })),
        ),
        Ok(Err(e)) => (StatusCode::BAD_GATEWAY, Json(json!({"error": e}))),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        ),
    }
}

async fn search_post(
    State(gateway): State<Gateway>,
    Json(request): Json<SearchRequest>,
) -> (StatusCode, Json<Value>) {
    run_search(gateway, request).await
}

async fn search_get(
    State(gateway): State<Gateway>,
    Query(request): Query<SearchRequest>,
) -> (StatusCode, Json<Value>) {
    run_search(gateway, request).await
}

async fn info(State(gateway): State<Gateway>) -> Json<Value> {
    let m = gateway.engine.index().manifest();
    Json(json!({
        "name": m.name,
        "documents": m.documents,
        "index_id": m.index_id(),
        "probes": gateway.plan.probes,
        "pages": gateway.plan.pages,
    }))
}

/// Serve on `address` until interrupted.
pub async fn run(engine: Engine, plan: SearchPlan, address: &str) -> Result<(), String> {
    let gateway = Gateway {
        engine: Arc::new(engine),
        plan,
    };
    let app = Router::new()
        .route("/search", post(search_post).get(search_get))
        .route("/info", get(info))
        .with_state(gateway);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|e| format!("{address}: {e}"))?;
    eprintln!("private search gateway on http://{address}  (POST /search {{\"query\": \"...\"}})");
    axum::serve(listener, app).await.map_err(|e| e.to_string())
}
