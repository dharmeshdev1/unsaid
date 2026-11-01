//! A local HTTP gateway, so programs in any language can search privately.
//!
//! The program sends plain JSON to this process on its own machine. This process embeds and
//! encrypts the query, and only ciphertext leaves.

use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::{Query, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
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

/// The gateway is for programs on this machine. Browsers mark requests made by web pages, and a
/// page that rebinds its own domain to this address still has to send that domain as the host.
async fn local_only(State(address): State<String>, request: Request, next: Next) -> Response {
    let headers = request.headers();
    let from_a_page = headers.contains_key(header::ORIGIN)
        || headers
            .get("sec-fetch-site")
            .is_some_and(|site| site != "none");
    let host = headers.get(header::HOST).and_then(|h| h.to_str().ok());
    let host_ok = host.is_some_and(|host| {
        let name = host.rsplit_once(':').map_or(host, |(name, _)| name);
        host == address || matches!(name, "localhost" | "127.0.0.1" | "[::1]")
    });
    if from_a_page || !host_ok {
        return StatusCode::FORBIDDEN.into_response();
    }
    next.run(request).await
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
        "index_id": gateway.engine.index().id(),
        "probes": gateway.plan.probes,
        "pages": gateway.plan.pages,
    }))
}

fn routes(gateway: Gateway, address: &str) -> Router {
    Router::new()
        .route("/search", post(search_post).get(search_get))
        .route("/info", get(info))
        .with_state(gateway)
        .layer(middleware::from_fn_with_state(
            address.to_owned(),
            local_only,
        ))
}

/// Serve on `address` until interrupted.
pub async fn run(engine: Engine, plan: SearchPlan, address: &str) -> Result<(), String> {
    let engine = Arc::new(engine);
    engine.prepare_in_background(plan, 2);
    let gateway = Gateway { engine, plan };
    let app = routes(gateway, address);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|e| format!("{address}: {e}"))?;
    eprintln!("private search gateway on http://{address}  (POST /search {{\"query\": \"...\"}})");
    axum::serve(listener, app).await.map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use tower::ServiceExt;

    use super::*;

    async fn status(headers: &[(&str, &str)]) -> StatusCode {
        async fn ok() -> &'static str {
            "ok"
        }
        let app = Router::new()
            .route("/", get(ok))
            .layer(middleware::from_fn_with_state(
                "127.0.0.1:8378".to_owned(),
                local_only,
            ));
        let mut request = HttpRequest::get("/");
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        let request = request.body(Body::empty()).unwrap();
        app.oneshot(request).await.unwrap().status()
    }

    #[tokio::test]
    async fn programs_on_this_machine_are_served() {
        for host in [
            "127.0.0.1:8378",
            "localhost:8378",
            "localhost",
            "[::1]:8378",
        ] {
            assert_eq!(status(&[("host", host)]).await, StatusCode::OK, "{host}");
        }
    }

    #[tokio::test]
    async fn web_pages_and_other_hosts_are_refused() {
        let refused = [
            vec![("host", "evil.example:8378")],
            vec![],
            vec![
                ("host", "127.0.0.1:8378"),
                ("origin", "https://evil.example"),
            ],
            vec![("host", "127.0.0.1:8378"), ("sec-fetch-site", "cross-site")],
        ];
        for headers in refused {
            assert_eq!(status(&headers).await, StatusCode::FORBIDDEN, "{headers:?}");
        }
    }
}
