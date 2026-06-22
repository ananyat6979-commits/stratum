//! HTTP ingress: the request pipeline that wires signing, SLA assignment,
//! rate limiting, and proto transcoding into a running Axum server.
//!
//! # Pipeline (strict order)
//! 1. Read raw body bytes (required before parsing — see ADR-003)
//! 2. Extract `Authorization` header
//! 3. Parse JSON body into [`crate::proto::OpenAiCompatRequest`]
//! 4. Assign SLA class from the auth header (via `proto::to_inference_request`,
//!    which internally calls `sla::assign_sla_class`)
//! 5. Check the rate limiter for that SLA class — reject with 429 if exhausted
//! 6. Build the `InferenceRequest` proto (signs `replay_key` from raw bytes)
//! 7. Return a stub response — there is no router/worker yet to forward to.
//!    Phase 2 replaces step 7 with an actual gRPC call to `stratum-router`.
//!
//! # Why raw bytes are read manually, not via Axum's `Json<T>` extractor
//! `Json<T>` parses and discards the raw bytes in one step. ADR-003 requires
//! `replay_key` to be signed from the exact bytes the client sent, so this
//! handler uses the `Bytes` extractor and parses manually with
//! `serde_json::from_slice`.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json};
use axum::routing::post;
use axum::Router;
use serde_json::json;

use crate::proto::{to_inference_request, OpenAiCompatRequest};
use crate::rate_limit::RateLimiter;
use crate::sla::assign_sla_class;

/// Shared state injected into every request handler via Axum's `State`
/// extractor. `Arc`-wrapped so cloning it per-request is cheap (refcount
/// bump only) — the `RateLimiter` itself is internally mutex-guarded.
#[derive(Clone)]
pub struct AppState {
    pub rate_limiter: Arc<RateLimiter>,
    pub node_id: Arc<str>,
}

impl AppState {
    pub fn new(node_id: impl Into<Arc<str>>) -> Self {
        Self {
            rate_limiter: Arc::new(RateLimiter::with_defaults()),
            node_id: node_id.into(),
        }
    }
}

/// Builds the Axum router with all routes wired to their handlers.
///
/// Kept separate from `main()` so tests can construct the router and
/// drive it with `tower::ServiceExt::oneshot` without binding a real
/// TCP listener.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .with_state(state)
}

/// Returns the current wall-clock time in nanoseconds since the Unix epoch.
///
/// Separated into its own function so tests can verify handler logic
/// without depending on real time passing between request construction
/// and handler execution — though for this handler, only `proto.rs`'s
/// determinism tests (which take an explicit timestamp parameter) need
/// that control. This function is the one and only place "real now"
/// enters the gateway.
fn now_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before Unix epoch")
        .as_nanos() as i64
}

/// Extracts the `Authorization` header value as a `&str`, if present and
/// valid UTF-8. Malformed (non-UTF-8) header values are treated as absent
/// — `sla::assign_sla_class` already treats `None` as BATCH, so this is
/// a safe fallback rather than a special error path.
fn extract_auth_header(headers: &HeaderMap) -> Option<&str> {
    headers.get("authorization")?.to_str().ok()
}

/// POST /v1/chat/completions
///
/// # Cancellation safety
/// This handler performs no partial side effects before its first `await`
/// point that would need cleanup if cancelled — rate limiting (`check`) is
/// synchronous and either fully succeeds or fully fails atomically, and
/// no I/O occurs before it. If the client disconnects after rate limiting
/// succeeds but before the response is sent, the consumed token is not
/// returned (consistent with `RateLimiter::check`'s documented contract:
/// rate limiting must be conservative under cancellation).
async fn handle_chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let parsed: OpenAiCompatRequest = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("invalid request body: {e}") })),
            )
                .into_response();
        }
    };

    let auth_header = extract_auth_header(&headers);

    // SLA class is assigned twice in this function's call graph: once here
    // (to decide rate limiting) and once inside to_inference_request (to
    // populate the proto field). Both calls are pure and deterministic
    // over the same auth_header, so this is intentional duplication for
    // clarity, not a correctness risk — assign_sla_class has no side
    // effects and is cheap (a few string comparisons).
    let sla_class = assign_sla_class(auth_header);

    if !state.rate_limiter.check(sla_class) {
        tracing::warn!(
            stratum.sla_class = %sla_class.as_str(),
            stratum.rate_limit_allowed = false,
            "request rejected: rate limit exceeded"
        );
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({
                "error": "rate limit exceeded",
                "sla_class": sla_class.as_str(),
            })),
        )
            .into_response();
    }

    let inference_request =
        to_inference_request(&body, &parsed, auth_header, now_ns(), &state.node_id);

    tracing::info!(
        stratum.replay_key = %inference_request.replay_key,
        stratum.sla_class = %sla_class.as_str(),
        stratum.rate_limit_allowed = true,
        stratum.ingress_node_id = %state.node_id,
        "request accepted"
    );

    // Stub response: no router/worker exists yet to forward this to.
    // Phase 2 replaces this block with a gRPC call to stratum-router and
    // returns its actual inference result instead.
    (
        StatusCode::OK,
        Json(json!({
            "replay_key": inference_request.replay_key,
            "sla_class": sla_class.as_str(),
            "prompt_echo": inference_request.prompt,
            "status": "accepted_no_router_yet",
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt; // for `oneshot`

    fn test_state() -> AppState {
        AppState::new("test-node-0")
    }

    fn json_request(body: &str, auth: Option<&str>) -> Request<Body> {
        let mut builder = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json");

        if let Some(auth_value) = auth {
            builder = builder.header("authorization", auth_value);
        }

        builder.body(Body::from(body.to_string())).unwrap()
    }

    #[tokio::test]
    async fn valid_request_returns_200() {
        let app = build_router(test_state());
        let body =
            r#"{"model":"phi3:mini","messages":[{"role":"user","content":"hi"}],"max_tokens":50}"#;

        let response = app.oneshot(json_request(body, None)).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn malformed_json_returns_400() {
        let app = build_router(test_state());
        let response = app
            .oneshot(json_request("not valid json", None))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn missing_messages_field_returns_400() {
        let app = build_router(test_state());
        // "messages" is a required field on OpenAiCompatRequest with no
        // #[serde(default)] -- omitting it must fail to parse, not panic
        // or silently default to an empty prompt.
        let body = r#"{"model":"phi3:mini"}"#;

        let response = app.oneshot(json_request(body, None)).await.unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn realtime_class_rate_limit_exhausts_after_default_capacity() {
        // Default REALTIME bucket capacity is 10 (see rate_limit::RateLimiter::with_defaults).
        // The 11th immediate request must be rejected with 429.
        let state = test_state();
        let body =
            r#"{"model":"phi3:mini","messages":[{"role":"user","content":"hi"}],"max_tokens":50}"#;

        for i in 0..10 {
            let app = build_router(state.clone());
            let response = app
                .oneshot(json_request(body, Some("Bearer rt-abc123")))
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::OK,
                "request {i} should succeed within capacity"
            );
        }

        let app = build_router(state.clone());
        let response = app
            .oneshot(json_request(body, Some("Bearer rt-abc123")))
            .await
            .unwrap();

        assert_eq!(
            response.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "11th immediate request should exceed default REALTIME capacity of 10"
        );
    }

    #[tokio::test]
    async fn different_sla_classes_share_state_but_have_independent_buckets() {
        // Regression guard: cloning AppState per-request (Router::with_state
        // clones into each call) must NOT reset the underlying RateLimiter,
        // since it's Arc-wrapped. Exhaust REALTIME, confirm BATCH is unaffected
        // using the SAME AppState instance across both routers.
        let state = test_state();
        let body =
            r#"{"model":"phi3:mini","messages":[{"role":"user","content":"hi"}],"max_tokens":50}"#;

        for _ in 0..10 {
            let app = build_router(state.clone());
            app.oneshot(json_request(body, Some("Bearer rt-abc123")))
                .await
                .unwrap();
        }

        let app = build_router(state.clone());
        let exhausted = app
            .oneshot(json_request(body, Some("Bearer rt-abc123")))
            .await
            .unwrap();
        assert_eq!(exhausted.status(), StatusCode::TOO_MANY_REQUESTS);

        let app = build_router(state.clone());
        let batch_response = app
            .oneshot(json_request(body, None)) // None auth -> BATCH
            .await
            .unwrap();
        assert_eq!(
            batch_response.status(),
            StatusCode::OK,
            "BATCH bucket must be unaffected by REALTIME exhaustion"
        );
    }
}
