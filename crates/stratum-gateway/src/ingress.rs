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

use stratum_replay::event_log::AppendOnlyEventLog;
use stratum_router::router::{route_and_log, RoundRobinRouter, RouterStrategy, WorkerSpec};

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
    /// The active routing strategy. RoundRobinRouter for now, gateway
    /// doesn't manage a running cache-oracle instance, so SemanticRouter
    /// (which needs live oracle signals to be meaningful) is not yet the
    /// default here. Swapping this to SemanticRouter is a future step
    /// once the gateway has a real worker registry and oracle connection.
    pub router: Arc<dyn RouterStrategy>,
    /// The event log this gateway writes routing decisions to. Shared
    /// across requests, wrapped in Arc since AppState is cloned per-request
    /// by Axum's Router::with_state.
    pub event_log: Arc<AppendOnlyEventLog>,
    /// Static worker list for now, no real worker registry/health
    /// checking wired into the gateway yet. This is a known, deliberate
    /// simplification: routing logic is being proven correct end-to-end
    /// before worker discovery/health machinery is added on top of it.
    pub workers: Vec<WorkerSpec>,
}

impl AppState {
    pub fn new(
        node_id: impl Into<Arc<str>>,
        event_log_path: impl AsRef<std::path::Path>,
        workers: Vec<WorkerSpec>,
    ) -> Self {
        let event_log = AppendOnlyEventLog::open(event_log_path, "gateway-node-0")
            .expect("failed to open event log, check the path is writable");

        Self {
            rate_limiter: Arc::new(RateLimiter::with_defaults()),
            node_id: node_id.into(),
            router: Arc::new(RoundRobinRouter::new()),
            event_log: Arc::new(event_log),
            workers,
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

    // Extract prompt text for routing. InferenceRequest's `prompt` field
    // (built by proto.rs's transcoding) is what SemanticRouter-family
    // strategies would use for cache-hit similarity; RoundRobinRouter
    // (the current default) ignores it entirely.
    let prompt_text = &inference_request.prompt;

    // Route the request. ingress_event_id=0 for now, this gateway
    // does not yet write a RequestIngressEvent to the log before routing
    // (that's the full causal.proto RFC-001 wiring, not yet built here).
    // Using 0 as a placeholder dependency means routing decisions in
    // the event log currently have no real causal parent; this is a
    // known simplification, not a correctness claim about causal chains.
    let (routing_decision, _event) = match route_and_log(
        state.router.as_ref(),
        &inference_request.replay_key,
        prompt_text,
        0u128,
        &state.workers,
        &state.event_log,
    ) {
        Ok(result) => result,
        Err(e) => {
            tracing::warn!(error = %e, "routing failed");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "error": format!("routing failed: {e}"),
                })),
            )
                .into_response();
        }
    };

    // No real worker dispatch yet, stratum-gateway does not forward
    // HTTP requests to Ollama/vLLM workers as of this commit. This is
    // the next integration slice after this one.
    //
    // record_routing_outcome() is NOT called here yet. Per ADR-009, it
    // must be called on SemanticRouter specifically after a request is
    // dispatched, but RoundRobinRouter (this gateway's current default
    // strategy) has no cache-hit index to populate, and downcasting
    // from `dyn RouterStrategy` to check for SemanticRouter at runtime
    // would be solving a problem this commit doesn't have yet. Wire
    // this call in when SemanticRouter becomes the gateway's active
    // strategy, not before. Tracked as a known gap in ADR-009.

    (
        StatusCode::OK,
        Json(json!({
            "replay_key": inference_request.replay_key,
            "sla_class": sla_class.as_str(),
            "prompt_echo": inference_request.prompt,
            "routed_to_worker": routing_decision.worker.worker_id,
            "routing_score": routing_decision.score,
            "routing_reason": routing_decision.reason,
            "status": "routed_no_dispatch_yet",
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
        let log_path = std::env::temp_dir().join(format!(
            "stratum-gateway-test-{}.redb",
            uuid::Uuid::new_v4()
        ));
        AppState::new(
            "test-node-0",
            log_path,
            vec![WorkerSpec::new("worker-0", "http://127.0.0.1:11434")],
        )
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
        // #[serde(default)], omitting it must fail to parse, not panic
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
