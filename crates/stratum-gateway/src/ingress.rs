//! HTTP ingress: the request pipeline that wires signing, SLA assignment,
//! rate limiting, and proto transcoding into a running Axum server.
//!
//! # Pipeline (strict order)
//! 1. Read raw body bytes (required before parsing, see ADR-003)
//! 2. Extract `Authorization` header
//! 3. Parse JSON body into [`crate::proto::OpenAiCompatRequest`]
//! 4. Assign SLA class from the auth header (via `proto::to_inference_request`,
//!    which internally calls `sla::assign_sla_class`)
//! 5. Check the rate limiter for that SLA class, reject with 429 if exhausted
//! 6. Build the `InferenceRequest` proto (signs `replay_key` from raw bytes)
//! 7. Return a stub response, there is no router/worker yet to forward to.
//!    Phase 2 replaces step 7 with an actual gRPC call to `stratum-router`.
//!
//! # Why raw bytes are read manually, not via Axum's `Json<T>` extractor
//! `Json<T>` parses and discards the raw bytes in one step. ADR-003 requires
//! `replay_key` to be signed from the exact bytes the client sent, so this
//! handler uses the `Bytes` extractor and parses manually with
//! `serde_json::from_slice`.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json};
use axum::routing::post;
use axum::Router;
use serde_json::json;

use stratum_replay::event_log::AppendOnlyEventLog;
use stratum_router::backpressure::BackpressureController;
use stratum_router::http_signals_provider::HttpSignalsProvider;
use stratum_router::router::{route_and_log, RoundRobinRouter, RouterStrategy, WorkerSpec};
use stratum_router::semantic_router::SemanticRouter;
use stratum_router::worker_registry::WorkerRegistry;

use crate::proto::{to_inference_request, OpenAiCompatRequest};
use crate::rate_limit::RateLimiter;
use crate::sla::assign_sla_class;

/// Shared state injected into every request handler via Axum's `State`
/// extractor. `Arc`-wrapped so cloning it per-request is cheap (refcount
/// bump only): the `RateLimiter` itself is internally mutex-guarded.
#[derive(Clone)]
pub struct AppState {
    pub rate_limiter: Arc<RateLimiter>,
    pub node_id: Arc<str>,
    /// The active routing strategy. `RoundRobinRouter` by default (see
    /// `AppState::new`), `SemanticRouter` when constructed via
    /// `AppState::with_semantic_router`. Held as a trait object so
    /// `handle_chat_completions` doesn't need to know which strategy is
    /// active -- including for `record_outcome`, see
    /// `RouterStrategy::record_outcome`'s doc comment in stratum-router.
    pub router: Arc<dyn RouterStrategy>,
    /// The event log this gateway writes routing decisions to. Shared
    /// across requests, wrapped in Arc since AppState is cloned per-request
    /// by Axum's Router::with_state.
    pub event_log: Arc<AppendOnlyEventLog>,
    /// Static worker list, used directly when `worker_registry` is `None`.
    /// When a registry is present (`with_semantic_router`), the registry's
    /// `routable_workers()` is used instead on every request, so this
    /// field's contents become stale on purpose -- it's only the initial
    /// seed the registry was populated from at construction time. Kept
    /// (rather than removed) so `AppState::new`'s existing static-list
    /// behavior, and every existing test built on it, is unchanged.
    pub workers: Vec<WorkerSpec>,
    /// Health-aware worker registry. `None` for `AppState::new`/
    /// `with_rate_limiter` (the RoundRobinRouter-over-a-static-list path
    /// that predates this field and every existing gateway test depends
    /// on). `Some` for `AppState::with_semantic_router`, which is also
    /// the only constructor that populates it and the only one where
    /// `record_success`/`record_failure` are called from the dispatch
    /// path -- see `handle_chat_completions`.
    pub worker_registry: Option<Arc<WorkerRegistry>>,
    /// HTTP client used to dispatch requests to the routed worker.
    /// Constructed once and cloned (reqwest::Client is internally
    /// Arc-wrapped, so cloning is cheap) rather than built per-request,
    /// which would discard connection pooling on every call.
    pub worker_client: reqwest::Client,
}

impl AppState {
    pub fn new(
        node_id: impl Into<Arc<str>>,
        event_log_path: impl AsRef<std::path::Path>,
        workers: Vec<WorkerSpec>,
    ) -> Self {
        Self::with_rate_limiter(node_id, event_log_path, workers, RateLimiter::with_defaults())
    }

    /// Same as [`AppState::new`], but with an explicit [`RateLimiter`]
    /// instead of the production default.
    ///
    /// Exists so tests can substitute a bucket whose timing behavior is
    /// controlled (e.g. a refill rate low enough that it cannot regenerate
    /// a token during a multi-request warm-up loop, regardless of how much
    /// wall-clock time that loop takes under CI/test-suite CPU contention).
    /// See `ingress.rs`'s rate-limit exhaustion tests below for why this
    /// seam was needed: `RateLimiter::with_defaults()`'s REALTIME bucket
    /// (capacity 10, refill 10/sec) regenerates a token every 100ms, which
    /// a 10-request loop can cross under parallel `cargo test` execution
    /// even though each individual request is fast, turning an expected
    /// 429 into a 502 nondeterministically.
    pub fn with_rate_limiter(
        node_id: impl Into<Arc<str>>,
        event_log_path: impl AsRef<std::path::Path>,
        workers: Vec<WorkerSpec>,
        rate_limiter: RateLimiter,
    ) -> Self {
        let (event_log, worker_client) = Self::open_log_and_client(event_log_path);

        Self {
            rate_limiter: Arc::new(rate_limiter),
            node_id: node_id.into(),
            router: Arc::new(RoundRobinRouter::new()),
            event_log: Arc::new(event_log),
            workers,
            worker_registry: None,
            worker_client,
        }
    }

    /// Constructs an `AppState` with `SemanticRouter` as the active
    /// routing strategy, backed by a live `WorkerRegistry` and an
    /// `HttpSignalsProvider` polling a running cache-oracle instance.
    ///
    /// This is the integration step `docs/SCOPE.md` names as the
    /// project's immediate next step: every piece here (`SemanticRouter`,
    /// `HttpSignalsProvider`, `WorkerRegistry`) already exists and is
    /// tested in isolation in `stratum-router`; this constructor is what
    /// wires them into the gateway's actual request path for the first
    /// time.
    ///
    /// # Arguments
    /// * `initial_workers` - workers registered into the `WorkerRegistry`
    ///   at startup. Unlike `AppState::new`'s static list, this registry
    ///   is mutable afterward (health state changes as requests succeed
    ///   or fail) even though the constructor's input is still a fixed
    ///   list -- there's no dynamic worker discovery yet, only dynamic
    ///   health tracking of a fixed worker set. Full discovery is a
    ///   separate, later piece of work, not blocked on this one.
    /// * `cache_oracle_base_url` - e.g. `"http://127.0.0.1:8001"`. Must
    ///   point at a reachable cache-oracle for oracle signals to be
    ///   anything other than the neutral/unwarmed default -- see
    ///   `HttpSignalsProvider::new`'s doc comment for staleness handling
    ///   if the oracle is unreachable or goes down after startup.
    ///
    /// # Panics
    /// Must be called from within a Tokio runtime (constructs an
    /// `HttpSignalsProvider`, which spawns a background polling task).
    /// Calling this before `#[tokio::main]`'s runtime is active will
    /// panic. Every existing call site (`main.rs`) already satisfies
    /// this; it's noted here because it's the one behavioral difference
    /// from `AppState::new`, which has no such requirement.
    pub fn with_semantic_router(
        node_id: impl Into<Arc<str>>,
        event_log_path: impl AsRef<std::path::Path>,
        initial_workers: Vec<WorkerSpec>,
        cache_oracle_base_url: impl Into<String>,
    ) -> Self {
        let (event_log, worker_client) = Self::open_log_and_client(&event_log_path);

        let registry = Arc::new(WorkerRegistry::new());
        for worker in &initial_workers {
            registry.register(worker.clone());
        }

        // Poll every 2s, treat signals older than 10s as stale (falls back
        // to neutral/unwarmed rather than serving data that may no longer
        // be accurate). Matches the values already manually verified live
        // against a running cache-oracle in stratum-router's manual
        // verification binary (crates/stratum-router/src/main.rs).
        let signals_provider = Arc::new(HttpSignalsProvider::new(
            cache_oracle_base_url,
            Duration::from_secs(2),
            Duration::from_secs(10),
        ));

        let backpressure = Arc::new(BackpressureController::with_defaults());

        let router = Arc::new(SemanticRouter::new(
            Arc::clone(&registry),
            signals_provider,
            backpressure,
        ));

        Self {
            rate_limiter: Arc::new(RateLimiter::with_defaults()),
            node_id: node_id.into(),
            router,
            event_log: Arc::new(event_log),
            workers: initial_workers,
            worker_registry: Some(registry),
            worker_client,
        }
    }

    /// Shared construction logic for the event log and worker HTTP
    /// client, factored out so `with_semantic_router` doesn't duplicate
    /// the 120s-timeout rationale (see below) or the event-log-open
    /// error handling.
    fn open_log_and_client(
        event_log_path: impl AsRef<std::path::Path>,
    ) -> (AppendOnlyEventLog, reqwest::Client) {
        let event_log = AppendOnlyEventLog::open(event_log_path, "gateway-node-0")
            .expect("failed to open event log, check the path is writable");
        // 120s, not 30s. Measured on real hardware: phi3:mini on this dev
        // machine (CPU-only inference, no GPU acceleration active per
        // Ollama's own startup log) takes ~83s total for a 69-token warm
        // response (~1.1s/token, see skills.md). The original 30s was an
        // unmeasured default set before any real inference workload existed
        // to calibrate against, every gateway-mediated request against a
        // real worker on this hardware would time out before Ollama could
        // finish, regardless of model warmth. 120s gives headroom above the
        // measured ~83s warm-call baseline without being unbounded.
        let worker_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("failed to build worker HTTP client");
        (event_log, worker_client)
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
/// and handler execution, though for this handler, only `proto.rs`'s
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
/// , `sla::assign_sla_class` already treats `None` as BATCH, so this is
/// a safe fallback rather than a special error path.
fn extract_auth_header(headers: &HeaderMap) -> Option<&str> {
    headers.get("authorization")?.to_str().ok()
}

/// POST /v1/chat/completions
///
/// # Cancellation safety
/// This handler performs no partial side effects before its first `await`
/// point that would need cleanup if cancelled, rate limiting (`check`) is
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
    // clarity, not a correctness risk, assign_sla_class has no side
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

    // Effective worker list for this request: the health-filtered registry
    // view when one is present (AppState::with_semantic_router), otherwise
    // the static list every other constructor uses unchanged. This is the
    // one place routing decisions and dispatch see different worker sets
    // depending on which AppState constructor built this instance.
    // Timed explicitly: this is the one place per-request cost differs
    // structurally between the semantic and round_robin paths that
    // hasn't yet been measured. registry.routable_workers() clones
    // every WorkerSpec in the registry on every call; round_robin's
    // state.workers.clone() clones a static Vec held once in AppState.
    // Prior investigation (see semantic_router.rs and
    // http_signals_provider.rs diagnostic commits) ruled out route()'s
    // branch selection and HttpSignalsProvider's cache read as the
    // source of the bimodal p50 clustering seen in
    // semantic_vs_round_robin.yaml (six runs, 47ms vs 141-171ms, no
    // spread between). This is the remaining candidate. Safe to remove
    // once the investigation concludes.
    let effective_workers_start = std::time::Instant::now();
    let effective_workers: Vec<WorkerSpec> = match &state.worker_registry {
        Some(registry) => registry.routable_workers(),
        None => state.workers.clone(),
    };
    let effective_workers_us = effective_workers_start.elapsed().as_micros() as u64;
    tracing::debug!(
        stratum.effective_workers_us = effective_workers_us,
        stratum.effective_workers_count = effective_workers.len(),
        "effective workers computed"
    );

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
        &effective_workers,
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

    // Dispatch to the routed worker. Forwards the already-parsed request
    // as an Ollama-compatible /api/generate call. Worker unreachability
    // (connection refused, timeout, non-2xx) is a real, expected failure
    // mode in dev, no assumption here that a worker is actually running.
    let worker_url = format!("{}/api/generate", routing_decision.worker.address);
    let worker_payload = json!({
        "model": parsed.model,
        "prompt": inference_request.prompt,
        "stream": false,
    });

    let dispatch_result = state
        .worker_client
        .post(&worker_url)
        .json(&worker_payload)
        .send()
        .await;

    match dispatch_result {
        Ok(worker_response) if worker_response.status().is_success() => {
            let worker_body: serde_json::Value = worker_response
                .json()
                .await
                .unwrap_or_else(|_| json!({"error": "worker returned non-JSON response"}));

            // Two outcome-recording calls, both only meaningful on the
            // success path and both no-ops (or absent) on the paths that
            // predate this integration:
            //
            // - router.record_outcome(): populates SemanticRouter's
            //   cache-hit index for this (worker, prompt) pair. A no-op
            //   for RoundRobinRouter via the trait's default method --
            //   see RouterStrategy::record_outcome's doc comment.
            // - worker_registry.record_success(): resets the worker's
            //   consecutive-failure counter, restoring Degraded -> Healthy
            //   if applicable. Only present when worker_registry is Some,
            //   i.e. only for AppState::with_semantic_router.
            state
                .router
                .record_outcome(&routing_decision.worker.worker_id, prompt_text);
            if let Some(registry) = &state.worker_registry {
                registry.record_success(&routing_decision.worker.worker_id);
            }

            tracing::info!(
                stratum.replay_key = %inference_request.replay_key,
                stratum.routed_to_worker = %routing_decision.worker.worker_id,
                "request dispatched successfully"
            );

            (
                StatusCode::OK,
                Json(json!({
                    "replay_key": inference_request.replay_key,
                    "sla_class": sla_class.as_str(),
                    "routed_to_worker": routing_decision.worker.worker_id,
                    "routing_score": routing_decision.score,
                    "routing_reason": routing_decision.reason,
                    "status": "dispatched",
                    "worker_response": worker_body,
                })),
            )
                .into_response()
        }
        Ok(worker_response) => {
            let status = worker_response.status();
            if let Some(registry) = &state.worker_registry {
                registry.record_failure(&routing_decision.worker.worker_id);
            }
            tracing::warn!(
                stratum.replay_key = %inference_request.replay_key,
                stratum.routed_to_worker = %routing_decision.worker.worker_id,
                worker_status = %status,
                "worker returned non-success status"
            );
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({
                    "error": format!("worker returned status {status}"),
                    "replay_key": inference_request.replay_key,
                    "routed_to_worker": routing_decision.worker.worker_id,
                })),
            )
                .into_response()
        }
        Err(e) => {
            // Expected in dev: no worker running at routing_decision.worker.address.
            // Not a bug, a real, honest failure mode being surfaced correctly
            // rather than papered over with a fake success response.
            if let Some(registry) = &state.worker_registry {
                registry.record_failure(&routing_decision.worker.worker_id);
            }
            tracing::warn!(
                stratum.replay_key = %inference_request.replay_key,
                stratum.routed_to_worker = %routing_decision.worker.worker_id,
                error = %e,
                "worker dispatch failed"
            );
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({
                    "error": format!("worker unreachable: {e}"),
                    "replay_key": inference_request.replay_key,
                    "routed_to_worker": routing_decision.worker.worker_id,
                })),
            )
                .into_response()
        }
    }
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

    /// Test state whose REALTIME bucket cannot refill during the test,
    /// no matter how much wall-clock time the test's request loop takes
    /// under CPU contention. Same capacity (10) as `with_defaults()`, so
    /// the exhaustion count under test is unchanged, only the refill rate
    /// differs, isolating "10 tokens consumed" from "how long that took."
    ///
    /// Use this instead of `test_state_with_unreachable_worker()` for any
    /// test asserting exhaustion after an exact request count. See
    /// `AppState::with_rate_limiter`'s doc comment for the failure mode
    /// this replaces.
    fn test_state_with_frozen_realtime_bucket() -> AppState {
        let log_path = std::env::temp_dir().join(format!(
            "stratum-gateway-test-frozen-{}.redb",
            uuid::Uuid::new_v4()
        ));
        let rate_limiter = RateLimiter::new(
            crate::rate_limit::BucketConfig {
                capacity: 10.0,
                refill_rate_per_sec: 0.0,
            },
            crate::rate_limit::BucketConfig {
                capacity: 50.0,
                refill_rate_per_sec: 50.0,
            },
            crate::rate_limit::BucketConfig {
                capacity: 200.0,
                refill_rate_per_sec: 100.0,
            },
        );
        AppState::with_rate_limiter(
            "test-node-0",
            log_path,
            vec![WorkerSpec::new("worker-0", "http://127.0.0.1:0")],
            rate_limiter,
        )
    }

    /// Test state with a deliberately invalid worker address (port 0 is
    /// never a valid connection target). Used for tests that need dispatch
    /// to fail FAST and DETERMINISTICALLY, unlike a real connection-refused
    /// round trip to an unused local port, which still takes measurable
    /// wall-clock time and can introduce enough timing variance to affect
    /// rate-limiter tests that run many requests in a tight loop (the
    /// bucket refills lazily based on elapsed time, see rate_limit.rs).
    fn test_state_with_unreachable_worker() -> AppState {
        let log_path = std::env::temp_dir().join(format!(
            "stratum-gateway-test-unreachable-{}.redb",
            uuid::Uuid::new_v4()
        ));
        AppState::new(
            "test-node-0",
            log_path,
            vec![WorkerSpec::new("worker-0", "http://127.0.0.1:0")],
        )
    }

    /// Test state built via `AppState::with_semantic_router`: SemanticRouter
    /// as the active strategy, a real `WorkerRegistry` seeded with one
    /// deliberately-unreachable worker (port 0, same rationale as
    /// `test_state_with_unreachable_worker`), and an `HttpSignalsProvider`
    /// pointed at a cache-oracle base URL that is never actually queried
    /// by these tests (port 0 as well) -- staleness handling means an
    /// unreachable oracle degrades to neutral/unwarmed signals rather than
    /// erroring, so this is a valid, deterministic test configuration, not
    /// a shortcut. See `HttpSignalsProvider::new`'s doc comment.
    fn test_state_with_semantic_router() -> AppState {
        let log_path = std::env::temp_dir().join(format!(
            "stratum-gateway-test-semantic-{}.redb",
            uuid::Uuid::new_v4()
        ));
        AppState::with_semantic_router(
            "test-node-0",
            log_path,
            vec![WorkerSpec::new("worker-0", "http://127.0.0.1:0")],
            "http://127.0.0.1:0",
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
    async fn valid_request_without_running_worker_returns_502() {
        // Uses test_state_with_unreachable_worker(), not test_state(),
        // test_state() points at 127.0.0.1:11434, the real default Ollama
        // port. If a real Ollama instance happens to be running on the
        // test machine (as it now legitimately is, for manual verification
        //  see the live dispatch success confirmed 2026-07-30), that
        // request would genuinely succeed with a real 200, which is
        // CORRECT behavior, not a test failure,  but it means this
        // test's assertion of 502 was never actually testing "gateway
        // handles unreachable worker correctly," it was silently coupled
        // to whether Ollama happened to be running on the developer's
        // machine at test time. A test's correctness must never depend
        // on an external process's incidental state. Fixed to use the
        // deliberately-invalid port 0 worker, same as every other
        // dispatch-failure test in this file.
        let app = build_router(test_state_with_unreachable_worker());
        let body =
            r#"{"model":"phi3:mini","messages":[{"role":"user","content":"hi"}],"max_tokens":50}"#;

        let response = app.oneshot(json_request(body, None)).await.unwrap();

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn malformed_json_returns_400() {
        let app = build_router(test_state_with_unreachable_worker());
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
        let state = test_state_with_frozen_realtime_bucket();
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
                StatusCode::BAD_GATEWAY,
                "request {i} should pass rate limiting and reach dispatch (which fails, no worker running)"
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
        let state = test_state_with_frozen_realtime_bucket();
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
            StatusCode::BAD_GATEWAY,
            "BATCH bucket must be unaffected by REALTIME exhaustion \
             (reaches dispatch, which fails fast, port 0 is never valid, \
             rather than being rejected by rate limiting)"
        );
    }

    #[tokio::test]
    async fn rate_limiting_rejects_before_attempting_dispatch() {
        // Isolates the rate-limiter's own behavior from dispatch outcome:
        // the 11th REALTIME request must be rejected with 429 specifically,
        // not 502, proving rate limiting happens strictly before dispatch
        // is attempted, regardless of whether a worker is reachable.
        let state = test_state_with_frozen_realtime_bucket();
        let body =
            r#"{"model":"phi3:mini","messages":[{"role":"user","content":"hi"}],"max_tokens":50}"#;

        for _ in 0..10 {
            let app = build_router(state.clone());
            app.oneshot(json_request(body, Some("Bearer rt-abc123")))
                .await
                .unwrap();
        }

        let app = build_router(state.clone());
        let response = app
            .oneshot(json_request(body, Some("Bearer rt-abc123")))
            .await
            .unwrap();

        assert_eq!(
            response.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "11th request must be 429 (rate limited), not 502 (dispatch failure), \
             proving rate limiting happens before dispatch is attempted"
        );
    }

    // ---- SemanticRouter / WorkerRegistry integration ----
    //
    // These tests exercise AppState::with_semantic_router specifically,
    // through the real HTTP pipeline (build_router + oneshot), not just
    // stratum-router's own unit tests. That distinction matters: the
    // trait-object wiring (record_outcome_through_trait_object_reaches_
    // cache_hit_index in stratum-router/src/semantic_router.rs) proves
    // the mechanism works in isolation; these prove the gateway actually
    // calls it, on the real request path, with a real WorkerRegistry
    // whose state changes are then visible to routing on the *next*
    // request -- the actual end-to-end claim docs/SCOPE.md's "immediate
    // next step" was about.

    #[tokio::test]
    async fn semantic_router_state_starts_with_one_healthy_worker() {
        let state = test_state_with_semantic_router();
        let registry = state
            .worker_registry
            .as_ref()
            .expect("with_semantic_router must populate worker_registry");
        assert_eq!(registry.len(), 1);
        assert_eq!(
            registry.health("worker-0"),
            Some(stratum_router::worker_registry::WorkerHealth::Healthy)
        );
    }

    #[tokio::test]
    async fn semantic_router_dispatch_failures_degrade_then_exclude_worker() {
        // Full cycle through the real HTTP path: dispatch failures against
        // the deliberately-unreachable worker (port 0) accumulate in the
        // registry via handle_chat_completions' new record_failure calls,
        // and once the worker crosses the Unavailable threshold (10
        // consecutive failures, see WorkerRegistry::record_failure),
        // routable_workers() excludes it -- so routing itself fails
        // closed (503, "no workers available") rather than continuing to
        // dispatch-and-fail (502) against a worker already known to be down.
        let state = test_state_with_semantic_router();
        let body =
            r#"{"model":"phi3:mini","messages":[{"role":"user","content":"hi"}],"max_tokens":50}"#;

        // Requests 1-2: still Healthy (threshold is 3 consecutive failures).
        for i in 0..2 {
            let app = build_router(state.clone());
            let response = app.oneshot(json_request(body, None)).await.unwrap();
            assert_eq!(
                response.status(),
                StatusCode::BAD_GATEWAY,
                "request {i}: worker still routable while Healthy, dispatch fails (port 0)"
            );
        }

        let registry = state.worker_registry.as_ref().unwrap();
        assert_eq!(
            registry.health("worker-0"),
            Some(stratum_router::worker_registry::WorkerHealth::Healthy),
            "2 consecutive failures must not yet degrade the worker"
        );

        // Requests 3-9: crosses into Degraded (>=3 failures), still routable.
        for i in 2..9 {
            let app = build_router(state.clone());
            let response = app.oneshot(json_request(body, None)).await.unwrap();
            assert_eq!(
                response.status(),
                StatusCode::BAD_GATEWAY,
                "request {i}: worker Degraded but still routable (weight_multiplier > 0)"
            );
        }
        assert_eq!(
            registry.health("worker-0"),
            Some(stratum_router::worker_registry::WorkerHealth::Degraded),
            "3-9 consecutive failures must degrade, not yet exclude, the worker"
        );

        // Request 10: the 10th consecutive failure crosses the Unavailable
        // threshold. This request itself still dispatches (worker was
        // routable when routing ran) and fails with 502; the state change
        // to Unavailable only takes effect for requests *after* this one.
        let app = build_router(state.clone());
        let tenth = app.oneshot(json_request(body, None)).await.unwrap();
        assert_eq!(tenth.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            registry.health("worker-0"),
            Some(stratum_router::worker_registry::WorkerHealth::Unavailable),
            "10th consecutive failure must make the worker Unavailable"
        );

        // Request 11: routing now sees zero routable workers (the only
        // registered worker is Unavailable) and fails closed with 503,
        // never reaching dispatch at all.
        let app = build_router(state.clone());
        let eleventh = app.oneshot(json_request(body, None)).await.unwrap();
        assert_eq!(
            eleventh.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "with the only worker Unavailable, routing must fail closed (503) \
             rather than attempt dispatch against a worker already known to be down"
        );
    }

    #[tokio::test]
    async fn round_robin_path_never_touches_worker_registry() {
        // Regression guard for the additive design: AppState::new (the
        // RoundRobinRouter path every pre-existing test in this file
        // depends on) must have worker_registry == None, and dispatch
        // failures against it must not panic or behave differently now
        // that the registry-recording calls exist in the dispatch match
        // arms -- they're guarded by `if let Some(registry)`, this proves
        // that guard actually works when the value is None, not just that
        // it compiles.
        let state = test_state_with_unreachable_worker();
        assert!(state.worker_registry.is_none());

        let app = build_router(state.clone());
        let body =
            r#"{"model":"phi3:mini","messages":[{"role":"user","content":"hi"}],"max_tokens":50}"#;
        let response = app.oneshot(json_request(body, None)).await.unwrap();

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }
}
