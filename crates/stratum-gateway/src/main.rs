//! stratum-gateway binary entrypoint.
//!
//! Binds the Axum router built in `ingress.rs` to a TCP listener and
//! serves it. This is intentionally minimal, no config file parsing,
//! no graceful shutdown handling, no structured logging yet. Those are
//! real production requirements (see blueprint Section 3, "Operational
//! Philosophy": every service needs health checks, readiness probes,
//! graceful shutdown) but are deliberately deferred until telemetry.rs
//! and a config story exist, adding them now would be scaffolding
//! ahead of substance.
//!
//! # Configuration via environment variables
//!
//! `STRATUM_ROUTING_STRATEGY` (default `semantic`): `semantic` uses
//! `AppState::with_semantic_router`; `round_robin` uses `AppState::new`.
//! Exists so the SAME binary can be launched twice, with different
//! strategies, for a benchmark that compares them under matched
//! conditions, see `benchmarks/scenarios/semantic_vs_round_robin.yaml`.
//! Not a production feature flag: the production default (no env var
//! set) is still `semantic`, with the safe-fallback behavior described
//! below.
//!
//! `STRATUM_CACHE_ORACLE_URL` (default `http://127.0.0.1:8001`): only
//! consulted when strategy=semantic. Safe even when cache-oracle isn't
//! running, `HttpSignalsProvider`'s staleness handling degrades to
//! neutral/unwarmed signals when the oracle is unreachable, at which
//! point SemanticRouter's own pre-warmup fallback round-robins, an
//! already-tested path (see
//! `pre_warmup_fallback_distributes_evenly_across_workers` in
//! stratum-router).
//!
//! `STRATUM_GATEWAY_PORT` (default `8080`), `STRATUM_EVENT_LOG_PATH`
//! (default `gateway_event_log.redb`), `STRATUM_WORKER_0_URL` (default
//! `http://127.0.0.1:11434`): exist for the same reason as
//! STRATUM_ROUTING_STRATEGY. Running two instances of this binary side
//! by side (one per strategy) needs each instance on its own port, its
//! own event log file, and two processes opening the same redb path
//! concurrently fails; this project hit that exact DatabaseAlreadyOpen
//! error once already, in stratum-gateway's own test suite, before
//! tests were given per-call unique paths, and independently
//! pointable at a worker.

use stratum_gateway::ingress::{build_router, AppState};
use stratum_gateway::telemetry::init_telemetry;
use stratum_router::router::WorkerSpec;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

#[tokio::main]
async fn main() {
    init_telemetry();

    let strategy = env_or("STRATUM_ROUTING_STRATEGY", "semantic");
    let port = env_or("STRATUM_GATEWAY_PORT", "8080");
    let event_log_path = env_or("STRATUM_EVENT_LOG_PATH", "gateway_event_log.redb");
    let worker_0_url = env_or("STRATUM_WORKER_0_URL", "http://127.0.0.1:11434");
    let cache_oracle_url = env_or("STRATUM_CACHE_ORACLE_URL", "http://127.0.0.1:8001");
    let bind_addr = format!("127.0.0.1:{port}");

    let workers = vec![WorkerSpec::new("worker-0", worker_0_url)];

    let state = match strategy.as_str() {
        "round_robin" => {
            println!("stratum-gateway: routing strategy = round_robin");
            AppState::new("gateway-node-0", &event_log_path, workers)
        }
        "semantic" => {
            println!(
                "stratum-gateway: routing strategy = semantic (cache-oracle at {cache_oracle_url})"
            );
            println!(
                "  if cache-oracle isn't reachable there, routing falls back to \
                 round-robin automatically until it is (see main.rs doc comment)"
            );
            AppState::with_semantic_router(
                "gateway-node-0",
                &event_log_path,
                workers,
                cache_oracle_url,
            )
        }
        other => {
            panic!(
                "STRATUM_ROUTING_STRATEGY={other} is not recognized, expected \
                 \"semantic\" or \"round_robin\""
            );
        }
    };

    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(bind_addr.as_str())
        .await
        .unwrap_or_else(|e| {
            panic!("failed to bind to {bind_addr}, is the port already in use? ({e})")
        });
    println!("stratum-gateway listening on http://{bind_addr}");
    println!("try: curl -X POST http://{bind_addr}/v1/chat/completions \\");
    println!(r#"       -H "Content-Type: application/json" \"#);
    println!(
        r#"       -d '{{"model":"phi3:mini","messages":[{{"role":"user","content":"hello"}}],"max_tokens":50}}'"#
    );
    axum::serve(listener, app).await.expect("server error");
}