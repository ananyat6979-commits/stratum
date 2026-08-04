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
//! # Routing strategy: SemanticRouter by default, no config file needed
//! Defaults to `AppState::with_semantic_router`, pointed at
//! `STRATUM_CACHE_ORACLE_URL` (default `http://127.0.0.1:8001`, matching
//! the convention already established in stratum-router's manual
//! verification binary). This default is safe even when cache-oracle
//! isn't running: `HttpSignalsProvider`'s staleness handling degrades to
//! neutral/unwarmed signals when the oracle is unreachable, at which
//! point SemanticRouter's own pre-warmup fallback round-robins,
//! already a tested path (see
//! `pre_warmup_fallback_distributes_evenly_across_workers` in
//! stratum-router). So running this binary with no cache-oracle up
//! behaves like the RoundRobinRouter default it replaces; running it
//! with a real cache-oracle up gets real semantic routing, with no
//! separate binary or flag needed to get there.
use stratum_gateway::ingress::{build_router, AppState};
use stratum_gateway::telemetry::init_telemetry;
use stratum_router::router::WorkerSpec;
#[tokio::main]
async fn main() {
    init_telemetry();
    let workers = vec![WorkerSpec::new("worker-0", "http://127.0.0.1:11434")];
    let cache_oracle_url = std::env::var("STRATUM_CACHE_ORACLE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8001".to_string());
    println!("stratum-gateway: routing strategy = semantic (cache-oracle at {cache_oracle_url})");
    println!(
        "  if cache-oracle isn't reachable there, routing falls back to \
         round-robin automatically until it is (see main.rs doc comment)"
    );
    let state = AppState::with_semantic_router(
        "gateway-node-0",
        "gateway_event_log.redb",
        workers,
        cache_oracle_url,
    );
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080")
        .await
        .expect("failed to bind to 127.0.0.1:8080, is the port already in use?");
    println!("stratum-gateway listening on http://127.0.0.1:8080");
    println!("try: curl -X POST http://127.0.0.1:8080/v1/chat/completions \\");
    println!(r#"       -H "Content-Type: application/json" \"#);
    println!(
        r#"       -d '{{"model":"phi3:mini","messages":[{{"role":"user","content":"hello"}}],"max_tokens":50}}'"#
    );
    axum::serve(listener, app).await.expect("server error");
}