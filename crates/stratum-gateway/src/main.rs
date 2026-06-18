//! stratum-gateway binary entrypoint.
//!
//! Binds the Axum router built in `ingress.rs` to a TCP listener and
//! serves it. This is intentionally minimal -- no config file parsing,
//! no graceful shutdown handling, no structured logging yet. Those are
//! real production requirements (see blueprint Section 3, "Operational
//! Philosophy": every service needs health checks, readiness probes,
//! graceful shutdown) but are deliberately deferred until telemetry.rs
//! and a config story exist -- adding them now would be scaffolding
//! ahead of substance.

use stratum_gateway::ingress::{build_router, AppState};

#[tokio::main]
async fn main() {
    let state = AppState::new("gateway-node-0");
    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080")
        .await
        .expect("failed to bind to 127.0.0.1:8080 -- is the port already in use?");

    println!("stratum-gateway listening on http://127.0.0.1:8080");
    println!("try: curl -X POST http://127.0.0.1:8080/v1/chat/completions \\");
    println!(r#"       -H "Content-Type: application/json" \"#);
    println!(r#"       -d '{{"model":"phi3:mini","messages":[{{"role":"user","content":"hello"}}],"max_tokens":50}}'"#);

    axum::serve(listener, app)
        .await
        .expect("server error");
}