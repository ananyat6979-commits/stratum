//! Structured telemetry initialization for stratum-gateway.
//!
//! Emits tracing spans to stdout as JSON using tracing-subscriber.
//! The span field schema follows the stratum.* custom attribute
//! namespace defined in observability/otel/collector.yml (Phase 0
//! infrastructure, not yet running locally -- see blueprint Section 3).
//!
//! When an OTel collector becomes available (Docker or Oracle Cloud),
//! replace the stdout subscriber with an OTLP exporter:
//!   opentelemetry-otlp = "0.x"
//!   opentelemetry = "0.x"
//! The span structure produced here is already OTel-compatible.

use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Initialize the tracing subscriber for this process.
///
/// Call exactly once at process startup, before any spans are created.
/// Subsequent calls are no-ops (tracing-subscriber's set_global_default
/// returns an error if called twice, which we swallow intentionally --
/// test harnesses may call init_telemetry in each test, and that's fine).
///
/// Log level is controlled by RUST_LOG environment variable:
///   RUST_LOG=stratum_gateway=debug,info   (default: info)
pub fn init_telemetry() {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // JSON format: machine-parseable, Loki-compatible, OTel-compatible.
    // When switching to OTLP export, remove this layer and add the
    // opentelemetry tracing layer instead. Span field names stay the same.
    let fmt_layer = fmt::layer()
        .json()
        .with_current_span(true)
        .with_span_list(true)
        .with_target(true)
        .with_thread_ids(false) // not useful without NUMA context
        .with_file(false); // path info is noise in production logs

    // Intentionally ignore the error: if a subscriber is already set
    // (e.g., in tests), we don't need another one.
    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .try_init();
}

/// Canonical field names for the stratum.* OTel attribute namespace.
///
/// These constants are the source of truth for field names. Every span
/// in the gateway must use these constants, not string literals, so that
/// a rename shows up as a compile error across the entire codebase rather
/// than a silent drift between span emission and dashboard queries.
pub mod fields {
    /// The replay_key for this request. Links the span to the replay
    /// event log entry for deterministic debugging.
    pub const REPLAY_KEY: &str = "stratum.replay_key";

    /// SLA class as a string ("realtime", "interactive", "batch").
    /// Used as a Prometheus label and a Loki stream selector.
    pub const SLA_CLASS: &str = "stratum.sla_class";

    /// Whether the rate limiter allowed this request (true) or
    /// rejected it with 429 (false).
    pub const RATE_LIMIT_ALLOWED: &str = "stratum.rate_limit_allowed";

    /// The ingress node ID that signed this request's replay_key.
    pub const INGRESS_NODE_ID: &str = "stratum.ingress_node_id";
}
