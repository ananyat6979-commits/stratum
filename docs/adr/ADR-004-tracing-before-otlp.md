# ADR-004: Emit Tracing Spans via tracing-subscriber Now, Defer OTLP Export

**Status**: Accepted
**Date**: 2026-06-19
**Deciders**: Project owner

## Context

`stratum-gateway` needs to emit observability data with the `stratum.*`
custom attribute namespace specified in the blueprint (Section 3:
"Custom semantic conventions"). Three paths exist:

1. Defer all telemetry until a running OTel collector is available
2. Write a no-op telemetry layer now and fill it in later
3. Emit structured spans via the `tracing` crate to stdout now, with
   field names and structure that are OTel-compatible, and add the
   OTLP exporter later as an additive change

Docker is not available in the current development environment, which
means Jaeger (the trace collector) cannot run locally. Without a running
collector, OTLP export has nowhere to send spans.

## Options Considered

### Option A: Defer telemetry until Docker is available
**Pros**: No code written for infrastructure that isn't running.
**Cons**: Span field names- `stratum.replay_key`, `stratum.sla_class`,
etc. are established by convention, not by code. If multiple services
are built before telemetry is added, field names may diverge silently
across services. Retrofitting consistent field names across six services
simultaneously is more expensive than establishing them in the first
service.

### Option B: No-op telemetry layer (stub)
**Pros**: Satisfies "telemetry exists" without requiring infrastructure.
**Cons**: A no-op layer doesn't test that span fields are correctly
populated, that the init function doesn't panic, or that field names
are valid. It's scaffolding, exactly the pattern this project's
execution discipline forbids.

### Option C: tracing crate to stdout, OTel-compatible structure
**Pros**: The `tracing` crate is already in the dependency tree as a
transitive dependency of `axum`. Making it explicit adds zero new
transitive dependencies. `tracing-subscriber` with a JSON formatter
writes OTel-compatible structured spans to stdout — these are parseable
by Loki (when available) and verifiable locally by reading stdout.
Span field names are defined as Rust constants in `telemetry::fields`,
making them compile-checked across the codebase rather than freeform
strings. Switching to OTLP export is a single `Cargo.toml` addition
and one additional subscriber layer, the span structure is unchanged.
**Cons**: Spans go to stdout rather than a trace store. Not queryable
in Jaeger until the OTLP exporter is added.

## Decision

Use Option C. `telemetry.rs` initializes a `tracing-subscriber` JSON
formatter writing to stdout. Field name constants are defined in
`telemetry::fields`. `ingress.rs` emits `tracing::info!` on accepted
requests and `tracing::warn!` on 429 rejections with all four
`stratum.*` fields populated. `main.rs` calls `init_telemetry()` at
startup.

## Consequences

**Positive**:
- `stratum.replay_key`, `stratum.sla_class`, `stratum.rate_limit_allowed`,
  `stratum.ingress_node_id` are established as compile-checked constants
  before any other service is built, divergence is now a compile error,
  not a convention
- Spans are visible locally via stdout for debugging without any
  infrastructure dependency
- JSON format is compatible with Loki's structured log ingestion when
  the observability stack is available
- Switching to OTLP: add `opentelemetry-otlp` + `opentelemetry` to
  Cargo.toml, add an OTLP layer to the subscriber registry in
  `init_telemetry()`. The span structure is unchanged.

**Negative**:
- Spans go to stdout in development — this is noise if the developer
  doesn't want to see them. Controlled via `RUST_LOG` env var
  (`RUST_LOG=warn` silences info spans).
- Without a running collector, there is no distributed trace to view.
  Spans from different services cannot be correlated until Jaeger is
  available.

**Neutral**:
- `telemetry::fields` constants are strings at runtime. Renaming a
  constant updates all emission sites but does not automatically update
  Grafana dashboards or Loki queries, which use string field names.
  The constants eliminate in-code divergence but cannot eliminate
  infrastructure-config divergence.

## Revisit Trigger

Add the OTLP exporter when Docker or Oracle Cloud free tier is
available and Jaeger can run. The switch is additive, no existing
code needs to change, only `Cargo.toml` and `init_telemetry()`.
