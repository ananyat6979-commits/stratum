# RFC-001: Causal Observability — CausalDecisionEvent and the Replay Event Log

**Status**: Accepted
**Date**: 2026-06-19
**Author**: Project owner
**Implemented by**: Phase 2 (stratum-causal-observer, stratum-replay)

## Problem

Standard observability tells you *what happened*: request X arrived,
was routed to worker Y, took Z milliseconds. It does not tell you *why
the system made that choice*: which oracle signals drove the routing
decision, which weights the bandit assigned at decision time, which
KV pressure reading caused the router to prefer worker Y over worker Z.

Without causal observability, production debugging requires reading
logs and reconstructing causality by hand , archaeology, not engineering.
The deterministic replay system (Phase 2) is only as useful as the
fidelity of what it records. If it records outcomes but not the inputs
that drove them, replaying a historical failure may produce different
routing decisions than the original, making replay useless for debugging.

## Proposed Solution

Every routing decision emits a `CausalDecisionEvent` proto containing:
- The decision itself (which worker was selected)
- All oracle state used in making the decision (captured as a snapshot
  at decision time, not queried from the live oracle during replay)
- The bandit weights at decision time
- The causal parents of this decision (prior events that influenced it)

These events are appended to an LMDB-backed append-only event log with
Lamport timestamps. The replay engine reads this log and can reconstruct
any historical routing decision deterministically by injecting the
recorded oracle state rather than querying the live system.

## Proto Schema

```protobuf
// causal.proto additions (Phase 2)

message CausalDecisionEvent {
  // Lamport logical timestamp. Globally ordered by (lamport_ts, node_id).
  uint64 lamport_ts = 1;

  // Unique event ID (UUID v4).
  string event_id = 2;

  // IDs of events that causally preceded this decision.
  // Empty for ingress events. Routing decisions list the ingress event
  // and the oracle snapshot event as causal parents.
  repeated string dependency_ids = 3;

  // The node that emitted this event.
  string emitter_node_id = 4;

  oneof payload {
    RequestIngressEvent request_ingress = 5;
    RoutingDecisionEvent routing_decision = 6;
    InferenceResponseEvent inference_response = 7;
  }
}

message RequestIngressEvent {
  string replay_key = 1;
  InferenceRequest request = 2;
  // Wall clock at ingress -- preserved for reference, NOT used during replay
  int64 ingress_wall_clock_ns = 3;
}

message RoutingDecisionEvent {
  string replay_key = 1;
  string selected_worker_id = 2;
  double routing_score = 3;
  RoutingScoreComponents score_components = 4;
  // Snapshot of ALL oracle state used in this decision.
  // This is what makes replay possible: the oracle state is recorded
  // at decision time, so replay doesn't re-query the live oracle.
  OracleStateSnapshot oracle_state_snapshot = 5;
  // Bandit weights at decision time.
  RoutingScoreWeights bandit_weights_snapshot = 6;
}

message OracleStateSnapshot {
  map<string, double> kv_pressure_by_worker = 1;
  map<string, double> cache_hit_prob_by_worker = 2;
  uint64 oracle_reading_lamport_ts = 3;
}

message RoutingScoreComponents {
  double cache_hit_prob = 1;
  double inverse_latency = 2;
  double sla_affinity = 3;
  double pressure_avoidance = 4;
}

message RoutingScoreWeights {
  double cache_hit_prob = 1;
  double inverse_latency = 2;
  double sla_affinity = 3;
  double pressure_avoidance = 4;
}

message InferenceResponseEvent {
  string replay_key = 1;
  string worker_id = 2;
  // Recorded verbatim -- during replay, model calls are replaced with
  // a mock that serves this recorded response.
  bytes response_bytes = 3;
  double ttft_ms = 4;
  double total_latency_ms = 5;
  bool was_cache_hit = 6;
  uint32 kv_blocks_used = 7;
}
```

## Non-Determinism Handling in Replay

Sources of non-determinism in the serving path and how replay handles each:

| Source | Replay strategy |
|--------|-----------------|
| Model outputs | Replaced by mock serving `InferenceResponseEvent.response_bytes` |
| Wall clock time | Replaced by logical Lamport clock advancing per event |
| Oracle state | Replaced by `OracleStateSnapshot` recorded at decision time |
| Scheduler contention (concurrent events at same Lamport ts) | Ordered by `(lamport_ts, node_id, event_id)`: a stable total order |

The implication of the last row: replay produces identical routing
decisions but does not reproduce the exact wall-clock interleaving of
concurrent requests. For debugging routing logic, this is sufficient.
For debugging scheduler concurrency bugs, hardware-level event recording
would be required — outside STRATUM's scope.

## Event Log Design

LMDB-backed append-only log (`stratum-replay/src/event_log.rs`):
- Keys: `lamport_ts.to_be_bytes()` (big-endian for correct LMDB ordering)
- Values: `bincode::serialize(&CausalDecisionEvent)` (faster than proto
  for internal logs, acceptable since only Rust code reads the log)
- Transactions: each append is an independent LMDB write transaction

**Why LMDB**: memory-mapped reads are near-zero cost for cached pages,
append-only access pattern matches LMDB's B-tree structure, ACID
guarantees without WAL overhead for append-only workloads. Documented
in ADR-001 (storage) when that ADR is written.

**Why bincode not proto**: the event log is an internal Rust-only
artifact. Proto is the correct choice for the wire format between
services; bincode is correct for internal persistence where language
interoperability is not required and serialization speed matters.
If a future phase requires a non-Rust tool to read the event log
directly, add a proto export layer rather than switching the primary
format.

## Causal DAG Reconstruction

The `dependency_ids` field in `CausalDecisionEvent` encodes the causal
graph. During replay, the causal observer performs a topological sort
over the dependency DAG before feeding events to the replay engine.
This ensures events are processed in causal order, not chronological
order, a necessary distinction when events from different nodes can
have identical Lamport timestamps.

Cycle detection: the topological sort must detect and reject cycles.
A cycle in the causal DAG indicates a bug in event emission (a decision
claiming to depend on an event that came after it) and should halt
replay with an explicit error, not silently produce incorrect output.

## Implementation Plan (Phase 2)

1. Add proto messages above to `causal.proto`
2. Implement `AppendOnlyEventLog` in `stratum-replay/src/event_log.rs`
3. Implement `LogicalClock` (Lamport clock) in `stratum-replay/src/logical_clock.rs`
4. Implement `CausalObserver` Go service consuming events from the router
5. Implement `ReplaySession` in `stratum-replay/src/replayer.rs`
6. Integration test: record 100 routing sessions, replay all 100,
   assert 100% routing decision reproduction rate

The 100% reproduction rate is the Phase 2 exit criterion for replay
correctness. Any failure to reproduce a decision is a bug, not a
tolerance threshold.

## Open Questions

1. **Lamport clock initialization**: should each node start at 0 or at
   a value derived from the wall clock? Starting at 0 means clock values
   are not globally unique across restarts; starting from wall clock ns
   means the clock is monotonic but not Lamport-correct if a node
   receives a message with a higher timestamp before its first local event.
   Resolution: use wall clock ns as the initial value, update on receive
   as standard Lamport: `max(local, received) + 1`.

2. **Log compaction**: the event log is append-only and grows without
   bound. Snapshot-based compaction (similar to Raft log compaction) is
   deferred to Phase 4 when the log size becomes a real operational
   concern. For Phase 2, document the expected log growth rate in the
   runbook and set a size alert.

3. **Replay scope**: should replay reconstruct the full request lifecycle
   (ingress → routing → inference response) or only the routing decision?
   Phase 2 implements routing-decision replay only. Full lifecycle replay
   (including mock model responses) is Phase 2 stretch goal, moved to
   Phase 3 if time-constrained.
