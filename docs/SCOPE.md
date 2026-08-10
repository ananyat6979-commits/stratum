# SCOPE.md: What's Real vs. Deferred

**Status**: Living document. Source of truth for what this repository
actually contains, superseding any aspirational language in earlier
design documents. If this file and a design doc disagree, this file
is correct.

**Why this file exists**: the project's original design blueprint
described a much larger system than one person can build in a single
pass, and several early commits scaffolded named submodule files for
work that was planned but never written. Read on their own, those
filenames implied progress that didn't exist. This file draws the
line explicitly, and the stub-sweep commits immediately preceding this
one removed the files that blurred it. See those commits for the exact
removal list and reasoning.

---

## Real, implemented, tested

| Component | What it actually does | Evidence |
|---|---|---|
| `stratum-gateway` | Full HTTP/2 ingress: request signing, SLA classification, rate limiting, transcoding, real dispatch to a worker over HTTP. `SemanticRouter` is the default routing strategy, backed by a live `WorkerRegistry` with health tracking wired into the real dispatch path. `AppState::new` (RoundRobinRouter, no registry) still exists unchanged for callers that don't need it. | 37 tests (34 + 3 new, including a full request-by-request trace of Healthy -> Degraded -> Unavailable -> routing-fails-closed through the real HTTP path), 2 doctests, manually verified live against a running Ollama instance (real 200 responses, real model output round-tripped). Full local `cargo test` run confirmed 37/37 passing. |
| `stratum-router` | `RoundRobinRouter` and `SemanticRouter`, both wired into the gateway | 84 tests (82 + 2 new proving `RouterStrategy::record_outcome`'s trait-object wiring specifically) |
| `stratum-replay`- event log | Append-only, redb-backed event log with Lamport logical clock | 18 tests |
| `stratum-replay`- "replay" test | Proves a **stateless router re-derives the same routing decision** when re-invoked with the same recorded inputs (`replay_key`, prompt, worker set) | `tests/replay_determinism.rs` |
| `cache-oracle` | Real FastAPI service: KV-pressure prediction (Holt-Winters), FAISS-backed cache-hit indexing, worker registration API | 41 tests, verified by live execution, `faiss-cpu` confirmed building cleanly |

## A precise correction: what "replay" means here

Earlier planning language (and one prior engineering-journal draft)
described a replay engine that reconstructs a *historical* routing
decision from a *recorded oracle-state snapshot*, substituting a mock
model for non-deterministic outputs, without touching a live oracle.
**That component does not exist.** The three files that would have
contained it (`replayer.rs`, `mock_model.rs`, `dependency_graph.rs`)
were empty and have been removed.

What exists instead, and is real: an event log that durably records
routing decisions, and a test proving that `RoundRobinRouter`, which
is a pure function of its inputs, produces the same output when
re-invoked with the same inputs read back from that log. This is a
smaller, easier claim than historical-state reconstruction, and it's
worth being exact about the difference, because the harder version is
one of this project's most-cited pieces of intended signal. If it's
ever built, it belongs in these same three files, for real.

## Deferred, not started

| Component | Status | Why |
|---|---|---|
| `stratum-raft` | Empty crate (`Cargo.toml` + doc-comment-only `lib.rs`) | Not started. Config-plane consensus is real, useful work, but lower priority than finishing what's already 80% wired (see Next below). |
| `stratum-scheduler` | Empty crate | **Structurally blocked**, not just "not yet started": the design (NUMA-aware, predicted-length scheduling) requires backend-internal scheduling hooks (a forkable scheduler, block-table access) that this project's actual inference backend, Ollama, does not expose. This phase needs either a backend change (e.g. a real vLLM deployment) or a redesign around what Ollama can actually offer, before implementation makes sense. |
| `stratum-chaos` | Empty crate | Not started. A reduced taxonomy (process-kill, partition simulation) is achievable against Ollama; the original design's backend-internal fault modes (KV eviction storm, attention OOM) are not, for the same reason as the scheduler. |
| `causal-observer` (Go) | `cmd/observer/main.go` only, proves the Go toolchain builds, nothing else | Not started. |
| `experiment-engine`, `eval-fabric`, `reliability-model`, `synthgen` (Python) | `pyproject.toml` only, zero implementation files | Not started. No `msprt.py`, `estimator.py`, `cusum.py`, or `survival.py` exist. Any prior document citing these paths at a specific proficiency level was describing planned work, not completed work. |
| Custom Raft, mSPRT sequential testing, doubly-robust causal estimation, synthetic data generation, NUMA-aware scheduling | Not started | Real, well-specified ideas in the original design blueprint. None require the backend-choice resolution above except scheduling/chaos, so these are legitimate next-phase candidates once the wiring below is finished. |

## Resolved: the bimodal semantic_vs_round_robin latency is not caused by any traced STRATUM code path

Three diagnostic trace points were added across the semantic request
path and checked against a real run with `RUST_LOG=debug`:

- `HttpSignalsProvider`'s cache read: `signals_fetch_us` consistently
  255-626us across the entire run, both latency regimes.
- `SemanticRouter::route()`'s branch selection: constant throughout,
  `any_warmed` is false on every request since cache-oracle is never
  running in this benchmark, every request takes the identical
  `fallback:round_robin_pre_warmup` path.
- `handle_chat_completions`'s `effective_workers` computation:
  `effective_workers_us` consistently 4-6us across the entire run,
  tiny and stable. This was the leading hypothesis after the first two
  were ruled out; the actual measured cost of cloning two `WorkerSpec`
  values is roughly two orders of magnitude too small to explain a
  p50 swing between ~47ms and ~150ms, and the trace confirms it
  tracks neither latency cluster.

Every piece of STRATUM's own code that sits on the semantic request
path and was instrumented is fast and stable. None of it explains the
observed bimodal p50 clustering (47.00ms exactly in three of six
committed runs, 141-171ms in the other three). The cause is not in
application logic that has been checked. Remaining candidates, none
yet investigated, roughly in order of likely payoff: OS-level thread
scheduling variance on this specific Windows development machine
(consistent with the severe, independently-documented inference
latency variance elsewhere in this project, see skills.md and
benchmarks/README.md), reqwest connection-pool warmup/reuse behavior
differing between the two gateway processes' first N requests, or
something in Tokio's multi-threaded runtime scheduler specifically
under this benchmark's concurrent-arms setup (both arms share one
Python asyncio event loop in compare_interleaved.py, but each targets
a separate OS process, so this candidate would need to be about the
Rust side's own tokio runtime, not the harness).

Diagnostic tracing (3 debug-level trace points, in
http_signals_provider.rs, semantic_router.rs, ingress.rs) is left in
place rather than removed. It is free at info level and above, which
is what every other logged run in this project actually uses, and it
documents a real, completed investigation rather than speculation.

This is closed as "cause not found in application code" rather than
"cause found." All 9 committed benchmark runs remain valid data
documenting the observed behavior; none are retracted or need
re-running. Revisit only if routing-quality work (Phase 2, real
Ollama inference, see below) surfaces the same pattern somewhere it
would matter more.

## Operational note: gateway process instability during long benchmark sessions

Both gateway instances have now exited unexpectedly
(STATUS_CONTROL_C_EXIT, 0xc000013a) twice across this benchmark's
sessions, both times the semantic instance, both times with no panic
message, both times during a live benchmark run rather than idle.
This has moved from "possible one-off" to "reproducible enough to
plan around." No root cause identified yet -- no panic means this is
either an external signal (terminal/session/OS level, not a Rust
panic) or a crash mode that isn't producing a panic message before
exit, which are different problems requiring different fixes. Treat
as a known open item, not resolved by the operational workaround
already in place (keep terminal windows active, verify liveness
immediately before each run). If this recurs a third time, the next
step is running the gateway under a process supervisor that captures
exit reason (e.g. `wintun`/Task Scheduler with failure logging, or
simply capturing stdout/stderr to a file across the whole session)
rather than continuing to diagnose from terminal scrollback alone.