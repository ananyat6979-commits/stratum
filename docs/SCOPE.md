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

## Immediate next step

The SemanticRouter-vs-RoundRobinRouter interleaved benchmark
(`benchmarks/scenarios/semantic_vs_round_robin.yaml`,
`benchmarks/harness/compare_interleaved.py`) is built, runs, and
produces real, committed results, but the result itself is not yet
resolved. Six runs against the corrected topology (both workers
answered by `stub_worker.py`, a real always-up HTTP stub, not a
permanently-unreachable address, see that commit's message for why
the first two topologies both broke `WorkerRegistry`'s health tracking
instead of measuring anything) show SemanticRouter's gateway-side
p50 latency clustering into two distinct stacks: exactly 47.00ms in
three runs, 141-171ms in the other three. This is not sampling noise
around one true value, a real spread would not repeat the same figure
to the decimal across independent runs. Something in the request path
takes one of two distinct costs, and which one a given run mostly
lands on has not yet been isolated.

Two plausible mechanisms were traced against the real code and ruled
out: `SemanticRouter::route()`'s branch selection (the `reason` field)
does not change between clusters, `any_warmed` is false for the entire
benchmark since `cache-oracle` is never running, every request takes
the same `fallback:round_robin_pre_warmup` path regardless of which
latency cluster it falls into. `HttpSignalsProvider`'s staleness
cutover (cache starts fresh at construction, goes permanently stale
roughly 10s after process start since no poll ever succeeds) is a
real, confirmed state transition, but both the fresh and stale paths
feed into the same `any_warmed=false` branch, so it cannot explain a
route()-level difference either, at least not through the mechanism
initially suspected.

Two diagnostic trace points were added (commits `37bc02e`, `30c3105`)
to correlate a future run against real per-request state instead of
inferring it from latency alone: cache staleness state and age on
every signals read, and explicit timing around the one call in
`route()` with no equivalent in `RoundRobinRouter::route()`. Neither
has been read against a real run yet, that's the actual next step,
not a fourth hypothesis. Run the benchmark once with `RUST_LOG=debug`
on the semantic instance, then correlate the `signals_fetch_us` and
`signals_cache_stale` trace lines against which half of that run's
requests landed in which latency cluster.