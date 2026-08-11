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

## Resolved: semantic_vs_round_robin's latency spread is machine variance, not a code-path effect

The prior version of this section, written after 6 runs, described
this as two discrete latency clusters and closed the investigation
around ruling out three specific STRATUM code paths as the cause of
that clustering. Re-checked against the full 22 committed runs and
corrected: what 6 runs made look like two discrete stacks (47ms vs
141-171ms, nothing between) is, across all 19 clean runs (n_success=49
on both arms, excluding 2 runs broken by the pre-stub_worker topology
and 1 run truncated by the crash documented below), closer to a
continuous, right-skewed spread: 46, 46, 46, 46, 47, 47, 47, 48, 62,
63, 133, 141, 156, 156, 157, 171, 172, 173, 203 (ms, sorted).

The decisive evidence this correction rests on: round_robin's own p50, a strategy with no oracle call, no cache lookup, no registry, only
an atomic increment, swings from 139ms to 187ms across these same 19
runs, a 35% range on the simplest possible code path. If the arm with
no plausible mechanism for variance still varies this much, the
variance is a property of the machine and measurement conditions
(background load, OS scheduling, this repository's already-documented
severe inference-latency variance elsewhere, see skills.md and
benchmarks/README.md), not of SemanticRouter's specific logic.

The three diagnostic trace points added during the original
investigation (HttpSignalsProvider's cache read, SemanticRouter::route()'s
signals fetch, handle_chat_completions's effective_workers computation)
remain correct and remain useful: each measured consistently fast and
stable (255-626us, 4-6us respectively) in the one debug-level run they
were checked against, which is real evidence ruling out gross
inefficiency in those specific paths, even though it was insufficient
on its own to explain the full run-to-run p50 range, because that
range turns out to not be caused by any single request-scoped cost at
all.

This is now closed as: the semantic arm shows real, measurable overhead
relative to round_robin in aggregate (median-of-medians across the 19
clean runs: round_robin ~156ms, semantic ~144ms, overlapping, not
separable at this sample size and duration) with both arms subject to
substantial shared machine-level variance that swamps any per-request
routing-overhead signal at this benchmark's current duration (120s,
~49 requests per arm per run). A statistically defensible answer to
"what is SemanticRouter's routing overhead" requires either
substantially longer runs (more samples per run reduces the CI width
directly) or a controlled environment with less background variance
than this development machine provides. Not pursuing either right now: see "Immediate next step" below for why.

All 22 committed runs remain valid data; none are retracted. This
correction changes only the interpretation, not the underlying
measurements, which were always accurately recorded.

## Operational note: gateway process instability during long benchmark sessions

Both gateway instances have now exited unexpectedly
(STATUS_CONTROL_C_EXIT, 0xc000013a) twice across this benchmark's
sessions, both times the semantic instance, both times with no panic
message, both times during a live benchmark run rather than idle.
This has moved from "possible one-off" to "reproducible enough to
plan around." No root cause identified yet: no panic means this is
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

Affects one committed run directly (`ad62d6d2`, 2026-08-06, both arms
truncated to 15-18 successful requests out of 49 before the crash),
excluded from the clean-run analysis above for that reason, not
retracted.