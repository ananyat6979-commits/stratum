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

| `experiment-engine`: mSPRT core | `MSPRTConfig`/`MSPRTState`, the closed-form mixture-prior sequential test (Johari/Pekelis/Walsh 2015, Robbins 1970). O(1) per-observation update. Type-I error control (Ville's inequality) empirically confirmed via Monte Carlo, not just formula-verified: three alpha levels plus one alternate tau, each checked against both the proven Ville ceiling (hard, non-negotiable) and a directly measured asymptotic rate (soft, informational). All three alpha levels' expected rates are now directly measured, not assumed: alpha=0.05 -> 0.036, alpha=0.10 -> 0.0696, alpha=0.01 -> 0.009. All three initially shipped as unverified placeholders equal to alpha itself; two were wrong on first honest measurement and caught by the test itself, not assumed correct. The tau=4.0 case still checks only the hard Ville ceiling, no direct rate measurement exists for that configuration. | 25 tests, ~16 minutes full run (5000-simulation Monte Carlo per configuration, this is real computational cost, not a fast unit-test suite, run deliberately, not on every casual change) |
| `experiment-engine`: AIPW estimator | `estimate_ate()`, augmented inverse propensity weighting for the average treatment effect (Robins/Rotnitzky/Zhao 1994). The actual "doubly robust" claim, unbiased when either the propensity model or the outcome model is misspecified, not just when both are correct, empirically confirmed via Monte Carlo across both misspecification scenarios independently, plus a bias-detection sanity check confirming the test suite has real power to catch a violation. Both misspecification scenarios passed on the first honest measurement, no tolerance adjustment needed. | 13 tests (9 in test_estimator.py: 3 validation, 3 formula, 3 standard-error, all written in the same original commit, none pre-existing; 4 Monte Carlo double-robustness tests). Combined experiment-engine suite: 55 tests total (25 mSPRT + 17 Experiment + 13 estimator), independently re-collected and confirmed via `pytest --collect-only` against a fresh checkout, not carried forward from an earlier miscount. |

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
| `eval-fabric`, `reliability-model`, `synthgen` (Python) | `pyproject.toml` only, zero implementation files | Not started. No `cusum.py`, `survival.py` exist. Any prior document citing these paths at a specific proficiency level was describing planned work, not completed work. |
| Custom Raft, mSPRT sequential testing, doubly-robust causal estimation, synthetic data generation, NUMA-aware scheduling | Not started | Real, well-specified ideas in the original design blueprint. None require the backend-choice resolution above except scheduling/chaos, so these are legitimate next-phase candidates once the wiring below is finished. |

## Resolved: semantic_vs_round_robin's latency spread is machine variance, not a code-path effect

The prior version of this section, written after 6 runs, described
this as two discrete latency clusters and closed the investigation
around ruling out three specific STRATUM code paths as the cause of
that clustering. Re-checked against the full 22 committed runs, using
benchmarks/harness/aggregate_runs.py's actual output (not hand-tallied
values), corrected: what 6 runs made look like two discrete stacks
(47ms vs 141-171ms, nothing between) is, across the 17 clean runs
(n_success=49 on both arms; of the 22 committed runs, 5 are excluded:
2 broken by the pre-stub_worker topology, 1 truncated by the crash
documented below, and 2 more with a lower n_success than initially
tallied when this section was first written), closer to a continuous,
right-skewed spread: 46, 46, 46, 47, 47, 47, 62, 63, 140, 141, 156,
156, 156, 157, 171, 172, 203 (ms, sorted, semantic arm).

An earlier version of this section stated 19 clean runs and a
19-value sorted list containing a value (133) that no committed run
actually produced, and a run count that did not match
aggregate_runs.py's own output once that script was fixed and finally
run to completion. That was a real error, a sorted list assembled by
hand from partial data across several messages rather than generated
from the committed files directly, exactly the class of mistake this
section already exists to correct once (see the paragraph on 6-vs-22
below). Caught by finally running aggregate_runs.py against all 22
committed files and comparing its literal printed output against this
section's text, rather than trusting that they already matched.

The decisive evidence this correction rests on: round_robin's own p50, a strategy with no oracle call, no cache lookup, no registry, only
an atomic increment, ranges 139.99-187.99ms (17 values, sorted:
140, 140, 140, 140, 141, 141, 141, 156, 156, 156, 156, 156, 156, 156,
157, 157, 188, floating-point-rounded to whole milliseconds in the
prose above) across these same 17 runs, a 34% range on the simplest
possible code path. If the arm with
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
relative to round_robin in aggregate (median across the 17 clean runs:
round_robin 156ms, semantic 140ms, overlapping, not separable at
this sample size and duration) with both arms subject to substantial
shared machine-level variance that swamps any per-request
routing-overhead signal at this benchmark's current duration (120s,
~49 requests per arm per run). A statistically defensible answer to
"what is SemanticRouter's routing overhead" requires either
substantially longer runs (more samples per run reduces the CI width
directly) or a controlled environment with less background variance
than this development machine provides. Not pursuing either right now, see "Immediate next step" below for why.

All 22 committed runs remain valid data; none are retracted. This
correction changes only the interpretation, not the underlying
measurements, which were always accurately recorded. The run-count and
sorted-list correction above is different in kind from that: it is a
fix to a transcription error in this document's own prose, not a
reinterpretation of data, and it was only caught by mechanically
re-running aggregate_runs.py and diffing its literal output against
this section, rather than trusting the two already agreed.

This pooling was checked twice: first against commit messages alone
(`git log --oneline`) across the range spanning all 22 runs, which
was insufficient evidence and treated as such. Second, against the
actual line-by-line diffs (`git log -p`) for every commit touching
main.rs, semantic_router.rs, http_signals_provider.rs, ingress.rs, and
the scenario file across the full range. Confirmed: every change in
that range is either the scenario file's original creation, the
additive second-worker support, or a tracing::debug! call that
performs no work unless RUST_LOG=debug is set, which none of these 19
runs had set (all logged at INFO level only). No commit in range
altered routing logic, timeouts, or worker registration in a way that
would affect measured latency. The pooling is correct on verified
evidence, not inferred from commit message summaries.

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

## Phase 2: SemanticRouter routing-quality benchmark — complete, honest null result

**Status: complete.** A live, checkpointed, resumable mSPRT sequential
test comparing SemanticRouter against RoundRobinRouter under real
Ollama inference (`phi3:mini`), run continuously across 11 real days
(2026-08-17 to 2026-08-28), spanning many separate sessions on
developer hardware with zero dedicated infrastructure budget. See
`benchmarks/harness/phase2_runner.py` for the implementation and
`benchmarks/harness/phase2_full_run.json`'s commit history for the
full, checkpoint-by-checkpoint record.

**Result**: reached `max_observations` (2000) without mSPRT rejecting
the null hypothesis. Final `e_value = 0.160`, against a rejection
threshold of `20.0` (`alpha = 0.05`) — not close at any point in the
run's second half, and the trajectory oscillated within a bounded
band for the majority of the run rather than trending toward
rejection. `mean_difference` (semantic minus round_robin, seconds)
settled at approximately `-0.41s` across all 2000 paired observations
— semantic marginally faster on average, but the effect (if it is one
at all, rather than residual noise) is roughly 50x smaller than the
per-arm noise this test was calibrated against (`sigma ≈ 21.4s`,
measured directly from a real pilot run, not assumed).

**What this result means, stated precisely, not rounded up or down**:
at this sample size, under this workload (a deliberately mixed,
partially-repeated prompt stream, see `phase2_prompt_pool.py`'s design
rationale), and on this hardware, there is no statistically detectable
end-to-end latency benefit from SemanticRouter's cache-hit-aware
routing over simple round-robin. This is a real, negative finding
about the mechanism's practical impact under these specific
conditions — not a claim that the mechanism is broken (Phase 1 already
confirmed SemanticRouter's routing overhead is real but small, and
`stratum-router`'s own 84 tests confirm the cache-hit index and
scoring logic work correctly in isolation), and not a claim that a
real effect couldn't exist under different conditions (see "What this
result does not rule out" below).

**Why the noise floor swamped the signal, if a real effect exists at
all**: this machine's per-request Ollama inference latency variance
is severe and already independently documented elsewhere in this
project (2.8s-337s for an identical prompt, see `skills.md` and
`benchmarks/README.md`). A cache-hit locality benefit, if real, would
plausibly show up as a modest fraction of total inference time saved
on repeat-prompt requests specifically — an effect on the order of
single-digit seconds is entirely plausible as a true value, and is
exactly the scale this run's own measured `mean_difference` (-0.41s)
sits at. But an effect that size is roughly 50x smaller than this
run's noise floor, meaning even a real effect at that magnitude was
never likely to be distinguishable from CPU-contention noise at this
sample size, on this hardware, without either a much longer run or a
quieter measurement environment.

**Operational finding, distinct from the routing-quality question**:
of 4088 total real request-pairs attempted, 2088 (~51%) were skipped
due to non-200 dispatch failures — every single one a `502` from
Ollama itself, never a timeout (`PER_REQUEST_TIMEOUT_SECONDS=180` was
never the binding constraint). This is a real, separate, and
significant finding about running a single local Ollama instance
under sustained, continuous, paired request load over many days: at
this scale, Ollama itself failed to serve roughly half of all attempted
requests, independent of which routing strategy sent them. This is
worth treating as a legitimate production-readiness finding about
running Ollama as a long-lived service, not a benchmark artifact to
explain away.

**What this result does not rule out**: a real effect at a different
workload shape (heavier prompt repetition, larger models where
cache-hit locality matters more, a backend with genuinely lower
baseline variance than local CPU-bound Ollama), or a real effect too
small to matter practically even if statistically detectable at a
much larger sample size. Neither of these was tested here, and this
document does not claim they were.

**Engineering integrity of the run itself, verified, not asserted**:
the checkpointed, resumable design (atomic writes via temp-file +
`os.replace`, exact sufficient-statistics reconstruction on resume,
hard config-mismatch refusal) was built and offline-verified against
`stub_worker.py` before ever touching real Ollama (see
`phase2_runner.py`'s commit history), then survived two real incidents
across the 11-day run without losing data: a checkpoint lineage branch
around pair 335 (two independent resume sessions both continuing from
the same earlier state; resolved by identifying the more-advanced
branch as authoritative, nothing lost, both branches individually
valid) and a stale local file read around pair 1830 (a saved snapshot
lagging behind the live process's actual, more current on-disk state;
resolved by re-reading fresh rather than trusting the stale copy).
Both incidents are recorded in this file's own commit history at the
time they occurred, not retroactively cleaned up.

**Statistical design note**: this experiment used a sequential test
(mSPRT) specifically because the true effect size and this
environment's real variance were both unknown in advance — see
`phase2_power_check.py` for the pre-registered power analysis that
motivated this choice over a fixed-sample-size design, and its own
documented finding that this configuration's tolerance bands needed
correcting against real Monte Carlo measurement before being trusted
(the same discipline `test_msprt_type1_error.py` established for
`msprt.py` itself, applied here to a downstream user of it).