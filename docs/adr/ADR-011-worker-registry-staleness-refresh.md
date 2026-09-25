# ADR-011: Refresh Worker Health Timestamp on Every Success

**Status**: Accepted
**Date**: 2026-08-31
**Deciders**: Project owner

## Context

`WorkerEntry::record_success` is called by the gateway on every
successful dispatch to a worker. Its implementation only updated
`last_health_update` inside the branch that transitions a worker from
`Degraded` back to `Healthy`. A worker that remained continuously
`Healthy`, the common and correct case for a working system, never had
this timestamp touched again after construction or its last state
transition, no matter how many real successes it accumulated.

`mark_stale_unavailable`'s own doc comment describes its purpose as
marking workers that have not had a successful routing outcome within
a timeout. The actual check compared `now` against
`last_health_update` alone, with no reference to how many successes
had occurred. Because that timestamp could be frozen at construction
time for a healthy worker, the function's real behavior did not match
its documented intent. A periodic health sweep run against a
long-lived, continuously successful worker could incorrectly mark it
`Unavailable` purely because enough wall clock time had passed since
registration, regardless of ongoing success.

Found during an independent code audit. `mark_stale_unavailable` had
no test coverage before this fix.

## Decision

`record_success` now refreshes `last_health_update` unconditionally on
every call, not only inside the `Degraded` to `Healthy` branch. The
doc comment on `mark_stale_unavailable` was corrected to describe what
the function now actually does, marking workers whose last refresh,
via success, failure threshold crossing, or explicit health set, has
not happened within the timeout.

Two tests were added. One confirms a worker that keeps succeeding is
never marked stale regardless of a zero duration timeout, checked
immediately after the last success, which is a direct test of whether
the timestamp was just refreshed, not a race against wall clock drift.
The other confirms a worker that never records a success after
registration is correctly marked stale once the timeout elapses.

## Consequences

This changes real behavior under a long-running, healthy-but-idle
worker scenario. The severity of the previous bug depended on how
`mark_stale_unavailable` was actually invoked in a running system.
Confirmed by search, nothing in the current codebase calls this
function outside of its own tests, so it was not yet load bearing in
production behavior. This ADR exists to close the gap before that
sweep is wired into a real periodic health check, not to describe a
regression in something already running.

## Alternatives considered

Refreshing the timestamp only on a subset of successes, for example
sampling every Nth call, was not considered seriously. The cost of an
unconditional refresh is a single `Instant::now()` call, and
correctness here should not depend on a sampling heuristic.