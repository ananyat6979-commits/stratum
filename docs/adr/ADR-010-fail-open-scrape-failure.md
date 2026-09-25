# ADR-010: Do Not Treat Scrape Failure as Zero Utilization

**Status**: Accepted
**Date**: 2026-08-31
**Deciders**: Project owner

## Context

`cache-oracle`'s `MetricsCollector` polls each registered worker's
Ollama instance for KV cache utilization on a fixed interval. This
value flows into `SemanticRouter`'s scoring, where lower reported
utilization makes a worker more attractive to route to.

`_fetch_ollama_utilization` wraps its own HTTP call in a try/except
block and returns `0.0` unconditionally on any exception, connection
refused, timeout, malformed JSON, or anything else. This was written
as a convenience so callers would not have to special case every
possible network failure.

`_scrape_worker`, one level up, has its own try/except block, with a
comment stating that a caught exception should be recorded as high
pressure, a conservative choice meant to steer the router away from a
worker whose health is unknown.

These two facts are in direct conflict. Because
`_fetch_ollama_utilization` never lets an exception propagate, the
outer block's exception handler never fires for this failure mode. A
worker that is completely unreachable, crashed, or returning garbage
is scraped, the scrape technically succeeds with a value of `0.0`, and
that worker is recorded as `scrape_success=True` with `kv_utilization=
0.0`, the same signal a genuinely idle, healthy worker would produce.

This was found during an independent code audit of the repository, not
during development. No test file existed for `collector.py` before
this fix, which is consistent with the bug going unnoticed: nothing
exercised the failure path.

## Decision

`_fetch_ollama_utilization` and `_fetch_utilization` now return
`Optional[float]`. A genuine, successful scrape of an idle worker
still returns `0.0`, a real and legitimate reading. A failed scrape
returns `None`, an explicit, distinct signal that the fetch itself did
not succeed.

`_scrape_worker` checks for `None` before treating the result as a
normal reading. On `None`, it records the worker with
`kv_utilization=1.0` (maximum pressure) and `scrape_success=False`,
which is the behavior the original comment already described as
intended.

`test_collector.py` was added, with three tests: a connection failure
is recorded as high pressure and not success, a timeout is recorded
the same way, and a genuinely idle worker, one that returns a
successful response with zero models loaded, still correctly reports
`0.0` and `scrape_success=True`. The third test exists specifically to
confirm the fix does not overcorrect and start treating real idle
readings as failures.

## Consequences

A worker that goes offline during operation will now be correctly
deprioritized by the router rather than favored. This changes routing
behavior under a failure condition that was previously silent and
untested. No prior benchmark or experiment in this project (Phase 1,
Phase 2) is known to have been affected in a way that changes their
conclusions, since both ran against workers that were reachable for
the overwhelming majority of requests, and Phase 2's own accounting
already separately tracked and reported dispatch failures at the
gateway level, not through this collector path. This is noted as an
assumption, not independently re-verified against historical scrape
logs, since those logs were not retained at the granularity needed to
confirm it directly.

## Alternatives considered

Raising a custom exception type from `_fetch_ollama_utilization`
instead of returning `None` was considered. `Optional[float]` was
chosen because it keeps the failure signal within the existing type
system without introducing a new exception class for a single call
site, and because the caller already branches on the return value
rather than relying purely on exception handling.