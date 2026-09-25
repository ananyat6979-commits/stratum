# ADR-012: load_range Uses a Real Range Scan, Not Filter After Load All

**Status**: Accepted
**Date**: 2026-08-31
**Deciders**: Project owner

## Context

The event log's redb table is defined with key type `(u64, u128)`,
Lamport timestamp first, then event id. This ordering was chosen
deliberately so that a range of Lamport timestamps could be retrieved
directly from the table's own ordering, without touching entries
outside that range.

`load_range` did not use this. It called `load_all`, which
deserializes every event currently stored in the log, then filtered
the resulting in memory list down to the requested range. This is
correct in output, the existing test `load_range_filters_correctly`
passed before and after this fix, but it is an O(n) operation
regardless of how narrow the requested range is, where n is the total
size of the log, not the size of the range.

Found during an independent code audit. The schema was built for a
fast path that the function using it never took.

## Decision

`load_range` now opens the table directly within a read transaction
and calls `table.range` with the requested Lamport timestamp bounds,
only deserializing events that fall inside that range.

A new test was added that proves this is a real range scan and not
merely a correctness preserving refactor with an unchanged access
pattern underneath. It seeds the log with entries both inside and
outside a requested range, deliberately corrupts the on disk bytes of
entries outside the range so that attempting to deserialize them would
fail, then confirms `load_range` still succeeds and returns only the
in range entries. If the implementation still touched every entry,
this test would fail on the corrupted, out of range data.

## Consequences

Callers requesting a narrow range from a large log will see a real
performance improvement proportional to how much of the log falls
outside their requested range. No caller of `load_range` currently
exists outside of tests, confirmed by search, so this closes a real
correctness and performance gap before the function is exercised by
production code, similar in spirit to ADR-011.