# ADR-005: Use redb Instead of lmdb for the Replay Event Log

**Status**: Accepted
**Date**: 2026-06-24
**Deciders**: Project owner

## Context

`stratum-replay` needs an embedded key-value store for the append-only
event log. The requirements are: ACID writes, ordered key scanning
(to read events in Lamport timestamp order), and single-node local
storage (no network, no replication).

LMDB was the original design choice (referenced in RFC-001 and the
`event_log.rs` module doc comment). After implementation, it failed
to link on this project's platform.

## What Failed

`lmdb-sys` (the Rust binding to LMDB) compiles LMDB's C source and
links the resulting object files into the final binary. On Windows
with MSVC, LMDB's `mdb_env_setup_locks` function calls
`InitializeSecurityDescriptor` and `SetSecurityDescriptorDacl`, which
are exported from `advapi32.lib`. However, `lmdb-sys`'s build script
does not declare `advapi32` as a required Windows library via
`println!("cargo:rustc-link-lib=advapi32")`. The MSVC linker therefore
cannot resolve these symbols, producing:

```
LNK2019: unresolved external symbol __imp_InitializeSecurityDescriptor
LNK2019: unresolved external symbol __imp_SetSecurityDescriptorDacl
LNK1120: 2 unresolved externals
```

This is a bug in `lmdb-sys`, not in STRATUM's code. A workaround
exists (add a `build.rs` that emits the missing `cargo:rustc-link-lib`
directive), but it requires maintaining a fork or overlay of `lmdb-sys`
solely to fix a build script omission. This is disproportionate effort
for the payoff.

## Options Considered

### Option A: Fork lmdb-sys and add the missing link directive
**Pros**: Preserves LMDB semantics exactly as described in RFC-001.
**Cons**: Requires maintaining a C-toolchain-dependent fork indefinitely.
Every LMDB upstream release requires manual rebase. This is the same
class of fragility that led to abandoning `protobuf-src` (ADR-001).

### Option B: Use redb (pure-Rust embedded database)
**Pros**:
- Zero C dependencies: pure Rust, no MSVC/gcc/clang required, no
  platform-specific linker issues
- ACID transactions: each write is a committed transaction
- Ordered key scanning: `TableDefinition<(u64, u128), &[u8]>` with
  composite keys (lamport_ts, event_id) scans in correct causal order
- Single-file storage: simpler than LMDB's two-file layout
- `table.len()` and `table.iter()` cover all access patterns needed
**Cons**:
- Not LMDB. The original design specified LMDB explicitly. Any
  documentation or external references to "LMDB-backed event log"
  must be updated.
- redb is less battle-tested than LMDB in production systems.
  For a research/portfolio project at Phase 2 scale, this is acceptable.

### Option C: Use SQLite via rusqlite
**Pros**: Extremely mature, widely understood.
**Cons**: SQL is overengineered for an append-only sequential log.
rusqlite also has a C dependency (though it bundles SQLite source,
so linking is handled correctly, unlike lmdb-sys).

## Decision

Use redb (v2). The event log's access patterns (append-only writes,
sequential scan, range scan by timestamp) are a perfect fit for redb's
table abstraction. The composite key `(u64, u128)`: (lamport_ts,
event_id), provides correct causal ordering without a custom
comparator.

RFC-001's design intent is preserved: the event log is still an
ordered, ACID, append-only store keyed by Lamport timestamp. The
storage backend changed; the semantic contract did not.

## Consequences

**Positive**:
- Zero C toolchain dependency for the replay event log
- No platform-specific linker issues on any supported platform
- Clean API: redb's typed `TableDefinition` catches key/value type
  mismatches at compile time, which LMDB's untyped byte slices do not
- Single-file storage is simpler to manage and back up than LMDB's
  two-file layout

**Negative**:
- RFC-001 and the original `event_log.rs` module doc comment
  referenced LMDB explicitly, updated in the implementation
- redb is less mature than LMDB. If a production deployment later
  requires LMDB's specific performance characteristics (read-heavy
  workloads with memory-mapped I/O at scale), revisit this decision

**Neutral**:
- redb v2 vs v4: Cargo resolved v2.6.3 (the highest v2 compatible
  with the `= "2"` version requirement). v4 introduced breaking API
  changes. Pin to v2 until a planned migration.

## Pattern

This is the second time a C-toolchain-dependent dependency has been
replaced with a pure-Rust alternative after a platform failure:

| Rejected | Replacement | Reason |
|----------|-------------|--------|
| protobuf-src (ADR-001) | protox | CMake + abseil-cpp build failed |
| lmdb-sys | redb | advapi32 linker symbols unresolved on MSVC |

The pattern is consistent: for build-time or link-time C dependencies,
pure-Rust alternatives should be the first choice, not the fallback
after the C path fails.

## Revisit Trigger

Revisit if: (a) a production deployment requires LMDB's specific
performance profile and redb cannot meet SLOs, or (b) lmdb-sys fixes
the Windows MSVC linker issue in a future release.