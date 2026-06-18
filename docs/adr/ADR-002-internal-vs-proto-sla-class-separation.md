# ADR-002: Separate Internal SlaClass from Generated Proto SlaClass

**Status**: Accepted
**Date**: 2026-06-17
**Deciders**: Project owner

## Context

`inference.proto` defines an `SlaClass` enum for wire transmission:
`SLA_CLASS_UNSPECIFIED = 0`, `SLA_CLASS_REALTIME = 1`,
`SLA_CLASS_INTERACTIVE = 2`, `SLA_CLASS_BATCH = 3`. `prost-build` generates
a corresponding Rust enum (`crate::proto::SlaClass`) with matching
discriminants.

Separately, `stratum-gateway::sla` needs an `SlaClass` type for internal
use (specifically for priority) queue ordering in the future router
(REALTIME must compare greater than INTERACTIVE, which must compare
greater than BATCH), and for assigning a class from the `Authorization`
header before any proto type even exists in scope.

The question: should `sla.rs` reuse the generated proto enum directly, or
define its own enum and convert at the boundary?

## Options Considered

### Option A: Reuse the generated proto::SlaClass everywhere
**Pros**: One type, no conversion code, no risk of the two getting out of
sync.
**Cons**: The proto enum's discriminant values are a wire contract:
`SLA_CLASS_REALTIME = 1` must never change once any client depends on it,
per standard protobuf evolution rules. The internal priority-queue
ordering, by contrast, is pure implementation detail that should be free
to change without touching the wire format. Tying them together means a
future wire-format change (e.g. inserting a new SLA tier between existing
ones) could not be made without also being constrant by, or accidentally
corrupting, internal ordering logic, or vice versa, a refactor of
internal priority logic could not be done without proto compatibility
concerns. Additionally, `proto::SlaClass` requires `#[derive(prost::Enumeration)]`
machinery and an `i32` representation that has no inherent ordering
(`Unspecified=0` sorts below everything, which is wrong for a priority
comparison: BATCH, not UNSPECIFIED, should be the floor of real traffic).

### Option B: Define a separate internal SlaClass, convert at the proto boundary
**Pros**: The internal enum can have whatever representation and ordering
serves its actual purpose: `#[repr(u8)]` with `Batch = 0 < Interactive =
1 < Realtime = 2`, deriving `PartialOrd`/`Ord` directly, with no
`Unspecified` variant cluttering comparisons (a request is always
assignable to a concrete class; "unspecified" is a wire-only concept for
forward compatibility, not a real internal state). Wire format changes
and internal logic changes are decoupled. Each can evolve on its own
schedule. The conversion function (`to_proto_sla_class`) is a single,
explicit, testable seam.
**Cons**: Two types representing "the same" concept is more code, and
introduces a real risk: if a fifth SLA tier is added to the proto in the
future, `to_proto_sla_class`'s match statement will fail to compile only
if it's structured to be exhaustive (which it is, by using `match` without
a wildcard arm), but this depends on remembering to do so.

## Decision

Use Option B. `sla::SlaClass` is hand-written, `#[repr(u8)]`,
`Batch < Interactive < Realtime`, with no `Unspecified` variant. The
generated `proto::SlaClass` is used only at the wire boundary. A single
function, `proto::to_proto_sla_class`, performs the conversion and is
the only code in the crate permitted to construct a proto `SlaClass` from
an internal one.

## Consequences

**Positive**:
- Internal priority-queue ordering (`SlaClass: PartialOrd + Ord`) is
  correct by construction and independent of wire numbering
- Wire format can evolve (new tiers, renumbering considerations handled
  per standard protobuf compatibility rules) without touching internal
  comparison logic
- The conversion function is a explicit, single-purpose, fully unit-tested
  seam (`proto::tests::*_maps_to_*_sla` tests) rather than implicit
  coupling spread across the codebase

**Negative**:
- Two enums to maintain conceptually in sync; a reviewer unfamiliar with
  this ADR might initially see this as duplication
- `to_proto_sla_class`'s exhaustive match must be remembered/enforced as
  the proto enum grows, currently relies on Rust's exhaustiveness check
  catching it at compile time if a variant is ever added without updating
  the function, which is a reasonably strong guarantee but worth flagging
  explicitly here for future maintainers

**Neutral**:
- No round-trip conversion (`proto::SlaClass` → `sla::SlaClass`) exists
  yet because nothing currently needs to go that direction. Add one if
  Phase 2 (router) needs to read SLA class back off a deserialized proto.

## Revisit Trigger

Revisit if a fifth SLA tier is added, or if profiling ever shows the
conversion function is a measurable hot-path cost (extremely unlikely. It's a single match arm).
