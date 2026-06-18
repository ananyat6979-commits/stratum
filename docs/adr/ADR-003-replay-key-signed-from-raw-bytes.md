# ADR-003: Sign replay_key from Raw Request Bytes, Not the Parsed Struct

**Status**: Accepted
**Date**: 2026-06-17
**Deciders**: Project owner

## Context

`proto::to_inference_request` constructs the `replay_key` field, the
primary key for the future replay event log, by calling
`signing::compute_replay_key(ingress_timestamp_ns, &body_hash, node_id)`.

The `body_hash` argument can be computed from one of two inputs:

1. The raw HTTP request body, exactly as bytes received over the wire
2. The `OpenAiCompatRequest` struct, after `serde_json` has deserialized it
   (re-serialized to bytes for hashing, if needed)

This decision determines which one.

## Options Considered

### Option A: Hash the parsed struct (re-serialized)
**Pros**: Conceptually simpler, one source of truth (the parsed Rust
value) instead of having to thread raw bytes alongside the parsed result
through the request-handling pipeline.
**Cons**: `serde_json` deserialization is lossy with respect to exact byte
representation: field order in the original JSON is not preserved by
default, insignificant whitespace is discarded, and any field present in
the wire payload but absent from `OpenAiCompatRequest` (since the struct
only models fields STRATUM actually uses) is silently dropped before
hashing ever happens. This means two byte-for-byte *different* client
requests could produce the *same* replay key if they parse to the same
Rust value, and the same logical request could produce different keys
across `serde_json` versions if internal re-serialization behavior changes.
Either failure mode breaks the determinism guarantee the entire replay
system depends on.

### Option B: Hash the raw body bytes, before any parsing
**Pros**: The replay key is tied to exactly what the client transmitted,
with zero dependency on serde's behavior, field whitespace, key ordering,
or which fields `OpenAiCompatRequest` happens to model today. If a field
is added to `OpenAiCompatRequest` in a future phase, replay keys for
already-logged historical requests are unaffected — they were always
keyed on the original bytes, not on today's parsed-struct shape.
**Cons**: Requires the request-handling code (`ingress.rs`, not yet
written) to capture and pass through the raw body bytes alongside the
parsed result, rather than discarding the bytes immediately after
`serde_json::from_slice`. This is a small amount of extra plumbing.

## Decision

Use Option B. `to_inference_request` takes `raw_body: &[u8]` as an explicit
parameter, separate from `parsed: &OpenAiCompatRequest`, and the doc
comment on the function states this requirement directly: "the body must
be hashed before any parsing, so the replay key reflects exactly what the
client sent."

## Consequences

**Positive**:
- `replay_key` determinism is guaranteed independent of `serde_json`
  internals, struct field additions, or JSON formatting differences
  between semantically-identical requests
- Future-proof: adding new fields to `OpenAiCompatRequest` (e.g.
  `temperature`, `top_p` in a later phase) does not retroactively change
  what already-logged replay keys mean

**Negative**:
- `ingress.rs` (not yet written) must read the body as raw bytes first,
  then parse, meaning it cannot use Axum's automatic `Json<T>` extractor
  (which parses and discards raw bytes in one step) and must instead use
  `Bytes` extraction followed by manual `serde_json::from_slice`. This is
  a real, if small, ergonomic cost on the eventual ingress handler.

**Neutral**:
- Two semantically-identical requests with different byte-level formatting
  (e.g. different key ordering, different whitespace) will receive
  *different* replay keys under this design. This is intentional and
  consistent with "replay key reflects exactly what was sent". It is not
  a deduplication mechanism and was never meant to be one.

## Revisit Trigger

Revisit if a future phase needs request deduplication by semantic content
(as opposed to byte-identical content). That would require a *separate*
content hash computed over the parsed/normalized struct, used for
deduplication only, kept distinct from `replay_key` which must remain
byte-faithful for replay correctness.
