# ADR-001: Use protox Instead of protobuf-src for Protocol Buffer Codegen

**Status**: Accepted
**Date**: 2026-06-17
**Deciders**: Project owner

## Context

`stratum-gateway` needs to generate Rust types from `inference.proto` at
build time using `prost-build`. `prost-build` does not parse `.proto`
files itself, it requires a `protoc` (Protocol Buffer Compiler) binary,
or a pure-Rust alternative, to produce a `FileDescriptorSet` that it then
turns into Rust code.

Three paths exist to get a `FileDescriptorSet`:

1. Require a system-installed `protoc` binary on every contributor's machine
2. Vendor and compile `protoc` from C++ source at build time (`protobuf-src`)
3. Parse `.proto` files directly in pure Rust at build time (`protox`)

This decision was made after option 2 was attempted and failed.

## Options Considered

### Option A: System-installed protoc
**Pros**: Simple `build.rs`, well-trodden path, exact reference implementation
behavior.
**Cons**: Every contributor (and every CI runner) must install `protoc`
separately and keep it on PATH. On Windows specifically, `winget install
protocolbuffers.protoc` does not resolve to a valid package: the binary
must be downloaded manually from GitHub releases and added to PATH by hand.
This reproduces exactly the PATH-management friction already experienced
with `cargo`, `uv`, `go`, and `buf` during Phase 0 setup.

### Option B: protobuf-src (vendored C++ build)
**Pros**: No manual binary install, `cargo build` handles it transparently
in theory. Produces the canonical, spec-compliant `protoc`.
**Cons**: Compiles the full `protoc` C++ codebase (including abseil-cpp:
cordz, hashtablez, status, flags, and related internals) via CMake as part
of the Cargo build. This requires a working MSVC + CMake toolchain on every
machine, takes multiple minutes even when successful, and, in this
project's case, failed outright: `cmake --build ... --target install`
exited with code 1 partway through linking abseil-cpp static libraries,
with no actionable error surfaced in the truncated build output. No
amount of incremental fixing isolated the root cause within a reasonable
time budget.

### Option C: protox (pure-Rust proto parser)
**Pros**: Zero C++ dependency. Parses `.proto` syntax directly in Rust and
hands `prost_build::Config` a `FileDescriptorSet` via `skip_protoc_run()` +
`compile_fds()`. Identical behavior on every platform: no MSVC, no CMake,
no PATH entry required. Build time for the single `inference.proto` file
used by this crate: under a second once dependencies are cached.
**Cons**: Not the reference `protoc` implementation theoretically could
diverge on obscure proto3 edge cases (custom options, extensions, certain
well-known-type behaviors) that `protox` hasn't implemented. For this
project's proto usage (plain messages, one enum, no extensions, no custom
options), no such gap was encountered.

## Decision

Use `protox` (v0.7) as the `FileDescriptorSet` source, consumed by
`prost_build::Config::new().skip_protoc_run().compile_fds(...)`.

## Consequences

**Positive**:
- No system `protoc` install required on any platform, including CI runners
- No C++ toolchain dependency for proto codegen specifically (the C++
  toolchain is still required for other Rust dependencies' build scripts,
  e.g. `generic-array`'s linker requirement, but that's independent of
  this decision)
- Build time for proto codegen is sub-second
- Identical, reproducible behavior across Windows/Linux/macOS contributors

**Negative**:
- `protox` is a smaller, less battle-tested project than canonical
  `protoc`. If a future `.proto` file uses a feature `protox` doesn't
  support, this decision will need revisiting.
- Anyone reading `build.rs` without this ADR might assume `protoc` is
  required and waste time installing it unnecessarily, mitigated by the
  doc comment in `build.rs` linking back to this ADR.

**Neutral**:
- Generated Rust code is byte-for-byte compatible with what `prost-build`
  would produce from canonical `protoc` output, since `prost-build` itself
  is unchanged. Only the `FileDescriptorSet` source differs.

## Revisit Trigger

Revisit if: (a) a future proto file requires a proto3 feature `protox`
doesn't support and codegen fails, or (b) the project adds `tonic` gRPC
service definitions, since `tonic-build`'s integration path with `protox`
should be re-verified rather than assumed to work identically.
