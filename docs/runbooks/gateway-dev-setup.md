# Runbook: stratum-gateway Development Environment Setup (Windows)

**Audience**: Anyone setting up this repo on a fresh Windows machine.
**Last verified**: 2026-06-17

## Required Toolchain

| Tool | Install Command | Verify |
|------|------------------|--------|
| Rust (rustup) | `winget install Rustlang.Rustup` | `rustc --version` |
| MSVC C++ Build Tools | See "C++ Toolchain" below: **not** winget | `where.exe link.exe` |
| uv (Python) | `winget install astral-sh.uv` | `uv --version` |
| Go | `winget install GoLang.Go` | `go version` |
| buf (proto lint) | `winget install bufbuild.buf` | `buf --version` |

## PATH Refresh: Required After Every winget Install

Windows PowerShell does not pick up PATH changes from `winget install` in
the current session. After **every** install above, either open a fresh
terminal, or run:

```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

To avoid repeating this manually, add the line above to your PowerShell
profile (`notepad $PROFILE`) so every new terminal has it automatically.

## C++ Toolchain (MSVC linker)

`generic-array` (a transitive dependency of `sha2`, used by
`stratum-gateway::signing`) requires `link.exe` to build. The base
Visual Studio Build Tools installer does **not** include this by default.

1. Run the Build Tools installer (or open Visual Studio Installer if
   already installed)
2. Click **Modify** (not Repair. Repair does not add missing workloads)
3. Check **"Desktop development with C++"**
4. Click Modify/Install, wait ~10-20 minutes
5. Open a fresh terminal, refresh PATH (see above)
6. Verify: `where.exe link.exe` should print a path under
   `Program Files (x86)\Microsoft Visual Studio\...`

**Symptom if this step is skipped**: `cargo build`/`cargo test` fails
with `error: linker \`link.exe\` not found` while compiling
`generic-array`'s build script.

## Protobuf Codegen: Do NOT Install protoc

`stratum-gateway`'s `build.rs` uses `protox` (pure-Rust proto parser), not
a system `protoc` binary. **Do not** attempt to install `protoc` via
winget — `winget install protocolbuffers.protoc` does not resolve to a
valid package on this platform, and even if it did, it is unnecessary.

**Do not** add `protobuf-src` as a build-dependency either. It was tried
and abandoned (see ADR-001) — it compiles `protoc` from C++ source via
CMake and abseil-cpp, which failed under this project's MSVC environment
and is a multi-minute build even when it succeeds. `protox` requires
nothing beyond what `cargo build` already pulls in.

If you see a build script panic referencing `cmake`, `abseil-cpp`, or
`cordz`/`hashtablez`/`flags` internals, something has regressed back to
attempting a `protobuf-src`-style build — check `Cargo.toml`
`[build-dependencies]` for `protobuf-src` and remove it; `protox` should
be the only proto-related build dependency.

## First Build

```powershell
cd crates
cargo check        # workspace-wide sanity check
cargo test -p stratum-gateway
cd ..
```

Expected first-build time: 30-60 seconds (downloading + compiling
dependencies). Subsequent builds: a few seconds.

## Common Failure: Empty/Missing Generated Files After Editing in `code`

If `cargo build` reports a source file "cannot be read" / "system cannot
find the file specified" immediately after creating it via an editor
command, verify the file actually has content on disk before assuming
the build is broken:

```powershell
Get-Item crates\stratum-gateway\src\<file>.rs
```

If `Length` is 0 or the file doesn't exist, re-create it via
`Set-Content` (see examples throughout commit history) rather than
relying on an editor having saved it.

## buf lint

All `.proto` files in a package must declare an **identical**
`go_package` value. If you add a new proto file, copy the existing
`option go_package = "github.com/stratum-project/stratum/gen/go/stratum/v1";`
line exactly, a mismatch here is the most common `buf lint` failure
encountered so far.

```powershell
cd protos
buf lint
cd ..
```

Expected output: silence (zero errors).

## Related ADRs

- [ADR-001](../adr/ADR-001-protox-over-protobuf-src.md) — why protox, not protoc/protobuf-src
