//! Build script: compiles protobuf schemas into Rust types via prost.
//!
//! Generated code is written to `$OUT_DIR/stratum.v1.rs` and included
//! into `proto.rs` via `include!(concat!(env!("OUT_DIR"), "/stratum.v1.rs"))`.
//!
//! # Why protox, not protoc
//! `prost-build` normally shells out to a system `protoc` binary.
//! `protobuf-src` (compiling protoc from C++ source via CMake) was tried
//! and abandoned: building abseil-cpp from source under MSVC failed and
//! is a multi-minute build even when it succeeds -- too fragile for a
//! file this small.
//!
//! `protox` is a pure-Rust protobuf parser. It produces a
//! `FileDescriptorSet` directly, which `prost_build::Config` accepts via
//! `skip_protoc_run()` + `compile_fds()`. No C++ toolchain, no protoc
//! binary, no cmake -- works identically on every platform.
//!
//! Only `inference.proto` is compiled for now -- the other seven proto
//! files are syntax-only stubs (Phase 2+).

fn main() {
    let proto_root = "../../protos/stratum/v1";
    let proto_file = format!("{proto_root}/inference.proto");

    println!("cargo:rerun-if-changed={proto_file}");

    let file_descriptor_set = protox::compile([&proto_file], [proto_root])
        .expect("protox failed to parse inference.proto -- check proto syntax");

    prost_build::Config::new()
        .skip_protoc_run()
        .compile_fds(file_descriptor_set)
        .expect("prost-build failed to generate Rust types from descriptor set");
}
