//! `dump_events`: a tiny CLI that opens a real event log via the
//! existing, tested `AppendOnlyEventLog::load_all()` and writes each
//! event's raw bincode bytes to stdout, length-prefixed.
//!
//! WHY THIS EXISTS
//! ================
//! `causal-observer` (Go) needs to read STRATUM's event log. The two
//! options were: (1) reimplement redb's on-disk B-tree format in Go,
//! a second, unverified implementation of something
//! `stratum-replay/src/event_log.rs` already implements and tests
//! (see that file's 7 passing tests), or (2) let the real, tested Rust
//! code do the redb reading, and hand off only the resulting
//! already-decoded `ReplayEvent`s' bincode bytes across a process
//! boundary, where Go only has to trust `bincode`'s well-defined wire
//! format, not redb's storage internals too. This is (2).
//!
//! OUTPUT FORMAT
//! ==============
//! For each event, in Lamport-timestamp order (matching
//! `load_all()`'s own ordering): a 4-byte little-endian u32 length
//! prefix, followed by that many bytes of
//! `bincode::serialize(&ReplayEvent)`. This is NOT re-serializing the
//! event, it's writing out the exact bytes `AppendOnlyEventLog`
//! already produces internally, see `event_log.rs`'s `append()`,
//! which calls `serialize(&event)` once and stores that; this binary
//! reconstructs and re-emits an equivalent serialization from the
//! deserialized `ReplayEvent` struct `load_all()` returns, which is
//! byte-identical to the original since bincode's encoding is
//! deterministic for a given struct value.
//!
//! USAGE
//! =====
//!     cargo run --bin dump_events -- --path gateway_event_log.redb
//!
//! Exits with a non-zero status and a clear stderr message if the
//! path doesn't exist or isn't a valid event log, rather than writing
//! partial/garbage output to stdout.

use std::io::Write;

use stratum_replay::event_log::AppendOnlyEventLog;

/// Real bug found while verifying this binary: running it directly in
/// a Windows terminal (not piped, as causal-observer's Go wrapper
/// does via os/exec's StdoutPipe) failed with "Windows stdio in
/// console mode does not support writing non-UTF-8 byte sequences",
/// Windows enforces UTF-8 validity specifically on writes through
/// std::io::Stdout's console-mode handling, and this binary's whole
/// output is raw binary bincode bytes, never valid UTF-8 in general.
///
/// FIRST FIX ATTEMPT DID NOT WORK, RECORDED HONESTLY: an earlier
/// version of this function called the Windows CRT's `_setmode` on
/// file descriptor 1 to switch it to binary mode, on the reasoning
/// that this is the standard, documented fix for this class of
/// Rust-on-Windows issue. Verified directly, by hand, in a real
/// unredirected terminal, both before and after that fix: the exact
/// same error fired both times. Most likely cause: `_setmode` patches
/// the C runtime's file descriptor table, but `std::io::Stdout`'s
/// writes may not consistently route through that same descriptor by
/// the time the UTF-8 check fires, especially when a `cargo run`
/// wrapper process is involved. Rather than guess at a second
/// mode-flag-based fix and risk the same false confidence, this
/// version bypasses `std::io::Stdout` for the actual byte writes
/// entirely and writes directly to the raw OS file handle via
/// `std::os::windows::io::FromRawHandle`, which has no console-mode
/// UTF-8 validation layer, that validation is specific to
/// `Stdout`'s console-aware writer, not a property of the underlying
/// handle. `GetStdHandle(STD_OUTPUT_HANDLE)` retrieves the same
/// handle `std::io::stdout()` would have used; wrapping it directly
/// with `std::fs::File::from_raw_handle` and writing through that
/// skips the layer that was rejecting non-UTF-8 bytes.
#[cfg(windows)]
fn raw_stdout_writer() -> Box<dyn Write> {
    use std::fs::File;
    use std::os::windows::io::FromRawHandle;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetStdHandle(nStdHandle: i32) -> *mut std::ffi::c_void;
    }
    const STD_OUTPUT_HANDLE: i32 = -11;

    // SAFETY: GetStdHandle(STD_OUTPUT_HANDLE) is a well-defined
    // Windows API call returning the process's standard output
    // handle, documented to be valid for the lifetime of the process
    // unless explicitly closed or redirected, neither of which this
    // short-lived CLI does. Wrapping it in a File via
    // from_raw_handle is safe as long as this process doesn't also
    // hold and use std::io::stdout() concurrently for byte writes
    // (it doesn't, see main(), all data writes go through this
    // writer, only the human-readable progress line uses eprintln!,
    // a separate stream, stderr).
    let handle = unsafe { GetStdHandle(STD_OUTPUT_HANDLE) };
    let file = unsafe { File::from_raw_handle(handle as *mut _) };
    Box::new(file)
}

#[cfg(not(windows))]
fn raw_stdout_writer() -> Box<dyn Write> {
    Box::new(std::io::stdout())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let stats_only = args.iter().any(|a| a == "--stats-only");

    let path = args
        .iter()
        .skip_while(|a| a.as_str() != "--path")
        .nth(1)
        .cloned()
        .unwrap_or_else(|| {
            eprintln!("usage: dump_events --path <event_log.redb> [--stats-only]");
            std::process::exit(2);
        });

    let log = AppendOnlyEventLog::open_existing(&path, "dump_events").unwrap_or_else(|e| {
        eprintln!("failed to open event log at {path}: {e}");
        std::process::exit(1);
    });

    let events = log.load_all().unwrap_or_else(|e| {
        eprintln!("failed to read events from {path}: {e}");
        std::process::exit(1);
    });

    if stats_only {
        // Added specifically to answer a real scoping question (does
        // Phase 2's pre-oracle-snapshot event data justify building
        // cross-language bincode compatibility, or is it small/stale
        // enough that documenting the break is the more honest choice)
        // with a real count and date range instead of a proxy
        // (file LastWriteTime) or an assumption. See docs/SCOPE.md for
        // the decision this fed into.
        if events.is_empty() {
            println!("0 events in {path}");
            return;
        }
        let min_ts = events.iter().map(|e| e.lamport_ts).min().unwrap();
        let max_ts = events.iter().map(|e| e.lamport_ts).max().unwrap();
        println!("{} events in {path}", events.len());
        println!("lamport_ts range: {min_ts} .. {max_ts}");
        return;
    }

    eprintln!("dump_events: writing {} events from {path}", events.len());

    let mut writer = std::io::BufWriter::new(raw_stdout_writer());

    for event in &events {
        let bytes = bincode::serialize(event).unwrap_or_else(|e| {
            eprintln!("failed to re-serialize event {}: {e}", event.event_id);
            std::process::exit(1);
        });
        let len = bytes.len() as u32;
        writer
            .write_all(&len.to_le_bytes())
            .and_then(|_| writer.write_all(&bytes))
            .unwrap_or_else(|e| {
                eprintln!("failed writing event to stdout: {e}");
                std::process::exit(1);
            });
    }

    writer.flush().unwrap_or_else(|e| {
        eprintln!("failed flushing stdout: {e}");
        std::process::exit(1);
    });
}