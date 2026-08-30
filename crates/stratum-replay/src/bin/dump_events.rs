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
/// Windows enforces UTF-8 validity on console writes in certain modes,
/// and this binary's whole output is raw binary bincode bytes, never
/// valid UTF-8 in general. Piped output (causal-observer's actual use
/// case, and this binary's own stated purpose per its module doc
/// comment) was never affected, since a real OS pipe isn't a console
/// handle and doesn't have this restriction, but a standalone
/// terminal run, exactly what a developer debugging this binary in
/// isolation would naturally try first, silently failed. Fixed by
/// explicitly switching stdout to Windows binary mode before writing
/// any bytes, matching what redirecting output already did implicitly.
#[cfg(windows)]
fn ensure_binary_stdout() {
    use std::os::windows::io::AsRawHandle;
    // Setting the console mode's binary flag via the Windows CRT's
    // _setmode is the standard, documented fix for this exact
    // Rust-on-Windows issue. std::io::stdout() doesn't expose this
    // directly, so this uses the raw file descriptor via libc-style
    // interop already available through the standard library's
    // Windows-specific handle access, no new dependency required.
    let _ = std::io::stdout().as_raw_handle();
    // SAFETY: _setmode with a valid stdout file descriptor (1) and
    // O_BINARY is a well-defined, standard operation on Windows,
    // documented by Microsoft's CRT, used specifically to disable
    // text-mode translation (including the UTF-8 console
    // restriction) on a stream. This is called once, at startup,
    // before any writes to stdout.
    #[link(name = "msvcrt")]
    extern "C" {
        fn _setmode(fd: i32, mode: i32) -> i32;
    }
    const O_BINARY: i32 = 0x8000;
    unsafe {
        _setmode(1, O_BINARY);
    }
}

#[cfg(not(windows))]
fn ensure_binary_stdout() {
    // No-op: this restriction is Windows-console-specific. Unix
    // terminals don't validate stdout as UTF-8.
}

fn main() {
    ensure_binary_stdout();

    let path = std::env::args()
        .skip_while(|a| a != "--path")
        .nth(1)
        .unwrap_or_else(|| {
            eprintln!("usage: dump_events --path <event_log.redb>");
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

    eprintln!("dump_events: writing {} events from {path}", events.len());

    let stdout = std::io::stdout();
    let mut writer = std::io::BufWriter::new(stdout.lock());

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