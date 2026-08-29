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

fn main() {
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