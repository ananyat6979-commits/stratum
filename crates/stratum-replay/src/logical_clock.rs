//! Lamport logical clock for deterministic event ordering in the replay log.
//!
//! # Why Lamport Clocks
//! Wall clock time is not sufficient for causal ordering in a distributed
//! system: clocks on different nodes drift, NTP corrections can cause
//! non-monotonic wall time, and two events on different nodes can have
//! identical wall clock timestamps while having a clear causal ordering.
//!
//! Lamport clocks provide a total causal order:
//! - If event A causally precedes event B, then clock(A) < clock(B)
//! - The converse is not guaranteed: clock(A) < clock(B) does not imply
//!   A causally precedes B (concurrent events may have any clock ordering)
//!
//! For the replay engine's purposes, this is sufficient: we need to ensure
//! that when replaying, a RoutingDecisionEvent is never processed before
//! the RequestIngressEvent it depends on.
//!
//! # Update Rules (Lamport 1978)
//! - On local event: increment clock by 1
//! - On send: include current clock value in the message
//! - On receive: clock = max(local, received) + 1
//!
//! # Tie-breaking
//! When two events have identical Lamport timestamps (concurrent events),
//! they are ordered by (lamport_ts, node_id, event_id). This is a stable
//! total order used by the replay engine's topological sort.
//!
//! Reference: Lamport, L. (1978). "Time, clocks, and the ordering of events
//! in a distributed system." Communications of the ACM, 21(7), 558-565.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A Lamport logical clock for a single node.
///
/// Thread-safe: multiple threads may call `tick()` and `update()`
/// concurrently. The atomicity guarantee is that no two local events
/// receive the same timestamp from the same node.
///
/// `Arc`-wrapped for cheap cloning across async tasks.
#[derive(Debug, Clone)]
pub struct LogicalClock {
    /// The current clock value. Only increases, never decreases.
    value: Arc<AtomicU64>,
    /// The node ID that owns this clock instance.
    /// Used for tie-breaking when two events share a Lamport timestamp.
    pub node_id: Arc<str>,
}

impl LogicalClock {
    /// Create a new logical clock for the given node, initialized to 0.
    ///
    /// # Note on initialization
    /// Per RFC-001 (Open Question 1), the clock is initialized to 0
    /// rather than the wall clock. This means clock values are not
    /// globally unique across node restarts. If a node restarts and
    /// reuses clock values, the replay engine must detect this via the
    /// event log's monotonicity invariant (see `AppendOnlyEventLog`).
    /// This is acceptable for Phase 2; if cross-restart uniqueness is
    /// needed, initialize with wall clock ns (open question deferred).
    pub fn new(node_id: impl Into<Arc<str>>) -> Self {
        Self {
            value: Arc::new(AtomicU64::new(0)),
            node_id: node_id.into(),
        }
    }

    /// Advance the clock for a local event and return the new timestamp.
    ///
    /// Equivalent to Lamport's "local event" rule: increment by 1.
    /// The returned value is the timestamp assigned to this event.
    ///
    /// # Atomicity
    /// The fetch_add is atomic with SeqCst ordering, ensuring that no
    /// two concurrent callers receive the same timestamp on the same node.
    pub fn tick(&self) -> u64 {
        self.value.fetch_add(1, Ordering::SeqCst)
    }

    /// Update the clock after receiving a message with the given timestamp.
    ///
    /// Implements Lamport's "receive" rule: clock = max(local, received) + 1.
    /// Returns the new clock value after the update.
    pub fn update(&self, received_ts: u64) -> u64 {
        loop {
            let current = self.value.load(Ordering::SeqCst);
            let new_value = current.max(received_ts) + 1;
            match self.value.compare_exchange(
                current,
                new_value,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return new_value,
                // Another thread updated the clock concurrently; retry
                Err(_) => continue,
            }
        }
    }

    /// Returns the current clock value without advancing it.
    ///
    /// Use for reading the clock value for diagnostics only, not for
    /// assigning event timestamps. All event timestamps must go through
    /// `tick()` to guarantee uniqueness.
    pub fn current(&self) -> u64 {
        self.value.load(Ordering::SeqCst)
    }
}

/// Ordering key for a single event, used for stable total ordering
/// when multiple events share a Lamport timestamp.
///
/// Ordering: (lamport_ts ASC, node_id ASC, event_id ASC).
/// This is the order the replay engine uses for topological sort
/// tie-breaking. It is stable across replay sessions provided
/// node_id and event_id are deterministic (which they are: node_id
/// is configuration, event_id is UUID v4 committed to the event log).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EventOrderingKey {
    pub lamport_ts: u64,
    pub node_id: String,
    pub event_id: String,
}

impl EventOrderingKey {
    pub fn new(lamport_ts: u64, node_id: impl Into<String>, event_id: impl Into<String>) -> Self {
        Self {
            lamport_ts,
            node_id: node_id.into(),
            event_id: event_id.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn tick_starts_at_zero_and_increments() {
        let clock = LogicalClock::new("node-0");
        assert_eq!(clock.tick(), 0);
        assert_eq!(clock.tick(), 1);
        assert_eq!(clock.tick(), 2);
    }

    #[test]
    fn tick_is_monotonically_increasing() {
        let clock = LogicalClock::new("node-0");
        let mut last = clock.tick();
        for _ in 0..100 {
            let next = clock.tick();
            assert!(next > last, "clock must be strictly increasing: {next} <= {last}");
            last = next;
        }
    }

    #[test]
    fn update_advances_past_received_timestamp() {
        let clock = LogicalClock::new("node-0");
        clock.tick(); // local: 0
        clock.tick(); // local: 1

        // Receive a message with ts=10 (from ahead of us)
        let after_update = clock.update(10);

        // Must be max(2, 10) + 1 = 11
        assert_eq!(after_update, 11);
        // Subsequent ticks must continue from 11
        assert_eq!(clock.tick(), 12);
    }

    #[test]
    fn update_with_older_timestamp_still_increments() {
        let clock = LogicalClock::new("node-0");
        clock.tick(); // 0
        clock.tick(); // 1
        clock.tick(); // 2

        // Receive a message with ts=1 (older than us)
        let after_update = clock.update(1);

        // Must be max(3, 1) + 1 = 4
        assert_eq!(after_update, 4);
    }

    #[test]
    fn clock_never_decreases_under_concurrent_ticks() {
        // This test verifies the atomic ordering guarantee:
        // no two threads receive the same timestamp on the same node.
        let clock = Arc::new(LogicalClock::new("node-0"));
        let n_threads = 8;
        let ticks_per_thread = 1_000;

        let mut handles = vec![];
        for _ in 0..n_threads {
            let c = Arc::clone(&clock);
            handles.push(thread::spawn(move || {
                (0..ticks_per_thread).map(|_| c.tick()).collect::<Vec<_>>()
            }));
        }

        let mut all_timestamps: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();

        all_timestamps.sort_unstable();

        // If the atomic guarantee holds, all timestamps must be unique
        // (no two threads received the same value from fetch_add)
        let original_len = all_timestamps.len();
        all_timestamps.dedup();
        assert_eq!(
            all_timestamps.len(),
            original_len,
            "concurrent ticks produced duplicate timestamps -- atomicity guarantee violated"
        );

        assert_eq!(
            all_timestamps.len(),
            n_threads * ticks_per_thread,
            "total unique timestamps must equal total tick() calls"
        );
    }

    #[test]
    fn event_ordering_key_sorts_by_lamport_then_node_then_event() {
        let mut keys = vec![
            EventOrderingKey::new(5, "node-1", "event-z"),
            EventOrderingKey::new(3, "node-0", "event-a"),
            EventOrderingKey::new(5, "node-0", "event-b"),
            EventOrderingKey::new(3, "node-1", "event-a"),
        ];
        keys.sort();

        assert_eq!(keys[0], EventOrderingKey::new(3, "node-0", "event-a"));
        assert_eq!(keys[1], EventOrderingKey::new(3, "node-1", "event-a"));
        assert_eq!(keys[2], EventOrderingKey::new(5, "node-0", "event-b"));
        assert_eq!(keys[3], EventOrderingKey::new(5, "node-1", "event-z"));
    }

    #[test]
    fn cloned_clock_shares_state() {
        // Arc-cloning the clock must share the underlying counter,
        // not create an independent copy. This is the invariant that
        // allows a single logical clock to be passed to multiple async
        // tasks without losing monotonicity.
        let clock1 = LogicalClock::new("node-0");
        let clock2 = clock1.clone();

        clock1.tick(); // 0
        clock1.tick(); // 1
        let from_clone = clock2.tick(); // must see 2, not 0

        assert_eq!(from_clone, 2, "cloned clock must share the counter");
    }
}