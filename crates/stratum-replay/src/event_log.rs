//! Append-only event log backed by LMDB.
//!
//! Every routing decision, request ingress, and inference response is
//! written to this log as a [`ReplayEvent`]. The replay engine reads
//! the log to reconstruct historical routing decisions deterministically.
//!
//! # Storage Design
//! LMDB was chosen over SQLite and RocksDB for the following reasons:
//! 1. Memory-mapped I/O: reads are near-zero overhead for cached pages
//! 2. ACID transactions without WAL overhead for append-only workloads
//! 3. Key ordering: LMDB B-tree orders by key bytes, so big-endian
//!    Lamport timestamps produce correct chronological scan order
//!
//! Keys:   lamport_ts.to_be_bytes() ++ event_id (16 bytes total)
//!         The event_id suffix breaks ties when two events share a
//!         Lamport timestamp (concurrent events on the same node).
//! Values: bincode::serialize(&ReplayEvent)
//!
//! # Monotonicity Invariant
//! The log enforces that Lamport timestamps are strictly increasing
//! within a single node session. A write with a timestamp less than
//! or equal to the last written timestamp is rejected with an error.
//! This guards against clock bugs and out-of-order writes.
//!
//! # Why bincode not proto
//! The event log is a Rust-internal artifact. Proto is correct for
//! the wire format between services; bincode is correct for internal
//! persistence where language interoperability is not required and
//! serialization speed matters. If a non-Rust tool needs to read the
//! log, add a proto export layer rather than switching the primary format.
//! Documented in ADR-001 when written.

use std::path::Path;
use std::sync::{Arc, Mutex};

use bincode::{deserialize, serialize};
use lmdb::{Cursor, Database, Environment, Transaction, WriteFlags};
use serde::{Deserialize, Serialize};

use crate::logical_clock::{EventOrderingKey, LogicalClock};

/// A single event in the replay log.
///
/// The `payload` field is intentionally untyped (`Vec<u8>`) at this layer —
/// the event log is a generic append-only store. Callers serialize their
/// specific event types (e.g., the causal.proto messages) before passing
/// them here. This keeps the event log independent of the proto schema,
/// which is important because the proto schema may evolve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayEvent {
    /// Lamport timestamp assigned by the node's LogicalClock.
    pub lamport_ts: u64,

    /// Unique event identifier (UUID v4, as a 128-bit integer).
    /// Used for tie-breaking and dependency graph construction.
    pub event_id: u128,

    /// IDs of events this event causally depends on.
    /// Empty for root events (e.g., RequestIngressEvent).
    pub dependency_ids: Vec<u128>,

    /// The node that emitted this event.
    pub emitter_node_id: String,

    /// Serialized event payload (bincode of the specific event type).
    /// The event log does not interpret this — it is opaque bytes.
    pub payload: Vec<u8>,
}

impl ReplayEvent {
    /// Construct the LMDB key for this event.
    ///
    /// Key format: [lamport_ts as 8 big-endian bytes] ++ [event_id as 16 big-endian bytes]
    ///
    /// Big-endian ordering ensures LMDB's byte-comparison key ordering
    /// produces correct chronological ordering of events.
    ///
    /// The event_id suffix ensures uniqueness when two events share
    /// a Lamport timestamp (concurrent events).
    pub fn lmdb_key(&self) -> [u8; 24] {
        let mut key = [0u8; 24];
        key[..8].copy_from_slice(&self.lamport_ts.to_be_bytes());
        key[8..].copy_from_slice(&self.event_id.to_be_bytes());
        key
    }

    /// Return the ordering key for topological sort tie-breaking.
    pub fn ordering_key(&self, node_id: &str) -> EventOrderingKey {
        EventOrderingKey::new(self.lamport_ts, node_id, format!("{:032x}", self.event_id))
    }
}

/// Error types for event log operations.
#[derive(Debug)]
pub enum EventLogError {
    /// LMDB returned an error.
    Lmdb(lmdb::Error),
    /// Serialization/deserialization failed.
    Serialization(String),
    /// Attempted to write an event with a non-monotonic timestamp.
    /// This indicates a bug in the caller's clock management.
    NonMonotonicTimestamp { attempted: u64, last_written: u64 },
    /// The log is empty — no events have been written.
    Empty,
}

impl std::fmt::Display for EventLogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lmdb(e) => write!(f, "LMDB error: {e}"),
            Self::Serialization(s) => write!(f, "serialization error: {s}"),
            Self::NonMonotonicTimestamp {
                attempted,
                last_written,
            } => write!(
                f,
                "non-monotonic timestamp: attempted {attempted}, last written {last_written}"
            ),
            Self::Empty => write!(f, "event log is empty"),
        }
    }
}

impl From<lmdb::Error> for EventLogError {
    fn from(e: lmdb::Error) -> Self {
        Self::Lmdb(e)
    }
}

/// Append-only event log backed by LMDB.
///
/// Thread-safe: internal state is protected by a `Mutex`. LMDB itself
/// is not thread-safe for concurrent writes; the mutex ensures only one
/// write transaction is active at a time. Reads bypass the mutex (LMDB
/// supports concurrent readers) but the current implementation does not
/// expose a read-only transaction API separately.
pub struct AppendOnlyEventLog {
    env: Arc<Environment>,
    db: Database,
    /// The Lamport timestamp of the last successfully written event.
    /// Protected by the same mutex as write operations to prevent
    /// concurrent writes from violating the monotonicity invariant.
    last_written_ts: Arc<Mutex<Option<u64>>>,
    pub clock: LogicalClock,
}

impl AppendOnlyEventLog {
    /// Open (or create) an event log at the given path.
    ///
    /// `path` must be a directory. LMDB creates its data files (`data.mdb`,
    /// `lock.mdb`) inside this directory. The directory must exist.
    pub fn open(
        path: impl AsRef<Path>,
        node_id: impl Into<Arc<str>>,
    ) -> Result<Self, EventLogError> {
        let node_id: Arc<str> = node_id.into();

        let env = lmdb::Environment::new()
            .set_max_dbs(1)
            .set_map_size(1024 * 1024 * 1024) // 1GB max map size
            .open(path.as_ref())?;

        let db = env.open_db(Some("events"))?;

        Ok(Self {
            env: Arc::new(env),
            db,
            last_written_ts: Arc::new(Mutex::new(None)),
            clock: LogicalClock::new(node_id),
        })
    }

    /// Append a single event to the log.
    ///
    /// The event's `lamport_ts` is assigned by this function using the
    /// log's internal `LogicalClock`. Callers do not set timestamps —
    /// the log owns the clock for its node.
    ///
    /// # Monotonicity
    /// Returns `EventLogError::NonMonotonicTimestamp` if the assigned
    /// timestamp would violate strict monotonicity. This should never
    /// happen in correct usage (the LogicalClock is strictly increasing)
    /// but is checked defensively.
    ///
    /// # Arguments
    /// * `event_id` — unique identifier for this event (UUID as u128)
    /// * `dependency_ids` — causal dependencies of this event
    /// * `payload` — serialized event content (opaque bytes to this layer)
    pub fn append(
        &self,
        event_id: u128,
        dependency_ids: Vec<u128>,
        payload: Vec<u8>,
    ) -> Result<ReplayEvent, EventLogError> {
        let mut last_ts_guard = self.last_written_ts.lock().unwrap();

        let lamport_ts = self.clock.tick();

        // Monotonicity check
        if let Some(last) = *last_ts_guard {
            if lamport_ts <= last {
                return Err(EventLogError::NonMonotonicTimestamp {
                    attempted: lamport_ts,
                    last_written: last,
                });
            }
        }

        let event = ReplayEvent {
            lamport_ts,
            event_id,
            dependency_ids,
            emitter_node_id: self.clock.node_id.to_string(),
            payload,
        };

        let key = event.lmdb_key();
        let value = serialize(&event).map_err(|e| EventLogError::Serialization(e.to_string()))?;

        let mut txn = self.env.begin_rw_txn()?;
        txn.put(self.db, &key, &value, WriteFlags::NO_OVERWRITE)?;
        txn.commit()?;

        *last_ts_guard = Some(lamport_ts);

        Ok(event)
    }

    /// Load all events in the log, in Lamport timestamp order.
    ///
    /// Returns events sorted by (lamport_ts, event_id) — the same
    /// order as the LMDB key, which is chronological by construction.
    pub fn load_all(&self) -> Result<Vec<ReplayEvent>, EventLogError> {
        let txn = self.env.begin_ro_txn()?;
        let mut cursor = txn.open_ro_cursor(self.db)?;

        let mut events = Vec::new();
        for (_key, value) in cursor.iter() {
            let event: ReplayEvent = deserialize(value)
                .map_err(|e| EventLogError::Serialization(e.to_string()))?;
            events.push(event);
        }

        Ok(events)
    }

    /// Load events in the Lamport timestamp range [start_ts, end_ts].
    pub fn load_range(
        &self,
        start_ts: u64,
        end_ts: u64,
    ) -> Result<Vec<ReplayEvent>, EventLogError> {
        let all = self.load_all()?;
        Ok(all
            .into_iter()
            .filter(|e| e.lamport_ts >= start_ts && e.lamport_ts <= end_ts)
            .collect())
    }

    /// Return the total number of events in the log.
    pub fn len(&self) -> Result<usize, EventLogError> {
        let txn = self.env.begin_ro_txn()?;
        let mut cursor = txn.open_ro_cursor(self.db)?;
        let count = cursor.iter().count();
        Ok(count)
    }

    /// Return true if the log has no events.
    pub fn is_empty(&self) -> Result<bool, EventLogError> {
        Ok(self.len()? == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_log_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("stratum-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn appended_events_are_retrievable_in_order() {
        let dir = temp_log_dir();
        let log = AppendOnlyEventLog::open(&dir, "node-0").unwrap();

        let event_a = log.append(1, vec![], b"payload-a".to_vec()).unwrap();
        let event_b = log.append(2, vec![1], b"payload-b".to_vec()).unwrap();
        let event_c = log.append(3, vec![2], b"payload-c".to_vec()).unwrap();

        let events = log.load_all().unwrap();
        assert_eq!(events.len(), 3);

        // Events must be returned in Lamport order
        assert_eq!(events[0].event_id, event_a.event_id);
        assert_eq!(events[1].event_id, event_b.event_id);
        assert_eq!(events[2].event_id, event_c.event_id);
    }

    #[test]
    fn payload_is_preserved_exactly() {
        let dir = temp_log_dir();
        let log = AppendOnlyEventLog::open(&dir, "node-0").unwrap();

        let payload = b"exact-payload-bytes-must-survive-round-trip".to_vec();
        log.append(1, vec![], payload.clone()).unwrap();

        let events = log.load_all().unwrap();
        assert_eq!(events[0].payload, payload);
    }

    #[test]
    fn dependency_ids_are_preserved() {
        let dir = temp_log_dir();
        let log = AppendOnlyEventLog::open(&dir, "node-0").unwrap();

        let deps = vec![42u128, 99u128, 777u128];
        log.append(1, deps.clone(), b"payload".to_vec()).unwrap();

        let events = log.load_all().unwrap();
        assert_eq!(events[0].dependency_ids, deps);
    }

    #[test]
    fn lamport_timestamps_are_strictly_increasing() {
        let dir = temp_log_dir();
        let log = AppendOnlyEventLog::open(&dir, "node-0").unwrap();

        for i in 0..10u128 {
            log.append(i, vec![], b"payload".to_vec()).unwrap();
        }

        let events = log.load_all().unwrap();
        let timestamps: Vec<u64> = events.iter().map(|e| e.lamport_ts).collect();

        for window in timestamps.windows(2) {
            assert!(
                window[1] > window[0],
                "timestamps must be strictly increasing: {:?}",
                timestamps
            );
        }
    }

    #[test]
    fn len_returns_correct_count() {
        let dir = temp_log_dir();
        let log = AppendOnlyEventLog::open(&dir, "node-0").unwrap();

        assert_eq!(log.len().unwrap(), 0);
        log.append(1, vec![], b"a".to_vec()).unwrap();
        assert_eq!(log.len().unwrap(), 1);
        log.append(2, vec![], b"b".to_vec()).unwrap();
        assert_eq!(log.len().unwrap(), 2);
    }

    #[test]
    fn load_range_filters_correctly() {
        let dir = temp_log_dir();
        let log = AppendOnlyEventLog::open(&dir, "node-0").unwrap();

        for i in 0..10u128 {
            log.append(i, vec![], format!("payload-{i}").into_bytes())
                .unwrap();
        }

        let all = log.load_all().unwrap();
        let ts_4 = all[4].lamport_ts;
        let ts_6 = all[6].lamport_ts;

        let range = log.load_range(ts_4, ts_6).unwrap();
        assert_eq!(range.len(), 3, "range [4..=6] should include 3 events");
        assert_eq!(range[0].lamport_ts, ts_4);
        assert_eq!(range[2].lamport_ts, ts_6);
    }

    #[test]
    fn emitter_node_id_is_recorded() {
        let dir = temp_log_dir();
        let log = AppendOnlyEventLog::open(&dir, "test-node-42").unwrap();

        log.append(1, vec![], b"payload".to_vec()).unwrap();

        let events = log.load_all().unwrap();
        assert_eq!(events[0].emitter_node_id, "test-node-42");
    }
}
