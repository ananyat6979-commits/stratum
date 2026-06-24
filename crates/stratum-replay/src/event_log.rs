//! Append-only event log backed by redb.
//!
//! Every routing decision, request ingress, and inference response is
//! written to this log as a [`ReplayEvent`]. The replay engine reads
//! the log to reconstruct historical routing decisions deterministically.
//!
//! # Storage Design
//! redb was chosen over lmdb for the following reasons:
//! 1. Pure Rust: zero C dependencies, no MSVC linker issues on Windows
//! 2. ACID transactions with ordered key scanning
//! 3. Same semantic model as LMDB (key-value, ordered, append-efficient)
//!
//! lmdb was the original choice (ADR-001 draft) but failed to link on
//! Windows MSVC: lmdb-sys requires advapi32.lib (for
//! InitializeSecurityDescriptor, SetSecurityDescriptorDacl) but does
//! not declare this dependency. redb has identical operational semantics
//! with zero C toolchain exposure. See skills.md.
//!
//! Keys:   (lamport_ts: u64, event_id: u128) -- redb TableDefinition
//!         Ordered by (lamport_ts ASC, event_id ASC) automatically.
//! Values: bincode::serialize(&ReplayEvent) as &[u8]

use std::path::Path;
use std::sync::{Arc, Mutex};

use bincode::{deserialize, serialize};
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};

use crate::logical_clock::{EventOrderingKey, LogicalClock};

/// redb table: (lamport_ts, event_id) -> serialized ReplayEvent bytes
const EVENTS: TableDefinition<(u64, u128), &[u8]> = TableDefinition::new("events");

/// A single event in the replay log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayEvent {
    pub lamport_ts: u64,
    pub event_id: u128,
    pub dependency_ids: Vec<u128>,
    pub emitter_node_id: String,
    /// Serialized event payload. Opaque bytes to this layer.
    pub payload: Vec<u8>,
}

impl ReplayEvent {
    pub fn ordering_key(&self) -> EventOrderingKey {
        EventOrderingKey::new(
            self.lamport_ts,
            &self.emitter_node_id,
            format!("{:032x}", self.event_id),
        )
    }
}

/// Error types for event log operations.
#[derive(Debug)]
pub enum EventLogError {
    Redb(redb::Error),
    RedbDatabase(redb::DatabaseError),
    RedbTransaction(redb::TransactionError),
    RedbTable(redb::TableError),
    RedbCommit(redb::CommitError),
    RedbStorage(redb::StorageError),
    Serialization(String),
    NonMonotonicTimestamp { attempted: u64, last_written: u64 },
}

impl std::fmt::Display for EventLogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Redb(e) => write!(f, "redb error: {e}"),
            Self::RedbDatabase(e) => write!(f, "redb database error: {e}"),
            Self::RedbTransaction(e) => write!(f, "redb transaction error: {e}"),
            Self::RedbTable(e) => write!(f, "redb table error: {e}"),
            Self::RedbCommit(e) => write!(f, "redb commit error: {e}"),
            Self::RedbStorage(e) => write!(f, "redb storage error: {e}"),
            Self::Serialization(s) => write!(f, "serialization error: {s}"),
            Self::NonMonotonicTimestamp {
                attempted,
                last_written,
            } => write!(
                f,
                "non-monotonic timestamp: attempted {attempted}, last written {last_written}"
            ),
        }
    }
}

impl From<redb::Error> for EventLogError {
    fn from(e: redb::Error) -> Self {
        Self::Redb(e)
    }
}
impl From<redb::DatabaseError> for EventLogError {
    fn from(e: redb::DatabaseError) -> Self {
        Self::RedbDatabase(e)
    }
}
impl From<redb::TransactionError> for EventLogError {
    fn from(e: redb::TransactionError) -> Self {
        Self::RedbTransaction(e)
    }
}
impl From<redb::TableError> for EventLogError {
    fn from(e: redb::TableError) -> Self {
        Self::RedbTable(e)
    }
}
impl From<redb::CommitError> for EventLogError {
    fn from(e: redb::CommitError) -> Self {
        Self::RedbCommit(e)
    }
}
impl From<redb::StorageError> for EventLogError {
    fn from(e: redb::StorageError) -> Self {
        Self::RedbStorage(e)
    }
}
impl From<Box<bincode::ErrorKind>> for EventLogError {
    fn from(e: Box<bincode::ErrorKind>) -> Self {
        Self::Serialization(e.to_string())
    }
}

/// Append-only event log backed by redb.
pub struct AppendOnlyEventLog {
    db: Arc<Database>,
    last_written_ts: Arc<Mutex<Option<u64>>>,
    pub clock: LogicalClock,
}

impl AppendOnlyEventLog {
    /// Open (or create) an event log at the given file path.
    ///
    /// Unlike LMDB, redb uses a single file, not a directory.
    /// The parent directory must exist; redb creates the file if absent.
    pub fn open(
        path: impl AsRef<Path>,
        node_id: impl Into<Arc<str>>,
    ) -> Result<Self, EventLogError> {
        let db = Database::create(path.as_ref())?;

        // Ensure the table exists
        let write_txn = db.begin_write()?;
        write_txn.open_table(EVENTS)?;
        write_txn.commit()?;

        Ok(Self {
            db: Arc::new(db),
            last_written_ts: Arc::new(Mutex::new(None)),
            clock: LogicalClock::new(node_id),
        })
    }

    /// Append a single event to the log.
    pub fn append(
        &self,
        event_id: u128,
        dependency_ids: Vec<u128>,
        payload: Vec<u8>,
    ) -> Result<ReplayEvent, EventLogError> {
        let mut last_ts_guard = self.last_written_ts.lock().unwrap();
        let lamport_ts = self.clock.tick();

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

        let value = serialize(&event)?;

        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(EVENTS)?;
            table.insert((lamport_ts, event_id), value.as_slice())?;
        }
        write_txn.commit()?;

        *last_ts_guard = Some(lamport_ts);
        Ok(event)
    }

    /// Load all events in Lamport timestamp order.
    pub fn load_all(&self) -> Result<Vec<ReplayEvent>, EventLogError> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(EVENTS)?;

        let mut events = Vec::new();
        for result in table.iter()? {
            let (_key, value) = result?;
            let event: ReplayEvent = deserialize(value.value())?;
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
        Ok(self
            .load_all()?
            .into_iter()
            .filter(|e| e.lamport_ts >= start_ts && e.lamport_ts <= end_ts)
            .collect())
    }

    /// Return the total number of events in the log.
    pub fn len(&self) -> Result<usize, EventLogError> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(EVENTS)?;
        Ok(table.len()? as usize)
    }

    pub fn is_empty(&self) -> Result<bool, EventLogError> {
        Ok(self.len()? == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_log_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("stratum-test-{}.redb", uuid::Uuid::new_v4()))
    }

    #[test]
    fn appended_events_are_retrievable_in_order() {
        let log = AppendOnlyEventLog::open(temp_log_path(), "node-0").unwrap();
        let a = log.append(1, vec![], b"payload-a".to_vec()).unwrap();
        let b = log.append(2, vec![1], b"payload-b".to_vec()).unwrap();
        let c = log.append(3, vec![2], b"payload-c".to_vec()).unwrap();

        let events = log.load_all().unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event_id, a.event_id);
        assert_eq!(events[1].event_id, b.event_id);
        assert_eq!(events[2].event_id, c.event_id);
    }

    #[test]
    fn payload_is_preserved_exactly() {
        let log = AppendOnlyEventLog::open(temp_log_path(), "node-0").unwrap();
        let payload = b"exact-payload-bytes-must-survive-round-trip".to_vec();
        log.append(1, vec![], payload.clone()).unwrap();
        let events = log.load_all().unwrap();
        assert_eq!(events[0].payload, payload);
    }

    #[test]
    fn dependency_ids_are_preserved() {
        let log = AppendOnlyEventLog::open(temp_log_path(), "node-0").unwrap();
        let deps = vec![42u128, 99u128, 777u128];
        log.append(1, deps.clone(), b"payload".to_vec()).unwrap();
        let events = log.load_all().unwrap();
        assert_eq!(events[0].dependency_ids, deps);
    }

    #[test]
    fn lamport_timestamps_are_strictly_increasing() {
        let log = AppendOnlyEventLog::open(temp_log_path(), "node-0").unwrap();
        for i in 0..10u128 {
            log.append(i, vec![], b"payload".to_vec()).unwrap();
        }
        let events = log.load_all().unwrap();
        let timestamps: Vec<u64> = events.iter().map(|e| e.lamport_ts).collect();
        for window in timestamps.windows(2) {
            assert!(window[1] > window[0]);
        }
    }

    #[test]
    fn len_returns_correct_count() {
        let log = AppendOnlyEventLog::open(temp_log_path(), "node-0").unwrap();
        assert_eq!(log.len().unwrap(), 0);
        log.append(1, vec![], b"a".to_vec()).unwrap();
        assert_eq!(log.len().unwrap(), 1);
        log.append(2, vec![], b"b".to_vec()).unwrap();
        assert_eq!(log.len().unwrap(), 2);
    }

    #[test]
    fn load_range_filters_correctly() {
        let log = AppendOnlyEventLog::open(temp_log_path(), "node-0").unwrap();
        for i in 0..10u128 {
            log.append(i, vec![], format!("p{i}").into_bytes()).unwrap();
        }
        let all = log.load_all().unwrap();
        let ts_4 = all[4].lamport_ts;
        let ts_6 = all[6].lamport_ts;
        let range = log.load_range(ts_4, ts_6).unwrap();
        assert_eq!(range.len(), 3);
        assert_eq!(range[0].lamport_ts, ts_4);
        assert_eq!(range[2].lamport_ts, ts_6);
    }

    #[test]
    fn emitter_node_id_is_recorded() {
        let log = AppendOnlyEventLog::open(temp_log_path(), "test-node-42").unwrap();
        log.append(1, vec![], b"payload".to_vec()).unwrap();
        let events = log.load_all().unwrap();
        assert_eq!(events[0].emitter_node_id, "test-node-42");
    }
}
