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
//! Keys:   (lamport_ts: u64, event_id: u128), redb TableDefinition
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
    Redb(Box<redb::Error>),
    RedbDatabase(Box<redb::DatabaseError>),
    RedbTransaction(Box<redb::TransactionError>),
    RedbTable(Box<redb::TableError>),
    RedbCommit(Box<redb::CommitError>),
    RedbStorage(Box<redb::StorageError>),
    Serialization(String),
    NonMonotonicTimestamp { attempted: u64, last_written: u64 },
    LogNotFound(String),
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
            Self::LogNotFound(path) => write!(
                f,
                "event log not found at {path}, refusing to silently create a new, \
                 empty log; use AppendOnlyEventLog::open() instead if creating a new \
                 log was actually intended"
            ),
        }
    }
}

impl From<redb::Error> for EventLogError {
    fn from(e: redb::Error) -> Self {
        Self::Redb(Box::new(e))
    }
}
impl From<redb::DatabaseError> for EventLogError {
    fn from(e: redb::DatabaseError) -> Self {
        Self::RedbDatabase(Box::new(e))
    }
}
impl From<redb::TransactionError> for EventLogError {
    fn from(e: redb::TransactionError) -> Self {
        Self::RedbTransaction(Box::new(e))
    }
}
impl From<redb::TableError> for EventLogError {
    fn from(e: redb::TableError) -> Self {
        Self::RedbTable(Box::new(e))
    }
}
impl From<redb::CommitError> for EventLogError {
    fn from(e: redb::CommitError) -> Self {
        Self::RedbCommit(Box::new(e))
    }
}
impl From<redb::StorageError> for EventLogError {
    fn from(e: redb::StorageError) -> Self {
        Self::RedbStorage(Box::new(e))
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
    ///
    /// # A real bug this fixed
    /// Previously called `Database::create()` unconditionally, which
    /// silently creates a fresh, empty database at `path` if nothing
    /// exists there yet, redb's own documented behavior for
    /// `create()`. This meant a caller who intended to open an
    /// EXISTING log (e.g. `dump_events`, reading a real event log for
    /// analysis) but passed a slightly wrong relative path got no
    /// error, just a silently empty result indistinguishable from "this
    /// log genuinely has zero events." Confirmed as a real, live bug:
    /// `dump_events path ..\..\benchmarks\harness\gw_sem_phase2full.redb`
    /// from `crates/stratum-replay`, run against a real, populated
    /// event log from an actual 2000-observation benchmark run,
    /// silently created a brand-new empty file at that path (both
    /// LastWriteTime and a suspiciously round, identical byte count
    /// across two unrelated invocations confirmed this directly) and
    /// reported "0 events" with no error, rather than failing loudly
    /// on the real, underlying path mismatch. This constructor now
    /// requires callers to be explicit about which behavior they want.
    pub fn open(
        path: impl AsRef<Path>,
        node_id: impl Into<Arc<str>>,
    ) -> Result<Self, EventLogError> {
        Self::open_impl(path, node_id, /* create_if_missing */ true)
    }

    /// Same as [`open`], but returns [`EventLogError::LogNotFound`]
    /// instead of silently creating a new, empty log if `path` doesn't
    /// already exist. Use this whenever the caller's intent is to read
    /// or analyze an EXISTING log, not to start a new one, exactly
    /// the bug `open`'s doc comment above describes.
    pub fn open_existing(
        path: impl AsRef<Path>,
        node_id: impl Into<Arc<str>>,
    ) -> Result<Self, EventLogError> {
        Self::open_impl(path, node_id, /* create_if_missing */ false)
    }

    fn open_impl(
        path: impl AsRef<Path>,
        node_id: impl Into<Arc<str>>,
        create_if_missing: bool,
    ) -> Result<Self, EventLogError> {
        if !create_if_missing && !path.as_ref().exists() {
            return Err(EventLogError::LogNotFound(
                path.as_ref().display().to_string(),
            ));
        }

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

    /// Load events in the Lamport timestamp range [start_ts, end_ts]
    /// (inclusive on both ends).
    ///
    /// # A real bug this fixed
    /// Previously called `load_all()` (deserializing every event in
    /// the entire log) and filtered the resulting Vec in memory. This
    /// is O(n) in the TOTAL size of the log on every call, regardless
    /// of how narrow the requested range is, even though the table's
    /// key, `(u64, u128)` with `lamport_ts` first (see this module's
    /// doc comment: "Ordered by (lamport_ts ASC, event_id ASC)
    /// automatically"), was deliberately chosen to make a real,
    /// bounded range scan possible via redb's own `Table::range`. The
    /// schema was built for the fast path and this function simply
    /// wasn't using it. For a replay session pulling a narrow window
    /// out of a long-lived, large event log, this was the difference
    /// between touching a handful of keys and deserializing the whole
    /// log on every call.
    ///
    /// `event_id` is `u128`, so the range bound on the key's second
    /// component spans its full domain (`u128::MIN..=u128::MAX`): a
    /// range query bounded only on `lamport_ts` must not accidentally
    /// exclude a real event at `start_ts` or `end_ts` whose
    /// `event_id` happens to sort outside some arbitrarily chosen
    /// narrower bound.
    pub fn load_range(
        &self,
        start_ts: u64,
        end_ts: u64,
    ) -> Result<Vec<ReplayEvent>, EventLogError> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(EVENTS)?;

        let range_start = (start_ts, u128::MIN);
        let range_end = (end_ts, u128::MAX);

        let mut events = Vec::new();
        for result in table.range(range_start..=range_end)? {
            let (_key, value) = result?;
            let event: ReplayEvent = deserialize(value.value())?;
            events.push(event);
        }
        Ok(events)
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

    /// Regression test for the real bug: `load_range` used to call
    /// `load_all()` and filter in memory, so it deserialized every
    /// event in the log on every call, regardless of the requested
    /// range's width. `load_range_filters_correctly` above checks only
    /// output correctness and passes identically whether `load_range`
    /// is implemented as a native range scan or as `load_all` plus a
    /// filter, so it cannot tell the two implementations apart on its
    /// own; this test can, because it deliberately corrupts the raw
    /// bytes of events OUTSIDE the requested range at the storage
    /// layer. A `load_all()`-based implementation must deserialize
    /// those corrupted bytes too (since it reads the whole table
    /// before filtering) and would return `Err`, whereas a real range
    /// scan never reads them and succeeds.
    #[test]
    fn load_range_only_touches_keys_in_range_not_the_whole_table() {
        let path = temp_log_path();
        let log = AppendOnlyEventLog::open(&path, "node-0").unwrap();
        for i in 0..10u128 {
            log.append(i, vec![], format!("p{i}").into_bytes()).unwrap();
        }
        let all = log.load_all().unwrap();
        let ts_4 = all[4].lamport_ts;
        let ts_6 = all[6].lamport_ts;

        // Corrupt the stored value bytes for every event OUTSIDE
        // [ts_4, ts_6] directly at the redb storage layer, bypassing
        // this module's own serialize()/append() entirely, so a
        // correct range scan of [ts_4, ts_6] never reads these keys
        // at all, while load_all() would read every one of them.
        {
            let write_txn = log.db.begin_write().unwrap();
            {
                let mut table = write_txn.open_table(EVENTS).unwrap();
                for event in &all {
                    if event.lamport_ts < ts_4 || event.lamport_ts > ts_6 {
                        table
                            .insert(
                                (event.lamport_ts, event.event_id),
                                b"THIS IS NOT VALID BINCODE".as_slice(),
                            )
                            .unwrap();
                    }
                }
            }
            write_txn.commit().unwrap();
        }

        // A real range scan must succeed: it never touches the
        // corrupted keys outside [ts_4, ts_6].
        let range = log.load_range(ts_4, ts_6).unwrap();
        assert_eq!(range.len(), 3);
        assert_eq!(range[0].lamport_ts, ts_4);
        assert_eq!(range[2].lamport_ts, ts_6);

        // Sanity check that the corruption above is real and would
        // actually be detected: load_all() (which does read
        // everything) must now fail on the corrupted keys. If this
        // assertion doesn't hold, the corruption step above didn't do
        // anything, and the test above isn't actually proving anything.
        assert!(
            log.load_all().is_err(),
            "expected load_all() to fail on the deliberately corrupted \
             out-of-range keys; if it succeeds, this test's corruption \
             step is not working and load_range_only_touches_keys_in_range \
             is not actually distinguishing a range scan from a full scan"
        );
    }

    #[test]
    fn open_existing_refuses_to_silently_create_a_new_log() {
        // The exact bug this fix addresses: a genuinely nonexistent
        // path must error, not silently produce an empty, freshly
        // created database indistinguishable from "this log has zero
        // events."
        let path = temp_log_path();
        assert!(!path.exists());
        let result = AppendOnlyEventLog::open_existing(&path, "node-0");
        assert!(matches!(result, Err(EventLogError::LogNotFound(_))));
        assert!(!path.exists(), "open_existing must not create a file on failure");
    }

    #[test]
    fn open_existing_succeeds_against_a_real_prior_log() {
        let path = temp_log_path();
        // First, a real log with real content, via the normal open().
        {
            let log = AppendOnlyEventLog::open(&path, "node-0").unwrap();
            log.append(1, vec![], b"real event".to_vec()).unwrap();
        }
        // Then, open_existing must find it and see the real content.
        let reopened = AppendOnlyEventLog::open_existing(&path, "node-0").unwrap();
        let events = reopened.load_all().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].payload, b"real event");
    }

    #[test]
    fn open_still_creates_a_new_log_when_that_is_the_actual_intent() {
        // open() (not open_existing()) must still work exactly as
        // before for the legitimate case: starting a brand new log.
        // This fix changes behavior only for open_existing(), not for
        // open()'s own documented, intentional create-if-missing role.
        let path = temp_log_path();
        assert!(!path.exists());
        let log = AppendOnlyEventLog::open(&path, "node-0").unwrap();
        assert!(log.is_empty().unwrap());
    }

    #[test]
    fn emitter_node_id_is_recorded() {
        let log = AppendOnlyEventLog::open(temp_log_path(), "test-node-42").unwrap();
        log.append(1, vec![], b"payload".to_vec()).unwrap();
        let events = log.load_all().unwrap();
        assert_eq!(events[0].emitter_node_id, "test-node-42");
    }
}