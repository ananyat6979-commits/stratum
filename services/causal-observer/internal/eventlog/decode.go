// Package eventlog decodes STRATUM's redb-backed event log
// (crates/stratum-replay/src/event_log.rs) directly, in Go, without
// going through Rust at all.
//
// WHY A HAND-WRITTEN DECODER, NOT A LIBRARY
// ==============================================
// The event log's on-disk values are bincode::serialize(&ReplayEvent)
// bytes, see event_log.rs's own module doc comment: "bincode not
// proto: the event log is an internal Rust-only artifact." No
// well-established, widely-trusted Go implementation of Rust's
// specific bincode encoding conventions exists that this project can
// point to as canonical. Rather than depend on an unverified
// third-party decoder for something this load-bearing, this package
// implements bincode's DEFAULT encoding directly, matching exactly
// what the `bincode` crate (as used, unconfigured, in event_log.rs
// and router.rs) actually produces:
//   - Fixed-width integers: little-endian, no padding.
//   - String / Vec<T>: u64 little-endian length prefix, then the raw
//     bytes (String) or that many serialized elements (Vec<T>).
//   - Struct fields: serialized in declaration order, no field tags,
//     no type markers, this means the decoder MUST match the exact
//     field order of the Rust struct being decoded, verified against
//     the real source (see decodeRoutingDecisionPayload's doc comment
//     for the exact struct this matches).
//
// This package does NOT read redb's on-disk file format directly.
// redb is accessed via a small companion Rust CLI
// (cmd/dump_events/main.rs, in stratum-replay) that opens the real
// database with the real, tested AppendOnlyEventLog::load_all() and
// writes each event's raw bincode bytes to stdout, newline-delimited
// with a length prefix, decoding redb's B-tree format independently
// in Go would be a second, unverified implementation of something
// crates/stratum-replay/src/event_log.rs already implements and tests
// correctly. This package only has to trust bincode decoding, not
// redb's storage format too.
package eventlog

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// ReplayEvent mirrors crates/stratum-replay/src/event_log.rs's
// ReplayEvent struct EXACTLY, including field order, which bincode's
// default encoding depends on. If that Rust struct's fields are
// reordered, added to, or removed, this must be updated to match, or
// decoding will silently produce wrong values without necessarily
// erroring (bincode has no self-describing tags to catch this).
//
// Rust source (verified against the real file before writing this):
//
//	pub struct ReplayEvent {
//	    pub lamport_ts: u64,
//	    pub event_id: u128,
//	    pub dependency_ids: Vec<u128>,
//	    pub emitter_node_id: String,
//	    pub payload: Vec<u8>,
//	}
type ReplayEvent struct {
	LamportTs      uint64
	EventID        [16]byte // u128, little-endian, as raw bytes, see readU128
	DependencyIDs  [][16]byte
	EmitterNodeID  string
	Payload        []byte
}

type byteReader struct {
	r   *bufio.Reader
	err error
}

func (br *byteReader) readU64() uint64 {
	if br.err != nil {
		return 0
	}
	var buf [8]byte
	_, err := io.ReadFull(br.r, buf[:])
	if err != nil {
		br.err = fmt.Errorf("reading u64: %w", err)
		return 0
	}
	return binary.LittleEndian.Uint64(buf[:])
}

// readU128 reads bincode's 16-byte little-endian encoding of Rust's
// u128, returned as raw bytes rather than a Go integer type (Go has
// no native 128-bit integer as of this project's Go version, and
// event_id/dependency_ids are only ever compared for equality or
// displayed, never arithmetically manipulated, so raw bytes are
// sufficient and avoid a lossy or overly-complex big.Int conversion).
func (br *byteReader) readU128() [16]byte {
	var buf [16]byte
	if br.err != nil {
		return buf
	}
	_, err := io.ReadFull(br.r, buf[:])
	if err != nil {
		br.err = fmt.Errorf("reading u128: %w", err)
	}
	return buf
}

// maxReasonableFieldBytes bounds any single length-prefixed field this
// decoder will attempt to allocate for. Found necessary directly, not
// theoretically: a schema mismatch between this decoder and the real
// event log it was reading against (see RoutingDecisionPayload's
// oracle-snapshot extension) produced exactly this failure mode,
// reading a length prefix at the wrong byte offset, in the wrong
// field, yields a garbage uint64 that could be enormous. Without this
// cap, make([]byte, length) would attempt to allocate that much
// memory and likely panic (OOM) rather than return a clean decode
// error, a crash for the whole process over one malformed event,
// not the "skip this event, keep going" behavior the rest of this
// package is designed around. 64MB is far larger than any real field
// this decoder handles (prompts, worker IDs, reason strings), while
// still catching genuinely corrupted/misaligned reads.
const maxReasonableFieldBytes = 64 * 1024 * 1024

// maxReasonableDependencyCount bounds ReplayEvent.DependencyIDs'
// element count before make([][16]byte, depCount) allocates for it.
// depCount is read via readU64() directly in DecodeReplayEvent below,
// bypassing readBytes() and its maxReasonableFieldBytes guard
// entirely, so it is subject to the exact same failure mode that
// guard exists to prevent: a schema mismatch or misaligned read
// yielding a garbage uint64, which here would attempt to allocate up
// to depCount*16 bytes and likely panic (OOM) on one malformed event.
//
// The bound below is set from the real, verified usage of
// dependency_ids in this codebase, not a guess: the only call site
// that constructs a ReplayEvent's dependencies is
// crates/stratum-router/src/router.rs's route_and_log, which always
// passes vec![ingress_event_id], a single element (verified directly
// against that source; grep for ".append(" in stratum-router turns up
// exactly one call site). 64 gives comfortable headroom for a future
// event kind with a handful of real causal parents, while still
// catching a schema mismatch or corrupted read by many orders of
// magnitude. If a real use case for more dependencies per event is
// added later, raise this deliberately alongside that change, with
// the same kind of verified justification, not by picking a larger
// round number preemptively.
const maxReasonableDependencyCount = 64

func (br *byteReader) readBytes() []byte {
	length := br.readU64()
	if br.err != nil {
		return nil
	}
	if length > maxReasonableFieldBytes {
		br.err = fmt.Errorf(
			"refusing to allocate %d bytes for a length-prefixed field "+
				"(max %d) : almost certainly a schema mismatch or "+
				"corrupted read, not a real field this large",
			length, maxReasonableFieldBytes,
		)
		return nil
	}
	buf := make([]byte, length)
	_, err := io.ReadFull(br.r, buf)
	if err != nil {
		br.err = fmt.Errorf("reading %d-byte buffer: %w", length, err)
		return nil
	}
	return buf
}

func (br *byteReader) readString() string {
	return string(br.readBytes())
}

// DecodeReplayEvent decodes one bincode-encoded ReplayEvent from r.
// Field order MUST match event_log.rs's ReplayEvent struct exactly 
// see this file's package doc comment.
func DecodeReplayEvent(r io.Reader) (ReplayEvent, error) {
	br := &byteReader{r: bufio.NewReader(r)}

	var event ReplayEvent
	event.LamportTs = br.readU64()
	event.EventID = br.readU128()

	depCount := br.readU64()
	if br.err == nil && depCount > maxReasonableDependencyCount {
		br.err = fmt.Errorf(
			"refusing to allocate %d dependency IDs (max %d): "+
				"almost certainly a schema mismatch or corrupted read, "+
				"not a real event with this many causal dependencies",
			depCount, maxReasonableDependencyCount,
		)
	}
	if br.err == nil {
		event.DependencyIDs = make([][16]byte, depCount)
		for i := uint64(0); i < depCount; i++ {
			event.DependencyIDs[i] = br.readU128()
		}
	}

	event.EmitterNodeID = br.readString()
	event.Payload = br.readBytes()

	if br.err != nil {
		return ReplayEvent{}, br.err
	}
	return event, nil
}

// RoutingDecisionPayload mirrors crates/stratum-router/src/router.rs's
// RoutingDecisionPayload struct EXACTLY, including field order
// verified against the real source before writing this:
//
//	pub struct RoutingDecisionPayload {
//	    pub replay_key: String,
//	    pub selected_worker_id: String,
//	    pub routing_score: f64,
//	    pub strategy_name: String,
//	    pub reason: String,
//	}
//
// This is a distinct decode step from DecodeReplayEvent: a
// ReplayEvent's Payload field is opaque bytes at the event-log layer
// (see event_log.rs's own doc comment: "Opaque bytes to this layer"),
// only meaningful once further decoded as a specific payload type.
// Not every ReplayEvent in the log is necessarily a routing decision
// callers should be prepared for DecodeRoutingDecisionPayload to
// fail on payloads from a different, not-yet-supported event kind,
// and treat that as "skip this event," not a fatal error for the
// whole log.
// RoutingDecisionPayload mirrors crates/stratum-router/src/router.rs's
// RoutingDecisionPayload struct EXACTLY, including field order,
// verified against the real source before writing this. The five
// Oracle* fields are new: bincode encodes Option<T> as a single tag
// byte (0x00 = None, 0x01 = Some, followed by T's bytes if Some),
// see readOptionalF64/readOptionalU64 below for the matching decode.
//
//	pub struct RoutingDecisionPayload {
//	    pub replay_key: String,
//	    pub selected_worker_id: String,
//	    pub routing_score: f64,
//	    pub strategy_name: String,
//	    pub reason: String,
//	    pub oracle_cache_hit_prob: Option<f64>,
//	    pub oracle_predicted_latency_ms: Option<f64>,
//	    pub oracle_sla_affinity: Option<f64>,
//	    pub oracle_kv_pressure: Option<f64>,
//	    pub oracle_n_observations: Option<u64>,
//	}
type RoutingDecisionPayload struct {
	ReplayKey                string
	SelectedWorkerID         string
	RoutingScore             float64
	StrategyName             string
	Reason                   string
	OracleCacheHitProb       *float64
	OraclePredictedLatencyMs *float64
	OracleSLAAffinity        *float64
	OracleKVPressure         *float64
	OracleNObservations      *uint64
}

// HasOracleSnapshot reports whether all five oracle fields are
// present, matching Rust's construction (see router.rs's route_and_log:
// all five come from the same Option<OracleSnapshot>, so they are
// always all-Some or all-None together, never a partial mix).
func (p RoutingDecisionPayload) HasOracleSnapshot() bool {
	return p.OracleCacheHitProb != nil
}

func (br *byteReader) readOptionTag() bool {
	if br.err != nil {
		return false
	}
	var tag [1]byte
	_, err := io.ReadFull(br.r, tag[:])
	if err != nil {
		br.err = fmt.Errorf("reading Option<T> tag byte: %w", err)
		return false
	}
	if tag[0] != 0 && tag[0] != 1 {
		br.err = fmt.Errorf("invalid Option<T> tag byte: %d (expected 0 or 1)", tag[0])
		return false
	}
	return tag[0] == 1
}

func (br *byteReader) readOptionalF64() *float64 {
	if br.err != nil {
		return nil
	}
	if !br.readOptionTag() {
		return nil // None, or an error already recorded by readOptionTag
	}
	var buf [8]byte
	_, err := io.ReadFull(br.r, buf[:])
	if err != nil {
		br.err = fmt.Errorf("reading Option<f64> inner value: %w", err)
		return nil
	}
	v := decodeFloat64(buf)
	return &v
}

func (br *byteReader) readOptionalU64() *uint64 {
	if br.err != nil {
		return nil
	}
	if !br.readOptionTag() {
		return nil
	}
	v := br.readU64()
	if br.err != nil {
		return nil
	}
	return &v
}

func DecodeRoutingDecisionPayload(payload []byte) (RoutingDecisionPayload, error) {
	br := &byteReader{r: bufio.NewReader(newSliceReader(payload))}

	var p RoutingDecisionPayload
	p.ReplayKey = br.readString()
	p.SelectedWorkerID = br.readString()

	if br.err == nil {
		var buf [8]byte
		_, err := io.ReadFull(br.r, buf[:])
		if err != nil {
			br.err = fmt.Errorf("reading routing_score f64: %w", err)
		} else {
			p.RoutingScore = decodeFloat64(buf)
		}
	}

	p.StrategyName = br.readString()
	p.Reason = br.readString()

	p.OracleCacheHitProb = br.readOptionalF64()
	p.OraclePredictedLatencyMs = br.readOptionalF64()
	p.OracleSLAAffinity = br.readOptionalF64()
	p.OracleKVPressure = br.readOptionalF64()
	p.OracleNObservations = br.readOptionalU64()

	if br.err != nil {
		return RoutingDecisionPayload{}, br.err
	}
	return p, nil
}

func decodeFloat64(buf [8]byte) float64 {
	bits := binary.LittleEndian.Uint64(buf[:])
	return math.Float64frombits(bits)
}

// newSliceReader avoids pulling in bytes.NewReader's slightly
// different io.Reader semantics inconsistency risk; a tiny local
// wrapper keeps this package's only import surface explicit.
func newSliceReader(b []byte) io.Reader {
	return &sliceReader{data: b}
}

type sliceReader struct {
	data []byte
	pos  int
}

func (s *sliceReader) Read(p []byte) (int, error) {
	if s.pos >= len(s.data) {
		return 0, io.EOF
	}
	n := copy(p, s.data[s.pos:])
	s.pos += n
	return n, nil
}