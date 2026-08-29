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

func (br *byteReader) readBytes() []byte {
	length := br.readU64()
	if br.err != nil {
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
type RoutingDecisionPayload struct {
	ReplayKey        string
	SelectedWorkerID string
	RoutingScore     float64
	StrategyName     string
	Reason           string
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