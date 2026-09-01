package eventlog

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// buildBincodeReplayEvent constructs the exact byte sequence bincode's
// default encoding would produce for a ReplayEvent with the given
// field values, by hand, matching the encoding rules documented in
// this package's doc comment. This is the independent check: if
// DecodeReplayEvent's output doesn't match what THIS function
// constructs, one of the two is wrong, and this function is built
// directly from bincode's documented wire format, not from
// DecodeReplayEvent's own logic, so it can't share a bug with it.
func buildBincodeReplayEvent(lamportTs uint64, eventID [16]byte, depIDs [][16]byte, emitterNodeID string, payload []byte) []byte {
	var buf bytes.Buffer

	var u64buf [8]byte
	binary.LittleEndian.PutUint64(u64buf[:], lamportTs)
	buf.Write(u64buf[:])

	buf.Write(eventID[:])

	binary.LittleEndian.PutUint64(u64buf[:], uint64(len(depIDs)))
	buf.Write(u64buf[:])
	for _, id := range depIDs {
		buf.Write(id[:])
	}

	binary.LittleEndian.PutUint64(u64buf[:], uint64(len(emitterNodeID)))
	buf.Write(u64buf[:])
	buf.WriteString(emitterNodeID)

	binary.LittleEndian.PutUint64(u64buf[:], uint64(len(payload)))
	buf.Write(u64buf[:])
	buf.Write(payload)

	return buf.Bytes()
}

func TestDecodeReplayEvent_MatchesHandBuiltBincodeBytes(t *testing.T) {
	eventID := [16]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	depID := [16]byte{99, 98, 97}
	raw := buildBincodeReplayEvent(42, eventID, [][16]byte{depID}, "gateway-node-0", []byte("hello payload"))

	event, err := DecodeReplayEvent(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("DecodeReplayEvent failed: %v", err)
	}

	if event.LamportTs != 42 {
		t.Errorf("LamportTs = %d, want 42", event.LamportTs)
	}
	if event.EventID != eventID {
		t.Errorf("EventID = %v, want %v", event.EventID, eventID)
	}
	if len(event.DependencyIDs) != 1 || event.DependencyIDs[0] != depID {
		t.Errorf("DependencyIDs = %v, want [%v]", event.DependencyIDs, depID)
	}
	if event.EmitterNodeID != "gateway-node-0" {
		t.Errorf("EmitterNodeID = %q, want %q", event.EmitterNodeID, "gateway-node-0")
	}
	if string(event.Payload) != "hello payload" {
		t.Errorf("Payload = %q, want %q", event.Payload, "hello payload")
	}
}

func TestDecodeReplayEvent_EmptyDependencyIDs(t *testing.T) {
	raw := buildBincodeReplayEvent(1, [16]byte{}, nil, "node", []byte("p"))
	event, err := DecodeReplayEvent(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("DecodeReplayEvent failed: %v", err)
	}
	if len(event.DependencyIDs) != 0 {
		t.Errorf("DependencyIDs = %v, want empty", event.DependencyIDs)
	}
}

// buildBincodeRoutingDecisionPayload mirrors
// RoutingDecisionPayload's field order (see this package's own
// RoutingDecisionPayload doc comment), constructed independently of
// DecodeRoutingDecisionPayload's implementation, same rationale as
// buildBincodeReplayEvent above.
// buildBincodeRoutingDecisionPayload matches RoutingDecisionPayload's
// CURRENT (post-oracle-snapshot) field order. oracleFields is nil for
// a None snapshot (all five Option fields absent, tag byte 0x00 each),
// or a *oracleTestFields for a Some snapshot (tag byte 0x01 each,
// followed by the real values), mirrors Rust's construction, where
// all five come from one Option<OracleSnapshot> and are always
// all-Some or all-None together.
type oracleTestFields struct {
	cacheHitProb       float64
	predictedLatencyMs float64
	slaAffinity        float64
	kvPressure         float64
	nObservations      uint64
}

func buildBincodeRoutingDecisionPayload(replayKey, workerID string, score float64, strategy, reason string, oracleFields *oracleTestFields) []byte {
	var buf bytes.Buffer
	writeString := func(s string) {
		var u64buf [8]byte
		binary.LittleEndian.PutUint64(u64buf[:], uint64(len(s)))
		buf.Write(u64buf[:])
		buf.WriteString(s)
	}
	writeOptionalF64 := func(v *float64) {
		if v == nil {
			buf.WriteByte(0)
			return
		}
		buf.WriteByte(1)
		var f64buf [8]byte
		binary.LittleEndian.PutUint64(f64buf[:], math.Float64bits(*v))
		buf.Write(f64buf[:])
	}
	writeOptionalU64 := func(v *uint64) {
		if v == nil {
			buf.WriteByte(0)
			return
		}
		buf.WriteByte(1)
		var u64buf [8]byte
		binary.LittleEndian.PutUint64(u64buf[:], *v)
		buf.Write(u64buf[:])
	}

	writeString(replayKey)
	writeString(workerID)

	var f64buf [8]byte
	binary.LittleEndian.PutUint64(f64buf[:], math.Float64bits(score))
	buf.Write(f64buf[:])

	writeString(strategy)
	writeString(reason)

	if oracleFields == nil {
		writeOptionalF64(nil)
		writeOptionalF64(nil)
		writeOptionalF64(nil)
		writeOptionalF64(nil)
		writeOptionalU64(nil)
	} else {
		writeOptionalF64(&oracleFields.cacheHitProb)
		writeOptionalF64(&oracleFields.predictedLatencyMs)
		writeOptionalF64(&oracleFields.slaAffinity)
		writeOptionalF64(&oracleFields.kvPressure)
		writeOptionalU64(&oracleFields.nObservations)
	}

	return buf.Bytes()
}

func TestDecodeRoutingDecisionPayload_MatchesHandBuiltBincodeBytes_NoOracleSnapshot(t *testing.T) {
	raw := buildBincodeRoutingDecisionPayload("replay-key-abc", "worker-1", 0.8734, "round_robin", "index 0", nil)

	payload, err := DecodeRoutingDecisionPayload(raw)
	if err != nil {
		t.Fatalf("DecodeRoutingDecisionPayload failed: %v", err)
	}

	if payload.ReplayKey != "replay-key-abc" {
		t.Errorf("ReplayKey = %q, want %q", payload.ReplayKey, "replay-key-abc")
	}
	if payload.SelectedWorkerID != "worker-1" {
		t.Errorf("SelectedWorkerID = %q, want %q", payload.SelectedWorkerID, "worker-1")
	}
	if payload.RoutingScore != 0.8734 {
		t.Errorf("RoutingScore = %v, want 0.8734", payload.RoutingScore)
	}
	if payload.StrategyName != "round_robin" {
		t.Errorf("StrategyName = %q, want %q", payload.StrategyName, "round_robin")
	}
	if payload.Reason != "index 0" {
		t.Errorf("Reason = %q, want %q", payload.Reason, "index 0")
	}
	if payload.HasOracleSnapshot() {
		t.Error("expected no oracle snapshot for a None-constructed payload")
	}
	if payload.OracleCacheHitProb != nil || payload.OraclePredictedLatencyMs != nil ||
		payload.OracleSLAAffinity != nil || payload.OracleKVPressure != nil ||
		payload.OracleNObservations != nil {
		t.Error("expected all five oracle fields to be nil together")
	}
}

func TestDecodeRoutingDecisionPayload_MatchesHandBuiltBincodeBytes_WithOracleSnapshot(t *testing.T) {
	oracle := &oracleTestFields{
		cacheHitProb:       0.42,
		predictedLatencyMs: 1234.5,
		slaAffinity:        0.9,
		kvPressure:         0.15,
		nObservations:      37,
	}
	raw := buildBincodeRoutingDecisionPayload("replay-key-def", "worker-0", 0.712, "semantic", "semantic:score=0.712", oracle)

	payload, err := DecodeRoutingDecisionPayload(raw)
	if err != nil {
		t.Fatalf("DecodeRoutingDecisionPayload failed: %v", err)
	}

	if !payload.HasOracleSnapshot() {
		t.Fatal("expected an oracle snapshot to be present")
	}
	if *payload.OracleCacheHitProb != 0.42 {
		t.Errorf("OracleCacheHitProb = %v, want 0.42", *payload.OracleCacheHitProb)
	}
	if *payload.OraclePredictedLatencyMs != 1234.5 {
		t.Errorf("OraclePredictedLatencyMs = %v, want 1234.5", *payload.OraclePredictedLatencyMs)
	}
	if *payload.OracleSLAAffinity != 0.9 {
		t.Errorf("OracleSLAAffinity = %v, want 0.9", *payload.OracleSLAAffinity)
	}
	if *payload.OracleKVPressure != 0.15 {
		t.Errorf("OracleKVPressure = %v, want 0.15", *payload.OracleKVPressure)
	}
	if *payload.OracleNObservations != 37 {
		t.Errorf("OracleNObservations = %v, want 37", *payload.OracleNObservations)
	}
}

func TestDecodeRoutingDecisionPayload_TruncatedBytesReturnsError(t *testing.T) {
	raw := buildBincodeRoutingDecisionPayload("key", "worker", 1.0, "round_robin", "index 0", nil)
	truncated := raw[:len(raw)-5] // cut off partway through the last field

	_, err := DecodeRoutingDecisionPayload(truncated)
	if err == nil {
		t.Error("expected an error decoding truncated bytes, got nil")
	}
}

func TestDecodeReplayEvent_RefusesUnreasonablyLargeLengthPrefix(t *testing.T) {
	// A length prefix claiming an absurd size (here, larger than
	// maxReasonableFieldBytes), the exact failure shape a schema
	// mismatch or corrupted read produces, must error cleanly, not
	// attempt to allocate that much memory.
	var buf bytes.Buffer
	var u64buf [8]byte
	binary.LittleEndian.PutUint64(u64buf[:], 1) // lamport_ts
	buf.Write(u64buf[:])
	buf.Write(make([]byte, 16)) // event_id
	binary.LittleEndian.PutUint64(u64buf[:], 0) // zero dependency_ids
	buf.Write(u64buf[:])
	binary.LittleEndian.PutUint64(u64buf[:], 1<<40) // absurd emitter_node_id length
	buf.Write(u64buf[:])

	_, err := DecodeReplayEvent(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("expected an error for an unreasonably large length prefix, got nil")
	}
}