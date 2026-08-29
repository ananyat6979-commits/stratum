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
func buildBincodeRoutingDecisionPayload(replayKey, workerID string, score float64, strategy, reason string) []byte {
	var buf bytes.Buffer
	writeString := func(s string) {
		var u64buf [8]byte
		binary.LittleEndian.PutUint64(u64buf[:], uint64(len(s)))
		buf.Write(u64buf[:])
		buf.WriteString(s)
	}

	writeString(replayKey)
	writeString(workerID)

	var f64buf [8]byte
	binary.LittleEndian.PutUint64(f64buf[:], math.Float64bits(score))
	buf.Write(f64buf[:])

	writeString(strategy)
	writeString(reason)

	return buf.Bytes()
}

func TestDecodeRoutingDecisionPayload_MatchesHandBuiltBincodeBytes(t *testing.T) {
	raw := buildBincodeRoutingDecisionPayload("replay-key-abc", "worker-1", 0.8734, "semantic", "cache_hit_prob dominant")

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
	if payload.StrategyName != "semantic" {
		t.Errorf("StrategyName = %q, want %q", payload.StrategyName, "semantic")
	}
	if payload.Reason != "cache_hit_prob dominant" {
		t.Errorf("Reason = %q, want %q", payload.Reason, "cache_hit_prob dominant")
	}
}

func TestDecodeRoutingDecisionPayload_TruncatedBytesReturnsError(t *testing.T) {
	raw := buildBincodeRoutingDecisionPayload("key", "worker", 1.0, "round_robin", "index 0")
	truncated := raw[:len(raw)-5] // cut off partway through the last field

	_, err := DecodeRoutingDecisionPayload(truncated)
	if err == nil {
		t.Error("expected an error decoding truncated bytes, got nil")
	}
}