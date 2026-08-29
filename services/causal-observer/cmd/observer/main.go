// Package main: stratum-causal-observer.
//
// SCOPE, STATED HONESTLY
// =======================
// This is Path A of a two-part plan (see docs/SCOPE.md): observability
// over the event log exactly as it is TODAY, not the full causal
// observability RFC-001 describes. RFC-001's CausalDecisionEvent proto
// (with OracleStateSnapshot, bandit weights, full request lifecycle)
// was never implemented, what actually exists, and what this
// service actually consumes, is stratum-router's much smaller
// RoutingDecisionPayload (replay_key, selected_worker_id,
// routing_score, strategy_name, reason), confirmed against the real
// source before writing this (crates/stratum-router/src/router.rs).
// Path B, extending RoutingDecisionPayload with an oracle state
// snapshot closer to RFC-001's original design, is a separate,
// later piece of work, not done here, this service is built to
// keep working unchanged once that lands, since it decodes payload
// fields it knows about and ignores anything it doesn't recognize.
//
// WHAT THIS DOES
// ================
// Invokes dump_events (stratum-replay's bin, see that file's own doc
// comment) as a subprocess against a given event log path, decodes
// each event via the eventlog package (a hand-written bincode
// decoder matching the real Rust struct layouts, verified field by
// field against the actual source), and reports summary statistics:
// total event count, per-worker routing decision count, and per-
// strategy routing decision count. This is the concrete thing this
// service can prove works correctly against real data today; a full
// OpenTelemetry export pipeline is the natural next extension once
// this core is trustworthy, not built speculatively ahead of it.
package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"

	"github.com/stratum-project/stratum/causal-observer/internal/eventlog"
)

func main() {
	eventLogPath := flag.String("path", "", "path to a stratum event log (.redb file)")
	dumpEventsBin := flag.String(
		"dump-events-bin", "",
		"path to the built dump_events binary. If empty, runs "+
			"'cargo run --bin dump_events --' from the stratum-replay crate directory.",
	)
	stratumReplayDir := flag.String(
		"stratum-replay-dir", "",
		"path to the stratum-replay crate directory (used only when "+
			"-dump-events-bin is not set)",
	)
	flag.Parse()

	if *eventLogPath == "" {
		fmt.Fprintln(os.Stderr, "usage: observer -path <event_log.redb> [-dump-events-bin <path> | -stratum-replay-dir <path>]")
		os.Exit(2)
	}

	events, err := readAllEvents(*eventLogPath, *dumpEventsBin, *stratumReplayDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "stratum-causal-observer: %v\n", err)
		os.Exit(1)
	}

	report := summarize(events)
	report.Print(os.Stdout)
}

// readAllEvents runs dump_events (either a pre-built binary, or via
// `cargo run` from the stratum-replay crate directory) against
// eventLogPath and decodes every event it emits.
func readAllEvents(eventLogPath, dumpEventsBin, stratumReplayDir string) ([]eventlog.ReplayEvent, error) {
	var cmd *exec.Cmd
	if dumpEventsBin != "" {
		cmd = exec.Command(dumpEventsBin, "--path", eventLogPath)
	} else {
		if stratumReplayDir == "" {
			return nil, fmt.Errorf("either -dump-events-bin or -stratum-replay-dir must be set")
		}
		cmd = exec.Command("cargo", "run", "--quiet", "--bin", "dump_events", "--", "--path", eventLogPath)
		cmd.Dir = stratumReplayDir
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("creating stdout pipe: %w", err)
	}
	cmd.Stderr = os.Stderr // pass dump_events' own diagnostic output straight through

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting dump_events: %w", err)
	}

	events, readErr := decodeStream(stdout)

	waitErr := cmd.Wait()
	if waitErr != nil {
		return nil, fmt.Errorf("dump_events exited with error: %w", waitErr)
	}
	if readErr != nil {
		return nil, fmt.Errorf("decoding dump_events output: %w", readErr)
	}
	return events, nil
}

// decodeStream reads dump_events' length-prefixed stream: a 4-byte
// little-endian u32 length, then that many bytes of
// bincode::serialize(&ReplayEvent), repeated until EOF. See
// dump_events.rs's own doc comment for this exact format.
func decodeStream(r io.Reader) ([]eventlog.ReplayEvent, error) {
	br := bufio.NewReader(r)
	var events []eventlog.ReplayEvent

	for {
		var lenBuf [4]byte
		_, err := io.ReadFull(br, lenBuf[:])
		if err == io.EOF {
			break // clean end of stream
		}
		if err != nil {
			return nil, fmt.Errorf("reading length prefix: %w", err)
		}
		length := binary.LittleEndian.Uint32(lenBuf[:])

		eventBytes := make([]byte, length)
		if _, err := io.ReadFull(br, eventBytes); err != nil {
			return nil, fmt.Errorf("reading %d-byte event body: %w", length, err)
		}

		event, err := eventlog.DecodeReplayEvent(bufio.NewReader(byteSliceReader(eventBytes)))
		if err != nil {
			return nil, fmt.Errorf("decoding event: %w", err)
		}
		events = append(events, event)
	}

	return events, nil
}

func byteSliceReader(b []byte) io.Reader {
	return &sliceReaderMain{data: b}
}

type sliceReaderMain struct {
	data []byte
	pos  int
}

func (s *sliceReaderMain) Read(p []byte) (int, error) {
	if s.pos >= len(s.data) {
		return 0, io.EOF
	}
	n := copy(p, s.data[s.pos:])
	s.pos += n
	return n, nil
}

// Report is the summary this service produces from a real event log.
type Report struct {
	TotalEvents             int
	UndecodableAsRouting    int
	RoutingDecisionsByWorker map[string]int
	RoutingDecisionsByStrategy map[string]int
}

func summarize(events []eventlog.ReplayEvent) Report {
	report := Report{
		TotalEvents:                len(events),
		RoutingDecisionsByWorker:   map[string]int{},
		RoutingDecisionsByStrategy: map[string]int{},
	}

	for _, event := range events {
		payload, err := eventlog.DecodeRoutingDecisionPayload(event.Payload)
		if err != nil {
			// Not every event is necessarily a routing decision, see
			// eventlog.DecodeRoutingDecisionPayload's own doc comment.
			// Skip, don't fail the whole report over one event kind
			// this service doesn't yet know how to interpret.
			report.UndecodableAsRouting++
			continue
		}
		report.RoutingDecisionsByWorker[payload.SelectedWorkerID]++
		report.RoutingDecisionsByStrategy[payload.StrategyName]++
	}

	return report
}

func (r Report) Print(w io.Writer) {
	fmt.Fprintf(w, "stratum-causal-observer report\n")
	fmt.Fprintf(w, "===============================\n")
	fmt.Fprintf(w, "total events in log:                %d\n", r.TotalEvents)
	fmt.Fprintf(w, "events not decodable as routing:     %d\n", r.UndecodableAsRouting)
	fmt.Fprintf(w, "\nrouting decisions by worker:\n")
	printSortedCounts(w, r.RoutingDecisionsByWorker)
	fmt.Fprintf(w, "\nrouting decisions by strategy:\n")
	printSortedCounts(w, r.RoutingDecisionsByStrategy)
}

func printSortedCounts(w io.Writer, counts map[string]int) {
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Fprintf(w, "  %-30s %d\n", k, counts[k])
	}
}