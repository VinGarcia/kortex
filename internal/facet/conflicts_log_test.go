package facet

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

// errWriter always fails with err, simulating a sink that cannot be written
// to (e.g. a closed file or a full disk) so conflictsLogger.record's
// u-never-fail-silently contract (return the error rather than swallow it,
// unlike facetLogger.record) can be exercised.
type errWriter struct{ err error }

func (w errWriter) Write(p []byte) (int, error) { return 0, w.err }

// shortWriter reports writing fewer bytes than it was given, without itself
// returning an error — the other failure shape conflictsLogger.record must
// catch (io.Writer contract: a short write with a nil error is valid and
// must still be treated as a failure by the caller).
type shortWriter struct{}

func (shortWriter) Write(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	return len(p) - 1, nil
}

func TestConflictsLogger_record_nilSinkIsNoOp(t *testing.T) {
	c := &conflictsLogger{}
	if err := c.record(ladderTailEntry{Outcome: ladderOutcomeApprove}); err != nil {
		t.Fatalf("nil sink must be a no-op, got error: %v", err)
	}
}

func TestConflictsLogger_record_nilReceiverIsNoOp(t *testing.T) {
	var c *conflictsLogger
	if err := c.record(ladderTailEntry{Outcome: ladderOutcomeApprove}); err != nil {
		t.Fatalf("nil receiver must be a no-op, got error: %v", err)
	}
}

func TestConflictsLogger_record_writeFailureIsReturned(t *testing.T) {
	c := &conflictsLogger{sink: errWriter{err: errors.New("disk full")}}
	err := c.record(ladderTailEntry{Outcome: ladderOutcomeApprove})
	if err == nil {
		t.Fatal("expected a write failure to be returned, got nil")
	}
	if !strings.Contains(err.Error(), "disk full") {
		t.Errorf("error = %v, want it to wrap the underlying write error", err)
	}
}

func TestConflictsLogger_record_shortWriteIsReturned(t *testing.T) {
	c := &conflictsLogger{sink: shortWriter{}}
	err := c.record(ladderTailEntry{Outcome: ladderOutcomeApprove})
	if err == nil {
		t.Fatal("expected a short write to be reported as an error, got nil")
	}
	if !strings.Contains(err.Error(), "wrote") {
		t.Errorf("error = %v, want it to describe the short write", err)
	}
}

// TestConflictsLogger_record_writesOneCompleteLine locks in the JSONL shape
// ladderTailEntry documents: one self-contained JSON object per ladder loop,
// carrying the ordered tail verbatim.
func TestConflictsLogger_record_writesOneCompleteLine(t *testing.T) {
	var buf bytes.Buffer
	c := &conflictsLogger{sink: &buf}
	tail := []ladderStep{
		{Type: "draft", Round: 1, Text: "rascunho"},
		{Type: "critique", Round: 1, Text: "falta contexto", Verdict: "revise"},
	}
	if err := c.record(ladderTailEntry{TurnIndex: 3, LoopID: "loop-1", Outcome: ladderOutcomeFinalRound, Rounds: 4, Tail: tail, Err: "ladder spent"}); err != nil {
		t.Fatalf("record() error = %v", err)
	}

	entry := decodeOnlyConflictsLogLine(t, &buf)
	if entry.TurnIndex != 3 {
		t.Errorf("turnIndex = %d, want 3", entry.TurnIndex)
	}
	if entry.LoopID != "loop-1" {
		t.Errorf("loopId = %q, want %q", entry.LoopID, "loop-1")
	}
	if entry.Outcome != ladderOutcomeFinalRound {
		t.Errorf("outcome = %q, want %q", entry.Outcome, ladderOutcomeFinalRound)
	}
	if entry.Rounds != 4 {
		t.Errorf("rounds = %d, want 4", entry.Rounds)
	}
	if len(entry.Tail) != 2 || entry.Tail[1].Verdict != "revise" {
		t.Errorf("tail = %+v, want the two steps passed in, verbatim", entry.Tail)
	}
	if entry.Err != "ladder spent" {
		t.Errorf("error = %q, want %q", entry.Err, "ladder spent")
	}
	if entry.Time == "" {
		t.Error("ts must be set")
	}
}

// TestConflictsLogger_record_approveOmitsError pins the package doc's
// Outcome vocabulary contract: a clean approve carries no error text (the
// json tag is `omitempty`), so a reader scanning for conflicts can filter on
// a non-empty "error" field alone.
func TestConflictsLogger_record_approveOmitsError(t *testing.T) {
	var buf bytes.Buffer
	c := &conflictsLogger{sink: &buf}
	if err := c.record(ladderTailEntry{Outcome: ladderOutcomeApprove, Rounds: 1}); err != nil {
		t.Fatalf("record() error = %v", err)
	}
	if strings.Contains(buf.String(), `"error"`) {
		t.Errorf("approve line must omit the error field entirely, got %s", buf.String())
	}
}

func TestConflictsLogger_record_appendsAcrossCalls(t *testing.T) {
	var buf bytes.Buffer
	c := &conflictsLogger{sink: &buf}
	if err := c.record(ladderTailEntry{Outcome: ladderOutcomeApprove, LoopID: "loop-1"}); err != nil {
		t.Fatalf("record() error = %v", err)
	}
	if err := c.record(ladderTailEntry{Outcome: ladderOutcomeAntiCycle, LoopID: "loop-2"}); err != nil {
		t.Fatalf("record() error = %v", err)
	}
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 appended lines, got %d: %q", len(lines), buf.String())
	}
}

func TestNewLoopID_isNonEmptyAndUnique(t *testing.T) {
	a := newLoopID()
	b := newLoopID()
	if a == "" || b == "" {
		t.Fatal("newLoopID must never return an empty string")
	}
	if a == b {
		t.Fatalf("two calls returned the same id: %q", a)
	}
	// 16 bytes, hex-encoded, so exactly 32 hex characters.
	if len(a) != 32 {
		t.Errorf("len(newLoopID()) = %d, want 32 (16 bytes hex-encoded)", len(a))
	}
}

// decodeOnlyConflictsLogLine asserts the sink captured exactly one
// ladderTailEntry line and decodes it, failing the test otherwise — the
// conflicts-log analog of session_tags_test.go's decodeOnlySessionTagsLogLine.
func decodeOnlyConflictsLogLine(t *testing.T, buf *bytes.Buffer) ladderTailEntry {
	t.Helper()
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 1 || lines[0] == "" {
		t.Fatalf("want exactly 1 conflicts-log line, got %d: %q", len(lines), buf.String())
	}
	var entry ladderTailEntry
	if err := json.Unmarshal([]byte(lines[0]), &entry); err != nil {
		t.Fatalf("line is not valid JSON: %v (%s)", err, lines[0])
	}
	return entry
}

// TestActiveSuperego_conflictsLogOnCleanApprove is the end-to-end wiring test:
// a gate-fired loop that the superego approves on round 1 must produce exactly
// one conflicts-log line with outcome "approve" — the clean-majority case the
// conflicts_log.go package doc calls out by name (a conflict-rate denominator
// is only honest if clean approves are logged too, not just the conflicted
// loops).
func TestActiveSuperego_conflictsLogOnCleanApprove(t *testing.T) {
	const draft = "rascunho original"
	eval := NewOutputEvaluator(OutputEvaluatorParams{
		Evaluator:    &scriptedEvaluator{steps: []scriptStep{gateFires()}},
		SystemPrompt: "avalie", Timeout: time.Second, Model: "haiku", GateMinInvestment: 4,
	})
	se := NewSuperego(SuperegoParams{
		Conversant:   &scriptedConversant{steps: []scriptStep{superegoVerdictStep("approve")}},
		SystemPrompt: "superego", Timeout: time.Second, Model: "opus",
	})
	var conflictsLog bytes.Buffer
	gov := NewActiveSuperego(ActiveSuperegoParams{Evaluator: eval, Superego: se, Model: "opus", ConflictsLogSink: &conflictsLog})

	text, archiveNote := gov.GovernOutput(context.Background(), 7, govMessages(), "", draft, (&redrafter{}).fn, (&finalizer{}).fn)
	if text != draft {
		t.Fatalf("approve on round 1 must deliver the original draft, got %q", text)
	}

	entry := decodeOnlyConflictsLogLine(t, &conflictsLog)
	if entry.Outcome != ladderOutcomeApprove {
		t.Errorf("outcome = %q, want %q", entry.Outcome, ladderOutcomeApprove)
	}
	if entry.TurnIndex != 7 {
		t.Errorf("turnIndex = %d, want 7", entry.TurnIndex)
	}
	if entry.LoopID == "" {
		t.Error("loopId must be set on a ladder loop")
	}
	if entry.Rounds != 1 {
		t.Errorf("rounds = %d, want 1", entry.Rounds)
	}
	if len(entry.Tail) != 2 || entry.Tail[0].Type != "draft" || entry.Tail[1].Type != "critique" {
		t.Errorf("tail = %+v, want [draft, critique]", entry.Tail)
	}
	// issue #5998: GovernOutput's archiveNote must name the SAME artifact this
	// test just decoded — the conflicts log's own loopId — not a second id for
	// a second file. A clean approve still ran a (1-round) ladder, so it is
	// archived exactly like any other outcome (conflicts_log.go's package doc:
	// every loop is logged, not just the conflicted ones).
	if archiveNote != "\n\n[superego archive #"+entry.LoopID+"]" {
		t.Errorf("archiveNote = %q, want it to reference the conflicts log's own loopId %q (reuse, not a second artifact)", archiveNote, entry.LoopID)
	}
}

// TestActiveSuperego_conflictsLogOffByDefault confirms the gating contract
// conflicts_log.go documents (mirroring session_tags.go's appendSessionTag):
// leaving ConflictsLogSink unset must never create any observable side
// effect — not even once the ladder actually runs — matching the facet's
// off-by-default convention.
func TestActiveSuperego_conflictsLogOffByDefault(t *testing.T) {
	const draft = "rascunho original"
	gov, _, _ := newTestGovernor(
		[]scriptStep{gateFires()},
		[]scriptStep{superegoVerdictStep("approve")},
		nil,
	)
	// gov.conflicts is always non-nil (NewActiveSuperego wraps a possibly-nil
	// sink), so record() must itself be the no-op — exercised directly here
	// instead of relying on "nothing crashed" as the only signal.
	if gov.conflicts == nil || gov.conflicts.sink != nil {
		t.Fatalf("expected a conflicts logger wrapping a nil sink, got %+v", gov.conflicts)
	}
	text, archiveNote := gov.GovernOutput(context.Background(), 0, govMessages(), "", draft, (&redrafter{}).fn, (&finalizer{}).fn)
	if text != draft {
		t.Fatalf("unexpected text = %q", text)
	}
	// issue #5998: with the conflicts log disabled, nothing was actually
	// archived, so archiveNote must stay "" — a reference naming a loop that
	// is not really in the log would be dangling and unrecoverable.
	if archiveNote != "" {
		t.Errorf("archiveNote = %q, want \"\" when the conflicts log is disabled", archiveNote)
	}
}

// TestActiveSuperego_panicInsideLadderStillLogsConflictsLine pins the fix for
// a gap found while adopting this orphaned change: GovernOutput's
// panic-recovery defer used to log the failure to the facet log but never to
// the conflicts log, so a loop that panicked mid-ladder (gate fired, ladder
// entered, then a round's superego call panicked) silently vanished from the
// conflicts log instead of landing a fail_open line — breaking the "every
// loop, including its failures, is one line" contract conflicts_log.go's
// package doc states explicitly. See superego_loop.go's GovernOutput:
// ladderEntered/lastRound are threaded into the panic-recovery defer so this
// case is covered too.
//
// This mirrors superego_loop_test.go's TestActiveSuperego_panicFailsOpen
// (same panicConversant, same gate-fires setup) but additionally wires a
// ConflictsLogSink to assert on the resulting line.
func TestActiveSuperego_panicInsideLadderStillLogsConflictsLine(t *testing.T) {
	const draft = "rascunho original"
	eval := NewOutputEvaluator(OutputEvaluatorParams{
		Evaluator:    &scriptedEvaluator{steps: []scriptStep{gateFires()}},
		SystemPrompt: "avalie", Timeout: time.Second, Model: "haiku", GateMinInvestment: 4,
	})
	se := NewSuperego(SuperegoParams{
		Conversant: panicConversant{}, SystemPrompt: "superego", Timeout: time.Second, Model: "opus",
	})
	var conflictsLog bytes.Buffer
	gov := NewActiveSuperego(ActiveSuperegoParams{Evaluator: eval, Superego: se, Model: "opus", ConflictsLogSink: &conflictsLog})

	text, archiveNote := gov.GovernOutput(context.Background(), 0, govMessages(), "", draft, (&redrafter{}).fn, (&finalizer{}).fn)
	if text != draft {
		t.Fatalf("panic must still fail open to the original draft, got %q", text)
	}

	entry := decodeOnlyConflictsLogLine(t, &conflictsLog)
	if entry.Outcome != ladderOutcomeFailOpen {
		t.Errorf("outcome = %q, want %q", entry.Outcome, ladderOutcomeFailOpen)
	}
	if entry.Rounds != 1 {
		t.Errorf("rounds = %d, want 1 (the panic happened inside round 1's review call)", entry.Rounds)
	}
	if !strings.Contains(entry.Err, "panic: boom") {
		t.Errorf("error = %q, want it to contain %q", entry.Err, "panic: boom")
	}
	// issue #5998: even a panic mid-ladder, once the ladder was entered, still
	// lands a real conflicts-log line (asserted above), so it must still be
	// reported as archived.
	if archiveNote != "\n\n[superego archive #"+entry.LoopID+"]" {
		t.Errorf("archiveNote = %q, want it to reference the conflicts log's own loopId %q", archiveNote, entry.LoopID)
	}
}

// TestActiveSuperego_panicBeforeLadderEntryLogsNoConflictsLine confirms the
// other half of the same fix is scoped correctly: a panic BEFORE the gate
// fires (so no ladder was ever entered) must still log nothing to the
// conflicts log, exactly like the pre-existing evaluator-failure branch
// (`!ok` in GovernOutput) that conflicts_log.go's package doc excludes by
// design — the denominator is ladder loops, not every evaluated message.
type panicEvaluatorFacet struct{}

func (panicEvaluatorFacet) Evaluate(_ context.Context, _ string, _ string) (Evaluation, error) {
	panic("boom before gate")
}

func TestActiveSuperego_panicBeforeLadderEntryLogsNoConflictsLine(t *testing.T) {
	const draft = "rascunho original"
	eval := NewOutputEvaluator(OutputEvaluatorParams{
		Evaluator: panicEvaluatorFacet{}, SystemPrompt: "avalie", Timeout: time.Second,
		Model: "haiku", GateMinInvestment: 4,
	})
	se := NewSuperego(SuperegoParams{
		Conversant:   &scriptedConversant{steps: []scriptStep{superegoVerdictStep("approve")}},
		SystemPrompt: "superego", Timeout: time.Second, Model: "opus",
	})
	var conflictsLog bytes.Buffer
	gov := NewActiveSuperego(ActiveSuperegoParams{Evaluator: eval, Superego: se, Model: "opus", ConflictsLogSink: &conflictsLog})

	text, archiveNote := gov.GovernOutput(context.Background(), 0, govMessages(), "", draft, (&redrafter{}).fn, (&finalizer{}).fn)
	if text != draft {
		t.Fatalf("panic must still fail open to the original draft, got %q", text)
	}
	if conflictsLog.Len() != 0 {
		t.Errorf("expected no conflicts-log line for a panic before the ladder was entered, got %q", conflictsLog.String())
	}
	// issue #5998: nothing was archived (no ladder, no tail beyond the single
	// draft already returned as text), so archiveNote must stay "".
	if archiveNote != "" {
		t.Errorf("archiveNote = %q, want \"\" when no ladder was ever entered", archiveNote)
	}
}

// TestActiveSuperego_archiveNoteEmptyWhenConflictsWriteErrors covers the
// remaining negative path: the sink is ENABLED (non-nil) but record() itself
// fails (e.g. a full disk). The ladder still fails open to the draft, and
// archiveNote must stay "" — a note must never name a loop that is not
// really sitting in the log, and conflictsLogger.record's own error here
// means it is not.
func TestActiveSuperego_archiveNoteEmptyWhenConflictsWriteErrors(t *testing.T) {
	const draft = "rascunho original"
	eval := NewOutputEvaluator(OutputEvaluatorParams{
		Evaluator:    &scriptedEvaluator{steps: []scriptStep{gateFires()}},
		SystemPrompt: "avalie", Timeout: time.Second, Model: "haiku", GateMinInvestment: 4,
	})
	se := NewSuperego(SuperegoParams{
		Conversant:   &scriptedConversant{steps: []scriptStep{superegoVerdictStep("approve")}},
		SystemPrompt: "superego", Timeout: time.Second, Model: "opus",
	})
	gov := NewActiveSuperego(ActiveSuperegoParams{
		Evaluator: eval, Superego: se, Model: "opus",
		ConflictsLogSink: errWriter{err: errors.New("disk full")},
	})

	text, archiveNote := gov.GovernOutput(context.Background(), 0, govMessages(), "", draft, (&redrafter{}).fn, (&finalizer{}).fn)
	if text != draft {
		t.Fatalf("approve must still deliver the draft even though the conflicts write failed, got %q", text)
	}
	if archiveNote != "" {
		t.Errorf("archiveNote = %q, want \"\" when the conflicts log write itself errors (sink enabled, write failed)", archiveNote)
	}
}
