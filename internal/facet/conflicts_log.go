package facet

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"
)

// This file is Part 1, sub-item 3 of the observability feature (issue #5368):
// the conflicts log. Every ActiveSuperego.GovernOutput ladder loop ends in
// exactly one terminal outcome (a clean approve, or one of the fail-open
// endings: fail_open, final_round, anti_cycle — see GovernOutput's Decision
// values). This file persists the COMPLETE ORDERED TAIL of that loop — draft,
// critique1, meta-message (when the DebateMetaMessage feature is on),
// draft2, critique2, ... — as one JSONL line, to a file separate from the
// main facets log (a "<KORTEX_LOG>.conflicts" sibling of it).
//
// Every loop is logged here, not just the conflicted ones: a clean approve is
// outcome "approve". Part 2 (stats aggregation, out of scope for this file)
// needs the TOTAL loop count as the denominator to compute a conflict rate,
// and deriving that denominator from this log is only correct if this log
// never silently drops the clean majority.
//
// Gating mirrors session_tags.go's appendSessionTag: a nil sink (the
// composition root's ConflictsLogSink is only wired when KORTEX_LOG is set
// and superego mode is "active") makes conflictsLogger.record a no-op — no
// file, no line, ever. That early return is the documented off-by-default
// gate, not a failure, so it is unlogged. Every OTHER failure (marshal,
// short write) is returned as an error, which the caller (GovernOutput)
// records through the existing facetLogger — never silently dropped
// (u-never-fail-silently): this file's own sink failing must never leave an
// operator believing the conflicts log is healthy when it is not.
//
// Writing to this log, like every facet observability path, must never
// change the ladder's own control flow or delay/alter the delivered text —
// GovernOutput calls conflictsLogger.record AFTER it has already decided the
// text to deliver, purely as a side observation.
//
// Issue #5998 adds one more reader of this same log: once a loop's line has
// actually landed here, GovernOutput can hand the session layer a short
// retrievalHint (below) pointing back at it by loopId, so the core gets a
// reference to its own archived conversation tail. This is the ONLY place
// that invariant is stated: no second tail file is ever created for it, and
// the session layer never sees this file's schema — retrievalHint renders
// the complete, ready-to-persist note itself.

// conflictsLogFacet names this producer's callLog entries distinctly from
// activeSuperegoFacet, when a conflicts-log write failure itself is routed
// through the shared facetLogger (see conflictsLogWriteFailed in
// superego_loop.go).
const conflictsLogFacet = "conflicts_log"

// Outcome vocabulary for ladderTailEntry.Outcome. Deliberately its OWN
// vocabulary, distinct from callLog.Decision ("deliver"/"revise"/
// "fail_open"/"final_round"): callLog's "deliver" conflates a clean approve
// with the anti-cycle guard's early delivery, which a downstream reader
// computing a conflict rate must NOT treat as the same thing. Anti-cycle
// delivers a draft the superego never approved — that is a conflict by this
// log's definition, not a clean end.
const (
	ladderOutcomeApprove    = "approve"
	ladderOutcomeFailOpen   = "fail_open"
	ladderOutcomeFinalRound = "final_round"
	ladderOutcomeAntiCycle  = "anti_cycle"
)

// ladderStep is one entry of the ordered tail persisted for a loop: the
// original draft, each round's superego critique, the core's optional debate
// meta-message attached to the draft/redraft that critique judged, and the
// following redraft. Type is one of "draft", "meta", "critique", "final".
// Round is the ladder round the step belongs to (the initial draft and its
// meta, if any, carry the round they were FIRST reviewed under; the final
// core-authored message, when present, carries no round).
type ladderStep struct {
	Type    string `json:"type"`
	Round   int    `json:"round,omitempty"`
	Text    string `json:"text,omitempty"`
	Verdict string `json:"verdict,omitempty"`
}

// ladderTailEntry is one JSONL line of the conflicts log: one line per
// GovernOutput ladder loop, whatever its outcome (see the package doc above
// for why clean approves are included too).
type ladderTailEntry struct {
	Time      string       `json:"ts"`
	TurnIndex int          `json:"turnIndex"`
	LoopID    string       `json:"loopId"`
	Outcome   string       `json:"outcome"`
	Rounds    int          `json:"rounds"`
	Tail      []ladderStep `json:"tail"`
	// Err carries the fail-open reason (evaluator/superego/redraft/finalize
	// error, or the anti-cycle note) when Outcome is not a clean approve.
	// Empty on approve.
	Err string `json:"error,omitempty"`
}

// conflictsLogger appends one JSON object per ladder loop to the sink. A nil
// sink (ConflictsLogSink left unset — the documented off-by-default gate)
// makes record a no-op; every other failure is RETURNED, not swallowed,
// unlike facetLogger.record's pre-existing silent-drop behavior (see
// annotator.go) — this new writer must not repeat that gap.
type conflictsLogger struct {
	mu   sync.Mutex
	sink io.Writer
}

func (c *conflictsLogger) record(entry ladderTailEntry) error {
	if c == nil || c.sink == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	entry.Time = time.Now().Format(time.RFC3339Nano)
	data, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("marshal conflicts log entry: %w", err)
	}
	line := append(data, '\n')
	n, err := c.sink.Write(line)
	if err != nil {
		return fmt.Errorf("write conflicts log entry: %w", err)
	}
	if n != len(line) {
		return fmt.Errorf("write conflicts log entry: wrote %d of %d bytes", n, len(line))
	}
	return nil
}

// enabled reports whether this logger will actually record a loop: false
// when the sink is nil (the documented off-by-default gate — see record's
// own doc comment), in which case record is always a no-op regardless of
// what it is asked to persist.
func (c *conflictsLogger) enabled() bool {
	return c != nil && c.sink != nil
}

// retrievalHint renders the complete note GovernOutput hands back as
// archiveNote for a loop whose line has just landed in this log under id
// (the caller, recordConflicts in superego_loop.go, only reaches this after
// record() ran with a non-nil sink and returned no error — never for a loop
// that is not really sitting here). The note is deliberately just a short
// reference: the mechanism (what the archive is, where it lives) is explained
// once in the system prompt (session.ArchiveSystemPromptNote), and the note is
// persisted in history only, never delivered to the user.
func (c *conflictsLogger) retrievalHint(id string) string {
	return fmt.Sprintf("\n\n[superego archive #%s]", id)
}

// newLoopID mints a fresh per-loop identifier: 16 bytes of crypto/rand,
// hex-encoded. Only intra-logfile uniqueness and equality matter here (it
// groups rows for one loop; it is never parsed as an RFC4122 UUID or exposed
// to a client), so a plain random hex string suffices — unlike main.go's
// newUUIDv4, which does need a true UUID (it names the kortex session). It
// panics on a crypto/rand read failure exactly like newUUIDv4 does: this is
// called from inside ActiveSuperego.GovernOutput, which already wraps its
// whole body in a recover-to-fail-open defer, so a panic here still degrades
// to delivering the draft rather than crashing the process.
func newLoopID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(fmt.Sprintf("facet: reading random for loop id: %v", err))
	}
	return fmt.Sprintf("%x", b)
}
