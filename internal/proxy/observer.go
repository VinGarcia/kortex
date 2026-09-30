package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/vingarcia/kortex/internal/history"
	"github.com/vingarcia/kortex/internal/protocol"
)

// observer is the F2a read-only interception layer: it takes every parsed
// wire line, classifies it into canonical-history intents, and (when an
// event log path is configured) appends a structured event log next to the
// raw traffic log. It never writes to the protocol's stdout/stderr, and any
// internal failure — parse error, panic — is swallowed so the
// byte-identical passthrough is never at risk.
type observer struct {
	mu       sync.Mutex
	recorder *history.Recorder
	file     *os.File // nil disables structured logging, never observation
	// turnEvaluator, when non-nil, is notified of each newly completed
	// turn. The notification happens under the observer mutex; the
	// implementation returns promptly by contract (see proxy.TurnEvaluator).
	turnEvaluator TurnEvaluator
}

func newObserver(eventLogPath string, stderr io.Writer, turnEvaluator TurnEvaluator) *observer {
	o := &observer{recorder: history.NewRecorder(), turnEvaluator: turnEvaluator}
	if eventLogPath == "" {
		return o
	}
	file, err := os.OpenFile(eventLogPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		fmt.Fprintf(stderr, "kortex: structured log disabled: %v\n", err)
		return o
	}
	o.file = file
	return o
}

// entry is one structured log line. Snapshot is only present on the entry
// for a terminal result event, closing the turn.
type entry struct {
	Time      string            `json:"ts"`
	Direction string            `json:"dir"`
	Type      string            `json:"type"`
	Subtype   string            `json:"subtype,omitempty"`
	Stream    string            `json:"streamEvent,omitempty"`
	ParseErr  string            `json:"parseError,omitempty"`
	Snapshot  *history.Snapshot `json:"history,omitempty"`
}

// turnNotice is one pending turn-completion notification, carried out of
// the locked section so the TurnEvaluator is never called under the mutex.
type turnNotice struct {
	index int
	text  string
}

// Observe folds one parsed wire event into the history and the structured
// log. Failures never propagate to the caller.
func (o *observer) Observe(direction protocol.Direction, event protocol.Event) {
	defer func() {
		if r := recover(); r != nil {
			o.logInternalError(fmt.Sprintf("observer panic: %v", r))
		}
	}()
	// The notification fires after the mutex is released: the evaluator
	// contract says "return promptly", but the pump's safety should not
	// depend on an implementation honoring it while holding the lock the
	// other pump goroutine needs.
	if notice := o.observeLocked(direction, event); notice != nil {
		o.turnEvaluator.EvaluateCompletedTurn(notice.index, notice.text)
	}
}

// observeLocked does the mutex-protected part of Observe: history
// recording and structured logging. It returns the pending turn-completion
// notification, if this event produced one.
func (o *observer) observeLocked(direction protocol.Direction, event protocol.Event) *turnNotice {
	o.mu.Lock()
	defer o.mu.Unlock()
	notice := o.record(direction, event)
	if o.file == nil {
		return notice
	}
	e := entry{
		Time:      time.Now().Format(time.RFC3339Nano),
		Direction: string(direction),
		Type:      event.Type,
		Subtype:   event.Subtype,
		Stream:    event.StreamEventType,
	}
	if event.ParseErr != nil {
		e.ParseErr = event.ParseErr.Error()
	}
	if event.Type == protocol.TypeResult {
		snapshot := compactSnapshot(o.recorder.Snapshot())
		e.Snapshot = &snapshot
	}
	o.writeEntry(e)
	return notice
}

// record classifies a wire event into history intents, returning the
// turn-completion notification to deliver (nil for none). Transport facts
// live here on purpose: which pipe carried the line and
// --replay-user-messages echoes are proxy knowledge, not history knowledge.
func (o *observer) record(direction protocol.Direction, event protocol.Event) *turnNotice {
	o.recorder.SetSessionID(event.SessionID)
	switch event.Type {
	case protocol.TypeUser:
		if event.Message == nil {
			return nil
		}
		// Tool results travel as user messages; they belong to the open
		// turn's tool calls no matter which direction carried them.
		for _, block := range event.Message.Content {
			if block.Type == "tool_result" {
				o.recorder.AttachToolResult(block.ToolUseID, block.ResultText(), block.IsError)
			}
		}
		// Only a fresh prompt on stdin opens a turn: stdout user events are
		// either --replay-user-messages echoes or tool-result carriers.
		if direction == protocol.ToBackend && !event.IsReplay {
			o.recorder.RecordUserPrompt(event.Message.TextContent())
		}
	case protocol.TypeAssistant:
		o.recorder.RecordAssistantMessage(event.Message)
	case protocol.TypeResult:
		// Only a result that closed a turn NOW produces a notification: a
		// stray duplicate result must not re-evaluate (and re-bill) the
		// previous turn.
		if !o.recorder.CompleteTurn(event.Result) {
			return nil
		}
		if o.turnEvaluator == nil {
			return nil
		}
		if index, turn, ok := o.recorder.LastCompletedTurn(); ok {
			return &turnNotice{index: index, text: turn.AssistantText}
		}
	}
	return nil
}

func (o *observer) writeEntry(e entry) {
	data, err := json.Marshal(e)
	if err != nil {
		return
	}
	o.file.Write(append(data, '\n'))
}

// logInternalError records an observer failure in the structured log; there
// is nowhere else safe to report it (protocol stderr belongs to the child).
func (o *observer) logInternalError(msg string) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.file == nil {
		return
	}
	o.writeEntry(entry{
		Time:     time.Now().Format(time.RFC3339Nano),
		Type:     "kortex_internal_error",
		ParseErr: msg,
	})
}

func (o *observer) Close() {
	// The mutex keeps the un-joined stdin pump from writing to a closed
	// file: after Close it sees nil and observation degrades to a no-op.
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.file != nil {
		o.file.Close()
		o.file = nil
	}
}

// snapshotFieldLimit caps each text field in the logged snapshot; the
// in-memory history keeps full text, the log only needs enough to recognize
// the turn.
const snapshotFieldLimit = 300

func compactSnapshot(s history.Snapshot) history.Snapshot {
	for i := range s.Turns {
		turn := &s.Turns[i]
		turn.UserText = truncate(turn.UserText)
		turn.AssistantText = truncate(turn.AssistantText)
		for j := range turn.ToolCalls {
			turn.ToolCalls[j].Input = truncate(turn.ToolCalls[j].Input)
			turn.ToolCalls[j].Result = truncate(turn.ToolCalls[j].Result)
		}
	}
	return s
}

func truncate(s string) string {
	if len(s) <= snapshotFieldLimit {
		return s
	}
	cut := snapshotFieldLimit
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + "…"
}
