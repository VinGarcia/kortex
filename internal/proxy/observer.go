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

// observer is the F2a read-only interception layer: it parses every wire
// line, folds it into the canonical session history, and (when KORTEX_LOG is
// set) appends a structured event log next to the raw traffic log. It never
// writes to the protocol's stdout/stderr, and any internal failure — parse
// error, panic — is swallowed so the byte-identical passthrough is never at
// risk.
type observer struct {
	mu       sync.Mutex
	recorder *history.Recorder
	file     *os.File // nil disables structured logging, never observation
}

// structuredLogPath derives the structured event log path from the raw
// traffic log path (KORTEX_LOG).
func structuredLogPath(rawLogPath string) string {
	return rawLogPath + ".events"
}

func newObserver(rawLogPath string, stderr io.Writer) *observer {
	o := &observer{recorder: history.NewRecorder()}
	if rawLogPath == "" {
		return o
	}
	file, err := os.OpenFile(structuredLogPath(rawLogPath), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
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

// Observe processes one wire line. Failures never propagate to the caller.
func (o *observer) Observe(direction protocol.Direction, line []byte) {
	defer func() {
		if r := recover(); r != nil {
			o.logInternalError(fmt.Sprintf("observer panic: %v", r))
		}
	}()
	event := protocol.Parse(line)

	o.mu.Lock()
	defer o.mu.Unlock()
	o.recorder.Observe(direction, event)
	if o.file == nil {
		return
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
	if o.file != nil {
		o.file.Close()
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
