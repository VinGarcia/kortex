package session

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/protocol"
	"github.com/vingarcia/kortex/internal/tools"
)

// seqUUID returns a deterministic uuid generator so emitted events are
// comparable across a test run.
func seqUUID() func() string {
	n := 0
	return func() string {
		n++
		return fmt.Sprintf("uuid-%d", n)
	}
}

// textResponder is an Anthropic test server that replies to every Messages API
// call with a fixed end_turn text, recording the system prompt and the message
// count it last received.
func textResponder(t *testing.T, reply string) (*httptest.Server, *recorded) {
	t.Helper()
	rec := &recorded{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			System   string              `json:"system"`
			Messages []anthropic.Message `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		rec.calls++
		rec.lastSystem = body.System
		rec.lastMessageCount = len(body.Messages)
		if n := len(body.Messages); n > 0 {
			if text, ok := body.Messages[n-1].Content.(string); ok {
				rec.lastUserText = text
			}
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"content":[{"type":"text","text":%q}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`, reply)
	}))
	t.Cleanup(server.Close)
	return server, rec
}

type recorded struct {
	calls            int
	lastSystem       string
	lastMessageCount int
	lastUserText     string
}

func initializeLine(prompt string) string {
	return fmt.Sprintf(`{"type":"control_request","request_id":"req-1","request":{"subtype":"initialize","appendSystemPrompt":%q}}`, prompt)
}

func userLine(sessionID string, text string) string {
	return fmt.Sprintf(`{"type":"user","session_id":%q,"message":{"role":"user","content":%q}}`, sessionID, text)
}

// parseLines splits captured stdout into parsed protocol events, skipping the
// control_response (which Parse surfaces as TypeControlResponse).
func parseLines(t *testing.T, out string) []protocol.Event {
	t.Helper()
	var events []protocol.Event
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		if line == "" {
			continue
		}
		events = append(events, protocol.Parse([]byte(line)))
	}
	return events
}

func TestRun_InitializeHandshakeAndSingleTurn(t *testing.T) {
	server, rec := textResponder(t, "blue")
	client := anthropic.NewClient("secret", server.URL, time.Second)
	store := NewStore(t.TempDir())

	in := strings.Join([]string{
		initializeLine("You are Sylphie."),
		userLine("sess-1", "what color?"),
	}, "\n") + "\n"
	var out strings.Builder

	err := Run(context.Background(), Config{
		Stdin:      strings.NewReader(in),
		Stdout:     &out,
		Client:     client,
		Dispatcher: tools.NewDispatcher(),
		Store:      store,
		SessionID:  "sess-1",
		Model:      "claude-haiku-4-5-20251001",
		MaxTokens:  1024,
		NewUUID:    seqUUID(),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// The system prompt must arrive from initialize, not argv — landing after
	// the Claude Code preamble the anthropic client prepends for subscription
	// tokens (see anthropic.claudeCodePreamble).
	wantSystem := "You are Claude Code, Anthropic's official CLI for Claude.\n\nYou are Sylphie."
	if rec.lastSystem != wantSystem {
		t.Errorf("system prompt = %q, want %q", rec.lastSystem, wantSystem)
	}

	events := parseLines(t, out.String())
	// Expect: control_response, system/init, assistant, result.
	var sawControlResponse, sawResult bool
	for _, ev := range events {
		switch ev.Type {
		case protocol.TypeControlResponse:
			sawControlResponse = true
			if ev.Subtype != "success" || ev.RequestID != "req-1" {
				t.Errorf("control_response = %+v, want success for req-1", ev)
			}
		case protocol.TypeResult:
			sawResult = true
			if ev.Result.IsError || ev.Result.Text != "blue" {
				t.Errorf("result = %+v, want success with text blue", ev.Result)
			}
		}
	}
	if !sawControlResponse || !sawResult {
		t.Errorf("missing events: control_response=%v result=%v (out=%q)", sawControlResponse, sawResult, out.String())
	}
}

func TestRun_MultiTurnAccumulatesHistory(t *testing.T) {
	server, rec := textResponder(t, "ok")
	client := anthropic.NewClient("secret", server.URL, time.Second)
	store := NewStore(t.TempDir())

	in := strings.Join([]string{
		initializeLine("sys"),
		userLine("sess-1", "first"),
		userLine("sess-1", "second"),
	}, "\n") + "\n"
	var out strings.Builder

	err := Run(context.Background(), Config{
		Stdin: strings.NewReader(in), Stdout: &out, Client: client,
		Dispatcher: tools.NewDispatcher(), Store: store, SessionID: "sess-1",
		Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if rec.calls != 2 {
		t.Fatalf("API calls = %d, want 2 (one per user turn)", rec.calls)
	}
	// The second turn must carry history: its own user msg plus the prior
	// user+assistant = 3 messages.
	if rec.lastMessageCount != 3 {
		t.Errorf("second turn message count = %d, want 3 (prior user+assistant + new user)", rec.lastMessageCount)
	}
	// Two user turns => two terminal results, but exactly ONE system/init for
	// the whole session (the real claude-cli announces the session once at
	// startup, not per turn).
	results, inits := 0, 0
	for _, ev := range parseLines(t, out.String()) {
		switch {
		case ev.Type == protocol.TypeResult:
			results++
		case ev.Type == protocol.TypeSystem && ev.Subtype == "init":
			inits++
		}
	}
	if results != 2 {
		t.Errorf("terminal results = %d, want 2", results)
	}
	if inits != 1 {
		t.Errorf("system/init count = %d, want exactly 1 for a multi-turn session", inits)
	}
}

func TestRun_CrossProcessResume(t *testing.T) {
	server, rec := textResponder(t, "ok")
	client := anthropic.NewClient("secret", server.URL, time.Second)
	dir := filepath.Join(t.TempDir(), "sessions")

	run := func(userText string) {
		var out strings.Builder
		err := Run(context.Background(), Config{
			Stdin:  strings.NewReader(initializeLine("sys") + "\n" + userLine("sess-1", userText) + "\n"),
			Stdout: &out, Client: client, Dispatcher: tools.NewDispatcher(),
			Store: NewStore(dir), SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		})
		if err != nil {
			t.Fatalf("Run(%q): %v", userText, err)
		}
	}

	// First process persists the turn; a second, fresh process (new Store) must
	// load that history and send it ahead of the new user message.
	run("first")
	run("second")

	if rec.lastMessageCount != 3 {
		t.Errorf("resumed turn message count = %d, want 3 (persisted user+assistant + new user)", rec.lastMessageCount)
	}
}

func TestRun_TurnErrorEmitsResultAndContinues(t *testing.T) {
	// A server that 500s fails the Messages call, so RunLoop errors; the session
	// must still emit a terminal error result and keep reading the next turn.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer server.Close()
	client := anthropic.NewClient("secret", server.URL, time.Second)

	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin:  strings.NewReader(initializeLine("sys") + "\n" + userLine("s", "hi") + "\n"),
		Stdout: &out, Client: client, Dispatcher: tools.NewDispatcher(),
		Store: NewStore(t.TempDir()), SessionID: "s", Model: "m", MaxTokens: 10, NewUUID: seqUUID(),
	})
	if err != nil {
		t.Fatalf("Run must not propagate a turn error: %v", err)
	}
	var errResult *protocol.Result
	for _, ev := range parseLines(t, out.String()) {
		if ev.Type == protocol.TypeResult {
			errResult = ev.Result
		}
	}
	if errResult == nil || !errResult.IsError {
		t.Errorf("want a terminal error result after a failed turn, got %+v", errResult)
	}
}
