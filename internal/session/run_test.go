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
	"github.com/vingarcia/kortex/internal/history"
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
			System []struct {
				Text string `json:"text"`
			} `json:"system"`
			Messages []anthropic.Message `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		rec.calls++
		parts := make([]string, len(body.System))
		for i, b := range body.System {
			parts[i] = b.Text
		}
		rec.lastSystem = strings.Join(parts, "\n\n")
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

// TestRun_EmptyFinalTurnEmitsErrorNotEmpty reproduces the live 2026-10-04
// failure end-to-end: the core model returns a final turn with only a thinking
// block and no text (stop_reason "max_tokens"). The session must emit a terminal
// is_error result naming the cause — never an empty assistant turn, which
// OpenClaw rejects as "CLI backend returned an empty response".
func TestRun_EmptyFinalTurnEmitsErrorNotEmpty(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"content":[{"type":"thinking","thinking":"","signature":"sig"}],"stop_reason":"max_tokens","usage":{"input_tokens":1,"output_tokens":8192}}`))
	}))
	defer server.Close()
	client := anthropic.NewClient("secret", server.URL, time.Second)

	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin:  strings.NewReader(initializeLine("sys") + "\n" + userLine("s", "think hard") + "\n"),
		Stdout: &out, Client: client, Dispatcher: tools.NewDispatcher(),
		Store: NewStore(t.TempDir()), SessionID: "s", Model: "claude-fable-5", MaxTokens: 8192, NewUUID: seqUUID(),
	})
	if err != nil {
		t.Fatalf("Run must not propagate the empty-final error: %v", err)
	}

	var result *protocol.Result
	for _, ev := range parseLines(t, out.String()) {
		switch ev.Type {
		case protocol.TypeResult:
			result = ev.Result
		case protocol.TypeAssistant:
			// The guard fires before any final assistant emission, so no empty
			// assistant turn must ever reach the wire.
			t.Errorf("an assistant event was emitted for an empty final turn: %q", out.String())
		}
	}
	if result == nil || !result.IsError {
		t.Fatalf("want a terminal error result for an empty final turn, got %+v", result)
	}
	if !strings.Contains(result.Text, "max_tokens") {
		t.Errorf("error result should name the stop_reason, got %q", result.Text)
	}
}

// TestToolCallsFromMessages_extractsThisTurnOnly pins the #4749 extraction:
// toolCallsFromMessages must open a ToolCall per tool_use block (preserving
// order), attach each tool_result to its call by id (success and is_error), and
// — because runTurn passes res.Messages[len(msgs):] — fold in only the slice it
// is given, never a prior turn's tool traffic.
func TestToolCallsFromMessages_extractsThisTurnOnly(t *testing.T) {
	messages := []anthropic.Message{
		// --- a PRIOR turn's exchange (index 0-1): must stay out of the result
		// when runTurn slices it off with res.Messages[len(msgs):]. ---
		{Role: "assistant", Content: []json.RawMessage{
			json.RawMessage(`{"type":"tool_use","id":"prior-1","name":"OldTool","input":{"x":1}}`),
		}},
		{Role: "user", Content: []anthropic.ToolResultBlock{
			anthropic.NewToolResultBlock("prior-1", "old result", false),
		}},
		// --- THIS turn (index 2 onward): the new user message, an assistant turn
		// mixing text with two tool_use blocks, then their two results. ---
		{Role: "user", Content: "check the PR"},
		{Role: "assistant", Content: []json.RawMessage{
			json.RawMessage(`{"type":"text","text":"let me look"}`),
			json.RawMessage(`{"type":"tool_use","id":"call-1","name":"Grep","input":{"pattern":"approved"}}`),
			json.RawMessage(`{"type":"tool_use","id":"call-2","name":"Bash","input":{"cmd":"ls"}}`),
		}},
		{Role: "user", Content: []anthropic.ToolResultBlock{
			anthropic.NewToolResultBlock("call-1", "match found", false),
			anthropic.NewToolResultBlock("call-2", "boom", true),
		}},
		{Role: "assistant", Content: "done"},
	}

	// priorLen mirrors runTurn's len(msgs): scan only this turn's appended slice.
	const priorLen = 2
	got := toolCallsFromMessages(messages[priorLen:])

	if len(got) != 2 {
		t.Fatalf("got %d tool calls, want 2 (prior turn must be excluded): %+v", len(got), got)
	}
	// Order follows tool_use order: Grep before Bash.
	if got[0].ID != "call-1" || got[0].Name != "Grep" {
		t.Errorf("call 0 = %+v, want id call-1 name Grep", got[0])
	}
	if got[0].Input != `{"pattern":"approved"}` {
		t.Errorf("call 0 input = %q, want the verbatim tool_use input JSON", got[0].Input)
	}
	if got[0].Result != "match found" || got[0].IsError {
		t.Errorf("call 0 result = %q isError=%v, want \"match found\" / false", got[0].Result, got[0].IsError)
	}
	if got[1].ID != "call-2" || got[1].Name != "Bash" {
		t.Errorf("call 1 = %+v, want id call-2 name Bash", got[1])
	}
	if got[1].Result != "boom" || !got[1].IsError {
		t.Errorf("call 1 result = %q isError=%v, want \"boom\" / true (is_error propagated)", got[1].Result, got[1].IsError)
	}

	// Guard the slicing claim directly: the full slice DOES fold the prior call
	// in, so it is the len(msgs) offset — not toolCallsFromMessages — that scopes
	// a Turn to its own tool traffic.
	if full := toolCallsFromMessages(messages); len(full) != 3 {
		t.Errorf("full-slice extraction got %d calls, want 3 (prior-1 + call-1 + call-2)", len(full))
	}
}

// captureEvaluator is a stub TurnEvaluator that records the completed
// history.Turn the session hands it, letting a test inspect the Turn built
// end-to-end (ToolCalls included) without exposing runTurn.
type captureEvaluator struct {
	turns []history.Turn
}

func (c *captureEvaluator) EvaluateCompletedTurn(turnIndex int, snapshot history.Snapshot) {
	c.turns = append(c.turns, snapshot.Turns[turnIndex])
}

// TestRun_ToolCallsPopulatedEndToEnd drives the REAL RunLoop/runTurn path with a
// stubbed Anthropic server that returns a tool_use block (executed by the real
// Dispatcher) and asserts the history.Turn the session builds carries the
// extracted ToolCalls end-to-end. This guards a silent-regression gap:
// toolCallsFromMessages type-switches on the concrete Content types RunLoop
// produces — []json.RawMessage for the assistant tool_use block and
// []anthropic.ToolResultBlock for the result. If anthropic.RawContent / RunLoop
// ever change those concrete types, the type switch falls through, turn.ToolCalls
// goes silently empty, and the superego regresses to emitting FALSE provenance
// positives with no failing test — the opposite of this feature's few-false-
// positives requirement. By exercising the live types (not hand-built ones, as
// TestToolCallsFromMessages_extractsThisTurnOnly does), a future type change
// breaks THIS test instead of silently emptying ToolCalls.
func TestRun_ToolCallsPopulatedEndToEnd(t *testing.T) {
	call := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call++
		w.Header().Set("Content-Type", "application/json")
		if call == 1 {
			// First turn: a tool_use the real Dispatcher executes (bash echo),
			// producing a genuine tool_result the loop feeds back.
			w.Write([]byte(`{
				"content": [{"type":"tool_use","id":"toolu_1","name":"bash","input":{"command":"echo provenance-ok"}}],
				"stop_reason": "tool_use",
				"usage": {"input_tokens": 10, "output_tokens": 5}
			}`))
			return
		}
		// Second turn: end_turn with text closes the round.
		w.Write([]byte(`{
			"content": [{"type":"text","text":"ran it"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 12, "output_tokens": 6}
		}`))
	}))
	defer server.Close()

	client := anthropic.NewClient("secret", server.URL, time.Second)
	capture := &captureEvaluator{}
	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin:         strings.NewReader(initializeLine("sys") + "\n" + userLine("s", "run echo") + "\n"),
		Stdout:        &out,
		Client:        client,
		Dispatcher:    tools.NewDispatcher(),
		Store:         NewStore(t.TempDir()),
		SessionID:     "s",
		Model:         "m",
		MaxTokens:     1024,
		NewUUID:       seqUUID(),
		TurnEvaluator: capture,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if call != 2 {
		t.Fatalf("API calls = %d, want 2 (tool_use round trip)", call)
	}
	if len(capture.turns) != 1 {
		t.Fatalf("captured %d completed turns, want 1", len(capture.turns))
	}

	calls := capture.turns[0].ToolCalls
	if len(calls) != 1 {
		t.Fatalf("turn.ToolCalls = %+v, want exactly 1 extracted from the live RunLoop types", calls)
	}
	if calls[0].ID != "toolu_1" || calls[0].Name != "bash" {
		t.Errorf("tool call = %+v, want id toolu_1 name bash", calls[0])
	}
	if calls[0].Input != `{"command":"echo provenance-ok"}` {
		t.Errorf("tool call input = %q, want the verbatim tool_use input JSON", calls[0].Input)
	}
	// The real bash output must be attached to the call (result pairing by id over
	// the live []anthropic.ToolResultBlock turn), and it is not an error.
	if !strings.Contains(calls[0].Result, "provenance-ok") || calls[0].IsError {
		t.Errorf("tool call result = %q isError=%v, want the real bash output attached / false", calls[0].Result, calls[0].IsError)
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
