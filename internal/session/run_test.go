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
			if text, ok := decodedMessageText(body.Messages[n-1].Content); ok {
				rec.lastUserText = text
				rec.userTexts = append(rec.userTexts, text)
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
	// userTexts records the last user content of every call, in order, so a test
	// can assert what each turn forwarded to the core (not just the final turn).
	userTexts []string
}

// decodedMessageText returns the plain text of a Message.Content value as
// decoded generically by encoding/json: a bare string, or — since the
// anthropic client's message-history cache marking rewrites a string's
// last-message content into the array-of-blocks form required to carry a
// cache_control marker —
// the []any of block maps that shape decodes into. It reads the first
// text-type block, which is the only shape any fixture in this file ever
// sends as the message under test; the cache_control marker alongside it (if
// present) is irrelevant to the text being asserted.
func decodedMessageText(content any) (string, bool) {
	switch v := content.(type) {
	case string:
		return v, true
	case []any:
		for _, b := range v {
			block, ok := b.(map[string]any)
			if !ok {
				continue
			}
			if text, ok := block["text"].(string); ok {
				return text, true
			}
		}
	}
	return "", false
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
	// Two user turns => two terminal results AND one system/init per turn. The
	// per-turn re-emit is the #6040 workaround: OpenClaw re-populates its per-turn
	// native-tool-authority allowlist only from a fresh system/init, and a warm
	// session whose prior run crashed loses the captured authority, so every turn
	// must re-announce it or tool calls stay rejected until /new. (This diverges
	// from the real claude-cli's once-per-process init; see StreamEmitter.systemInit.)
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
	if inits != 2 {
		t.Errorf("system/init count = %d, want one per turn (2) after the #6040 per-turn re-emit", inits)
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

// TestRun_PrunesStaleThinkingFromPastToolUseTurns is the end-to-end wiring
// proof for the "thinking-prune" seam in runTurn (session/run.go): it drives
// TWO user turns through the REAL Run/runTurn/RunLoop path, where turn 1
// contains a genuine tool_use round trip whose assistant message carries a
// thinking block alongside the tool_use block (exactly the shape a real
// thinking-capable core model returns). It then inspects the RAW request
// body the second turn's API call sends and asserts the thinking block from
// turn 1's now-past, completed tool_use message is gone while the tool_use
// block survives untouched. Without the anthropic.PruneThinkingFromHistory
// call at the top of runTurn, that stale thinking block would still be
// there — session.run.go's replaceFinalAssistantText only ever flattens a
// turn's own TERMINAL message, never an intermediate tool_use turn like
// this one.
func TestRun_PrunesStaleThinkingFromPastToolUseTurns(t *testing.T) {
	call := 0
	var turn2RequestBody struct {
		Messages []anthropic.Message `json:"messages"`
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call++
		if call == 3 {
			// Turn 2's one and only call: capture exactly what went over the
			// wire, since that is what this test is proving about.
			json.NewDecoder(r.Body).Decode(&turn2RequestBody)
		}
		w.Header().Set("Content-Type", "application/json")
		switch call {
		case 1:
			// Turn 1, round 1: a thinking block grounding the tool_use that
			// follows it, in the same shape MessageResponse.RawContent/
			// RunLoop echo verbatim into history (see client.go's
			// ContentBlock/RawContent doc).
			w.Write([]byte(`{
				"content": [
					{"type":"thinking","thinking":"let me check","signature":"sig-abc123"},
					{"type":"tool_use","id":"toolu_1","name":"bash","input":{"command":"echo hi"}}
				],
				"stop_reason": "tool_use",
				"usage": {"input_tokens": 10, "output_tokens": 5}
			}`))
		case 2:
			// Turn 1, round 2: the real Dispatcher's tool_result already fed
			// back by RunLoop; this closes turn 1.
			w.Write([]byte(`{
				"content": [{"type":"text","text":"done turn 1"}],
				"stop_reason": "end_turn",
				"usage": {"input_tokens": 8, "output_tokens": 4}
			}`))
		default:
			// Turn 2: whatever history led up to this call is what matters.
			w.Write([]byte(`{
				"content": [{"type":"text","text":"done turn 2"}],
				"stop_reason": "end_turn",
				"usage": {"input_tokens": 8, "output_tokens": 4}
			}`))
		}
	}))
	defer server.Close()

	client := anthropic.NewClient("secret", server.URL, time.Second)
	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin: strings.NewReader(strings.Join([]string{
			initializeLine("sys"),
			userLine("s", "first"),
			userLine("s", "second"),
		}, "\n") + "\n"),
		Stdout:     &out,
		Client:     client,
		Dispatcher: tools.NewDispatcher(),
		Store:      NewStore(t.TempDir()),
		SessionID:  "s",
		Model:      "m",
		MaxTokens:  1024,
		NewUUID:    seqUUID(),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if call != 3 {
		t.Fatalf("API calls = %d, want 3 (2 for turn 1's tool_use round trip, 1 for turn 2)", call)
	}

	// Message layout turn 2 must send: [user "first", assistant (turn 1's
	// intermediate tool_use turn), user (tool_result), assistant ("done turn
	// 1", flattened to plain text), user "second"]. Index 1 is the one under
	// test: originally thinking+tool_use, it must arrive with ONLY the
	// tool_use block.
	if len(turn2RequestBody.Messages) < 2 {
		t.Fatalf("turn 2 request messages = %d, want at least 2 (turn 1's history first)", len(turn2RequestBody.Messages))
	}
	blocks, ok := turn2RequestBody.Messages[1].Content.([]any)
	if !ok {
		t.Fatalf("messages[1].Content = %#v (%T), want a content-block array", turn2RequestBody.Messages[1].Content, turn2RequestBody.Messages[1].Content)
	}
	sawToolUse := false
	for _, b := range blocks {
		block, _ := b.(map[string]any)
		switch block["type"] {
		case "thinking", "redacted_thinking":
			t.Errorf("turn 2's request still carries a stale %v block from turn 1's completed tool_use turn: %v", block["type"], blocks)
		case "tool_use":
			sawToolUse = true
		}
	}
	if !sawToolUse {
		t.Errorf("turn 2's request lost the tool_use block entirely, want it to survive pruning: %v", blocks)
	}
}

// fakeToneDigester is a stub ToneDigester that returns a fixed digest and
// records the user texts it was handed, so a test can assert both the injected
// block AND that the digester saw the session's annotated user messages.
type fakeToneDigester struct {
	digest   string
	gotTexts [][]string
}

func (f *fakeToneDigester) Digest(userTexts []string) string {
	f.gotTexts = append(f.gotTexts, append([]string(nil), userTexts...))
	return f.digest
}

// TestRun_ToneDigestInjectedButNotPersisted is the end-to-end seam for the
// ephemeral tone digest, mirroring the superego ephemeral-tail contract: for a
// turn with annotations the digest MUST be present as the LAST message of the
// core's context, and it must NOT be persisted into the shared history
// afterward. A two-turn session proves both: each turn sends the digest as its
// trailing message, yet the second turn's history (and the persisted store)
// carry only the real user+assistant messages, never the digest.
func TestRun_ToneDigestInjectedButNotPersisted(t *testing.T) {
	const digest = "<<<TOM_DA_CONVERSA_INICIO>>>\n- cansaço — nível máx 4, valência negativa, 1x\n<<<TOM_DA_CONVERSA_FIM>>>"

	var lastMessages [][]anthropic.Message
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []anthropic.Message `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		lastMessages = append(lastMessages, body.Messages)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	client := anthropic.NewClient("secret", server.URL, time.Second)
	store := NewStore(t.TempDir())
	digester := &fakeToneDigester{digest: digest}

	in := strings.Join([]string{
		initializeLine("sys"),
		userLine("sess-1", "estou exausto"),
		userLine("sess-1", "e agora?"),
	}, "\n") + "\n"
	var out strings.Builder

	err := Run(context.Background(), Config{
		Stdin: strings.NewReader(in), Stdout: &out, Client: client,
		Dispatcher: tools.NewDispatcher(), Store: store, SessionID: "sess-1",
		Model: "m", MaxTokens: 1024, NewUUID: seqUUID(), ToneDigester: digester,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(lastMessages) != 2 {
		t.Fatalf("captured %d core calls, want 2 (one per user turn)", len(lastMessages))
	}

	// PRESENT: every turn's core context ends with the ephemeral digest.
	for turn, msgs := range lastMessages {
		if len(msgs) == 0 {
			t.Fatalf("turn %d sent no messages", turn+1)
		}
		last := msgs[len(msgs)-1]
		if last.Role != "user" || last.Content != digest {
			t.Errorf("turn %d: last core message = %+v, want the ephemeral digest", turn+1, last)
		}
	}

	// NOT PERSISTED: the second turn carries only the real prior user+assistant
	// plus the new user message plus this turn's digest = 4. Had the first turn's
	// digest been persisted into history, the count would be larger.
	if n := len(lastMessages[1]); n != 4 {
		t.Errorf("second turn sent %d messages, want 4 (prior user+assistant + new user + ephemeral digest); extra implies the digest leaked into history", n)
	}

	// NOT PERSISTED: the store holds only real conversation turns, never the
	// digest block.
	saved, err := store.Load("sess-1")
	if err != nil {
		t.Fatalf("store.Load: %v", err)
	}
	for i, m := range saved {
		if s, ok := m.Content.(string); ok && strings.Contains(s, "TOM_DA_CONVERSA") {
			t.Errorf("persisted message %d carries the ephemeral digest: %q", i, s)
		}
	}
	if len(saved) != 4 {
		t.Errorf("persisted %d messages, want 4 (2 turns of user+assistant, no digest)", len(saved))
	}

	// The digester was fed the session's annotated user texts, growing by one per
	// turn (prior turns' texts + the current one), never the assistant text.
	if len(digester.gotTexts) != 2 {
		t.Fatalf("digester called %d times, want 2", len(digester.gotTexts))
	}
	if got := digester.gotTexts[0]; len(got) != 1 || got[0] != "estou exausto" {
		t.Errorf("first digest call texts = %v, want [estou exausto]", got)
	}
	if got := digester.gotTexts[1]; len(got) != 2 || got[0] != "estou exausto" || got[1] != "e agora?" {
		t.Errorf("second digest call texts = %v, want [estou exausto, e agora?]", got)
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

// bigEnoughForFloor clears every model's cacheable floor (opus/sonnet/
// fable's 1024 tokens ~4096 bytes, haiku's 4096 tokens ~16384 bytes) by
// itself, so a test proving the message-history cache breakpoint lands
// doesn't have to reason about exact byte/token math.
var bigEnoughForFloor = strings.Repeat("prompt cache history content. ", 1000) // ~30k bytes

// TestTextOnlyRedraft_marksCanonicalHistoryNotTail proves textOnlyRedraft
// closes the coverage gap the cache-coverage refactor targeted: canonical
// (the core's res.Messages, which always ends in the echoed final assistant
// draft — see governTurn's reviewMessages comment) gets the client's
// prompt-cache breakpoint, the ephemeral tail never does, and two redraft
// rounds over the SAME canonical with a DIFFERENT tail send a byte-identical
// marked canonical prefix — the shape that lets the second round's call read
// the first round's cache entry back instead of re-billing the turn's whole
// history again.
func TestTextOnlyRedraft_marksCanonicalHistoryNotTail(t *testing.T) {
	var sentMessages [][]anthropic.Message
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []anthropic.Message `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		sentMessages = append(sentMessages, body.Messages)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"content":[{"type":"text","text":"revised"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	client := anthropic.NewClient("secret", server.URL, time.Second)
	cfg := Config{Client: client, Model: "claude-opus-4-8", MaxTokens: 1024}

	// canonical ends in the echoed raw assistant turn (text + tool_use blocks),
	// exactly the shape MessageResponse.RawContent produces and governTurn hands
	// to textOnlyRedraft as res.Messages — padded past opus's 1024-token floor,
	// since markMessagesCachePrefix's floor check runs over the whole array.
	canonical := []anthropic.Message{
		{Role: "user", Content: "please fix the bug. " + bigEnoughForFloor},
		{Role: "assistant", Content: []json.RawMessage{
			json.RawMessage(`{"type":"text","text":"done"}`),
			json.RawMessage(`{"type":"tool_use","id":"toolu_1","name":"bash","input":{"command":"pwd"}}`),
		}},
	}
	var usage anthropic.Usage

	for i, critique := range []string{"CRITIQUE-ROUND-1", "CRITIQUE-ROUND-2"} {
		tail := []anthropic.Message{{Role: "user", Content: critique}}
		if _, err := textOnlyRedraft(context.Background(), cfg, "system prompt", canonical, tail, &usage); err != nil {
			t.Fatalf("round %d: textOnlyRedraft error: %v", i+1, err)
		}
	}

	if len(sentMessages) != 2 {
		t.Fatalf("want 2 calls, got %d", len(sentMessages))
	}
	for i, msgs := range sentMessages {
		// canonical's 2 entries plus the 1-message tail.
		if len(msgs) != 3 {
			t.Fatalf("round %d: request carried %d messages, want 3 (canonical's 2 + 1 tail)", i+1, len(msgs))
		}
	}

	// canonical's last message (index 1, the echoed assistant turn) must carry
	// the breakpoint on both rounds, and be byte-identical across rounds: that
	// identity is the whole point — the second call's cache lookback finds the
	// first call's entry only if the marked prefix matches exactly.
	marked1, err1 := json.Marshal(sentMessages[0][1])
	marked2, err2 := json.Marshal(sentMessages[1][1])
	if err1 != nil || err2 != nil {
		t.Fatalf("marshal error: %v / %v", err1, err2)
	}
	if !strings.Contains(string(marked1), `"cache_control":{"type":"ephemeral"}`) {
		t.Errorf("round 1: canonical's last message not marked: %s", marked1)
	}
	if string(marked1) != string(marked2) {
		t.Errorf("canonical's marked message differs between rounds:\nround1=%s\nround2=%s", marked1, marked2)
	}

	// The tail (index 2, the per-round critique) must never carry the
	// breakpoint — marking it would turn every round into a fresh,
	// never-read cache write instead of reusing the one above.
	for i, msgs := range sentMessages {
		tailJSON, err := json.Marshal(msgs[2])
		if err != nil {
			t.Fatalf("round %d: marshal error: %v", i+1, err)
		}
		if strings.Contains(string(tailJSON), "cache_control") {
			t.Errorf("round %d: tail carries the cache breakpoint, want it only on canonical: %s", i+1, tailJSON)
		}
	}
}
