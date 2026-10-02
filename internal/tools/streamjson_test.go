package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/protocol"
)

// sequencedUUIDs returns a newUUID func handing out uuid-1, uuid-2, ... so the
// emitted event identities are deterministic for golden assertions.
func sequencedUUIDs() func() string {
	n := 0
	return func() string {
		n++
		return "uuid-" + string(rune('0'+n))
	}
}

// TestRunLoop_emitsStreamJSON drives the same bash round-trip as
// TestRunLoop_bashRoundTrip but with an Emitter wired, and asserts RunLoop
// produces the full stream-json sequence: system/init, the tool_use assistant
// turn, the tool_result user turn, the final text assistant turn, and the
// terminal result.
func TestRunLoop_emitsStreamJSON(t *testing.T) {
	call := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call++
		w.Header().Set("Content-Type", "application/json")
		if call == 1 {
			w.Write([]byte(`{
				"content": [{"type":"tool_use","id":"toolu_1","name":"bash","input":{"command":"echo loop-ok"},"caller":{"type":"direct"}}],
				"stop_reason": "tool_use",
				"usage": {"input_tokens": 10, "output_tokens": 5}
			}`))
			return
		}
		w.Write([]byte(`{
			"content": [{"type":"text","text":"the command printed loop-ok"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 12, "output_tokens": 6, "cache_read_input_tokens": 3}
		}`))
	}))
	defer server.Close()

	var out bytes.Buffer
	client := anthropic.NewClient("secret-token", server.URL, time.Second)
	_, err := RunLoop(context.Background(), client, NewDispatcher(), LoopRequest{
		Model:       "claude-haiku-4-5-20251001",
		MaxTokens:   1024,
		UserMessage: "run echo loop-ok",
		Emitter:     NewStreamEmitter(&out, "sess-1", sequencedUUIDs()),
	})
	if err != nil {
		t.Fatalf("RunLoop error: %v", err)
	}

	lines := bytes.Split(bytes.TrimRight(out.Bytes(), "\n"), []byte("\n"))
	wantTypes := []string{
		protocol.TypeSystem,
		protocol.TypeAssistant,
		protocol.TypeUser,
		protocol.TypeAssistant,
		protocol.TypeResult,
	}
	if len(lines) != len(wantTypes) {
		t.Fatalf("emitted %d lines, want %d:\n%s", len(lines), len(wantTypes), out.String())
	}

	// Parity: every emitted line decodes through the real protocol parser (the
	// read side OpenClaw uses) into the type the sequence expects.
	events := make([]protocol.Event, len(lines))
	for i, line := range lines {
		events[i] = protocol.Parse(line)
		if events[i].ParseErr != nil {
			t.Fatalf("line %d failed to parse: %v\n%s", i, events[i].ParseErr, line)
		}
		if events[i].Type != wantTypes[i] {
			t.Fatalf("line %d type = %q, want %q\n%s", i, events[i].Type, wantTypes[i], line)
		}
		if events[i].SessionID != "" && events[i].SessionID != "sess-1" {
			t.Errorf("line %d session_id = %q, want sess-1", i, events[i].SessionID)
		}
	}

	// The tool_use assistant turn round-trips to the block the model emitted.
	toolUse := events[1].Message.Content
	if len(toolUse) != 1 || toolUse[0].Type != "tool_use" || toolUse[0].ID != "toolu_1" || toolUse[0].Name != "bash" {
		t.Fatalf("assistant tool_use block = %+v", toolUse)
	}
	if events[1].Message.Usage == nil || events[1].Message.Usage.OutputTokens != 5 {
		t.Errorf("assistant usage = %+v", events[1].Message.Usage)
	}

	// The tool_result user turn carries the real bash output keyed to the call.
	toolResult := events[2].Message.Content
	if len(toolResult) != 1 || toolResult[0].Type != "tool_result" || toolResult[0].ToolUseID != "toolu_1" {
		t.Fatalf("user tool_result block = %+v", toolResult)
	}
	if got := toolResult[0].ResultText(); !strings.Contains(got, "loop-ok") {
		t.Errorf("tool_result text = %q, want it to contain loop-ok", got)
	}

	// The terminal result mirrors the loop outcome.
	result := events[4].Result
	if result == nil || result.StopReason != "end_turn" || result.NumTurns != 2 {
		t.Fatalf("result = %+v", result)
	}
	if result.Text != "the command printed loop-ok" {
		t.Errorf("result text = %q", result.Text)
	}
	if events[4].Subtype != "success" {
		t.Errorf("result subtype = %q, want success", events[4].Subtype)
	}
}

// TestRunLoop_nilEmitterSilent proves a nil Emitter leaves RunLoop's behavior
// unchanged — no emission, same result — so the sink is purely additive.
func TestRunLoop_nilEmitterSilent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	client := anthropic.NewClient("secret-token", server.URL, time.Second)
	result, err := RunLoop(context.Background(), client, NewDispatcher(), LoopRequest{
		Model:       "m",
		MaxTokens:   16,
		UserMessage: "hi",
	})
	if err != nil {
		t.Fatalf("RunLoop error: %v", err)
	}
	if result.FinalText != "hi" || result.StopReason != "end_turn" || result.Turns != 1 {
		t.Errorf("unexpected result with nil emitter: %+v", result)
	}
}

// TestStreamEmitter_resultStripsToolMarkup proves the terminal result honors
// EmitResult's no-raw-markup contract via StripToolMarkup (F3d slice-2a).
func TestStreamEmitter_resultStripsToolMarkup(t *testing.T) {
	var out bytes.Buffer
	emitter := NewStreamEmitter(&out, "sess-9", sequencedUUIDs())
	markup := `<invoke name="Bash"><parameter name="command">ls</parameter></invoke>`
	err := emitter.result(LoopResult{FinalText: "done " + markup, StopReason: "end_turn", Turns: 1}, anthropic.Usage{})
	if err != nil {
		t.Fatalf("result emit error: %v", err)
	}
	ev := protocol.Parse(out.Bytes())
	if ev.Result == nil {
		t.Fatalf("parsed result nil: %s", out.String())
	}
	if strings.Contains(ev.Result.Text, "<invoke") {
		t.Errorf("result text still carries tool markup: %q", ev.Result.Text)
	}
	if ev.Result.Text != "done " {
		t.Errorf("result text = %q, want %q", ev.Result.Text, "done ")
	}
}

// TestAssistantBlocks_roundTrip pins the anthropic -> protocol block mapping
// against the emitter+parser: the mapped blocks survive EmitAssistant ->
// Parse unchanged, and non-text/tool_use blocks are dropped.
func TestAssistantBlocks_roundTrip(t *testing.T) {
	content := []anthropic.ContentBlock{
		{Type: "text", Text: "hello"},
		{Type: "thinking"}, // dropped: not modeled on the wire for this client
		{Type: "tool_use", ID: "toolu_7", Name: "read", Input: json.RawMessage(`{"path":"main.go"}`)},
	}
	blocks := assistantBlocks(content)
	want := []protocol.ContentBlock{
		{Type: "text", Text: "hello"},
		{Type: "tool_use", ID: "toolu_7", Name: "read", Input: json.RawMessage(`{"path":"main.go"}`)},
	}
	if !reflect.DeepEqual(blocks, want) {
		t.Fatalf("assistantBlocks = %+v, want %+v", blocks, want)
	}

	line, err := protocol.EmitAssistant("sess-1", "uuid-1", blocks, protocol.Usage{})
	if err != nil {
		t.Fatalf("EmitAssistant error: %v", err)
	}
	ev := protocol.Parse(line)
	if !reflect.DeepEqual([]protocol.ContentBlock(ev.Message.Content), want) {
		t.Errorf("round-tripped blocks = %+v, want %+v", ev.Message.Content, want)
	}
}
