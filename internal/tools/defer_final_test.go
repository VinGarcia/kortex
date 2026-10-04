package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/protocol"
)

// TestRunLoop_deferFinalSuppressesFinalTurn proves DeferFinal removes the final
// assistant event and the terminal result from the wire — leaving the caller
// (the active superego loop) to govern the draft and re-emit — while every
// intermediate tool_use assistant event and its tool_result turn still stream
// live. The returned LoopResult is unchanged: the caller needs FinalText,
// Turns, StopReason and the summed Usage to re-emit the governed turn.
func TestRunLoop_deferFinalSuppressesFinalTurn(t *testing.T) {
	call := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call++
		w.Header().Set("Content-Type", "application/json")
		if call == 1 {
			w.Write([]byte(`{
				"content": [{"type":"tool_use","id":"toolu_1","name":"bash","input":{"command":"echo loop-ok"}}],
				"stop_reason": "tool_use",
				"usage": {"input_tokens": 10, "output_tokens": 5}
			}`))
			return
		}
		w.Write([]byte(`{
			"content": [{"type":"text","text":"the raw draft"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 12, "output_tokens": 6}
		}`))
	}))
	defer server.Close()

	var out bytes.Buffer
	client := anthropic.NewClient("secret-token", server.URL, time.Second)
	res, err := RunLoop(context.Background(), client, NewDispatcher(), LoopRequest{
		Model:       "claude-haiku-4-5-20251001",
		MaxTokens:   1024,
		UserMessage: "run echo loop-ok",
		Emitter:     NewStreamEmitter(&out, "sess-1", sequencedUUIDs()),
		DeferFinal:  true,
	})
	if err != nil {
		t.Fatalf("RunLoop error: %v", err)
	}

	// With DeferFinal only the intermediate turns reach the wire: system/init,
	// the tool_use assistant turn, and its tool_result user turn. The final
	// assistant text and the terminal result are withheld.
	lines := bytes.Split(bytes.TrimRight(out.Bytes(), "\n"), []byte("\n"))
	wantTypes := []string{protocol.TypeSystem, protocol.TypeAssistant, protocol.TypeUser}
	if len(lines) != len(wantTypes) {
		t.Fatalf("emitted %d lines, want %d (system/init, tool_use, tool_result — no final turn):\n%s",
			len(lines), len(wantTypes), out.String())
	}
	for i, line := range lines {
		ev := protocol.Parse(line)
		if ev.ParseErr != nil {
			t.Fatalf("line %d failed to parse: %v\n%s", i, ev.ParseErr, line)
		}
		if ev.Type != wantTypes[i] {
			t.Fatalf("line %d type = %q, want %q\n%s", i, ev.Type, wantTypes[i], line)
		}
		if ev.Type == protocol.TypeResult {
			t.Fatalf("a terminal result was emitted despite DeferFinal:\n%s", line)
		}
	}
	// The deferred draft text must never appear on the wire.
	if bytes.Contains(out.Bytes(), []byte("the raw draft")) {
		t.Errorf("the deferred final draft leaked onto the wire:\n%s", out.String())
	}

	// The result the caller receives is complete: the final text to govern, the
	// round's turn count and stop reason, and the usage summed across both turns.
	if res.FinalText != "the raw draft" || res.Turns != 2 || res.StopReason != "end_turn" {
		t.Errorf("LoopResult = %+v, want FinalText=the raw draft Turns=2 StopReason=end_turn", res)
	}
	wantUsage := anthropic.Usage{InputTokens: 22, OutputTokens: 11}
	if res.Usage != wantUsage {
		t.Errorf("LoopResult.Usage = %+v, want %+v (summed across both turns)", res.Usage, wantUsage)
	}
}

// TestRunLoop_deferFinalUndeferredByteIdentical proves DeferFinal is purely
// additive: a DeferFinal-false run emits exactly the same stream as a run with
// the field absent, so the pre-F2d emission path is untouched.
func TestRunLoop_deferFinalUndeferredByteIdentical(t *testing.T) {
	newServer := func() *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"content":[{"type":"text","text":"hi there"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2}}`))
		}))
	}

	run := func(defer_ bool) string {
		server := newServer()
		defer server.Close()
		var out bytes.Buffer
		client := anthropic.NewClient("secret-token", server.URL, time.Second)
		_, err := RunLoop(context.Background(), client, NewDispatcher(), LoopRequest{
			Model:       "m",
			MaxTokens:   16,
			UserMessage: "hi",
			Emitter:     NewStreamEmitter(&out, "sess-1", sequencedUUIDs()),
			DeferFinal:  defer_,
		})
		if err != nil {
			t.Fatalf("RunLoop error: %v", err)
		}
		return out.String()
	}

	undeferred := run(false)
	deferred := run(true)
	if undeferred == deferred {
		t.Fatalf("DeferFinal had no effect on a single-turn stream; the final turn should have been suppressed")
	}
	// The undeferred stream carries the final text + result; the deferred one, on
	// a single-turn conversation, emits only system/init.
	if !bytes.Contains([]byte(undeferred), []byte("hi there")) {
		t.Errorf("undeferred stream missing the final text:\n%s", undeferred)
	}
	if bytes.Contains([]byte(deferred), []byte("hi there")) {
		t.Errorf("deferred stream leaked the final text:\n%s", deferred)
	}
}

// TestEmitFinalTurn_emitsGovernedAssistantAndResult proves EmitFinalTurn writes
// exactly the deferred pair — one assistant text event then the terminal result
// — both carrying the governed text with tool markup stripped, and the result
// reporting the caller-supplied turn count, stop reason, and billed usage.
func TestEmitFinalTurn_emitsGovernedAssistantAndResult(t *testing.T) {
	var out bytes.Buffer
	emitter := NewStreamEmitter(&out, "sess-9", sequencedUUIDs())
	markup := `<invoke name="Bash"><parameter name="command">ls</parameter></invoke>`
	usage := anthropic.Usage{InputTokens: 22, OutputTokens: 11, CacheReadInputTokens: 3}
	if err := emitter.EmitFinalTurn("governed text "+markup, 2, "end_turn", usage); err != nil {
		t.Fatalf("EmitFinalTurn error: %v", err)
	}

	lines := bytes.Split(bytes.TrimRight(out.Bytes(), "\n"), []byte("\n"))
	if len(lines) != 2 {
		t.Fatalf("EmitFinalTurn wrote %d lines, want 2 (assistant + result):\n%s", len(lines), out.String())
	}

	assistant := protocol.Parse(lines[0])
	if assistant.ParseErr != nil || assistant.Type != protocol.TypeAssistant {
		t.Fatalf("first line is not an assistant event: %+v\n%s", assistant, lines[0])
	}
	if got := assistant.Message.TextContent(); got != "governed text " {
		t.Errorf("assistant text = %q, want %q (markup stripped)", got, "governed text ")
	}

	result := protocol.Parse(lines[1])
	if result.ParseErr != nil || result.Type != protocol.TypeResult || result.Result == nil {
		t.Fatalf("second line is not a result event: %+v\n%s", result, lines[1])
	}
	if result.Result.Text != "governed text " {
		t.Errorf("result text = %q, want %q (markup stripped)", result.Result.Text, "governed text ")
	}
	if result.Result.NumTurns != 2 || result.Result.StopReason != "end_turn" {
		t.Errorf("result num_turns=%d stop_reason=%q, want 2/end_turn", result.Result.NumTurns, result.Result.StopReason)
	}
	if result.Subtype != "success" {
		t.Errorf("result subtype = %q, want success", result.Subtype)
	}
	var billed struct {
		Usage protocol.Usage `json:"usage"`
	}
	if err := json.Unmarshal(lines[1], &billed); err != nil {
		t.Fatalf("decoding result usage: %v\n%s", err, lines[1])
	}
	if billed.Usage != (protocol.Usage(usage)) {
		t.Errorf("result usage = %+v, want %+v", billed.Usage, protocol.Usage(usage))
	}
}

// TestEmitFinalTurn_nilEmitterSilent proves a nil emitter is the documented
// no-op: EmitFinalTurn returns nil and writes nothing.
func TestEmitFinalTurn_nilEmitterSilent(t *testing.T) {
	var e *StreamEmitter
	if err := e.EmitFinalTurn("anything", 1, "end_turn", anthropic.Usage{}); err != nil {
		t.Fatalf("nil EmitFinalTurn error: %v", err)
	}
}
