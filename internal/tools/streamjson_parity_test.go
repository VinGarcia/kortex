package tools

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/protocol"
)

// This file proves the stream-json the kortex RunLoop emitter produces is
// STRUCTURALLY equivalent to what the real `claude` CLI emits for the same
// conversation. "Structurally equivalent" is defined against the only read
// side that matters — internal/protocol, the package that models exactly the
// fields OpenClaw consumes (F0). The comparison is therefore at the parsed
// level, not byte level: the fixtures in testdata/ are verbatim captures of
// `claude -p --output-format stream-json --verbose`, and both streams are run
// through protocol.Parse before comparison.
//
// Volatile fields (session id, per-event uuid, timestamps, request/message
// ids, token counts, cost, durations) are compared by presence/type, never by
// value — a captured fixture's uuid can never equal a freshly emitted one.
//
// Scope boundary: "structural equivalence" here is bounded by what
// internal/protocol models. The comparison only sees fields protocol.Event
// exposes, so a wire field protocol.go does not model is invisible to this
// test by construction. Concretely, claude's system/init carries tools, model,
// mcp_servers, slash_commands, etc. that kortex's EmitSystemInit omits; both
// collapse to {system,init} here. That is correct precisely because OpenClaw
// (the F0 reader protocol.go mirrors) consumes only the modeled subset — this
// test proves parity within that contract, not against the full claude payload.
//
// Two documented, deliberate differences are normalized away rather than
// treated as divergences, because OpenClaw's read side tolerates both:
//   - claude emits auxiliary events kortex has no analog for: rate_limit_event
//     and system/{thinking_tokens}. They carry no field this backend owns.
//   - claude emits thinking content blocks (and a separate assistant event per
//     content block); this client does not model thinking, so a thinking-only
//     assistant event has no kortex counterpart and is dropped.
//
// What remains after normalization — the system/init, the assistant turns
// carrying text/tool_use, the user turn echoing tool_result, and the terminal
// result — must match in type, order, and per-turn content-block shape.

// semEvent is the OpenClaw-consumed projection of one wire event: the event
// type, its disambiguating subtype (system/result only), and the ordered
// content-block types of a message event. It deliberately omits every volatile
// value so two independently produced streams of the same conversation project
// to the same slice.
type semEvent struct {
	Type       string
	Subtype    string
	BlockTypes []string
}

// normalize projects a parsed stream into the semantic skeleton OpenClaw
// reads, dropping the claude-only auxiliary events and thinking blocks that
// this backend legitimately never emits (see the file comment).
func normalize(events []protocol.Event) []semEvent {
	out := make([]semEvent, 0, len(events))
	for _, ev := range events {
		switch {
		case ev.Type == "rate_limit_event":
			continue
		case ev.Type == protocol.TypeSystem && ev.Subtype != "init":
			// thinking_tokens and any other informational system sub-event.
			continue
		case ev.Type == protocol.TypeAssistant:
			blocks := nonThinkingBlockTypes(ev.Message)
			if len(blocks) == 0 {
				// A thinking-only assistant event: kortex emits nothing for it.
				continue
			}
			out = append(out, semEvent{Type: ev.Type, BlockTypes: blocks})
		case ev.Type == protocol.TypeUser:
			out = append(out, semEvent{Type: ev.Type, BlockTypes: blockTypes(ev.Message)})
		default:
			out = append(out, semEvent{Type: ev.Type, Subtype: ev.Subtype})
		}
	}
	return out
}

func nonThinkingBlockTypes(m *protocol.Message) []string {
	if m == nil {
		return nil
	}
	var types []string
	for _, b := range m.Content {
		if b.Type == "thinking" {
			continue
		}
		types = append(types, b.Type)
	}
	return types
}

func blockTypes(m *protocol.Message) []string {
	if m == nil {
		return nil
	}
	types := make([]string, len(m.Content))
	for i, b := range m.Content {
		types[i] = b.Type
	}
	return types
}

func parseStream(t *testing.T, data []byte) []protocol.Event {
	t.Helper()
	lines := bytes.Split(bytes.TrimRight(data, "\n"), []byte("\n"))
	events := make([]protocol.Event, 0, len(lines))
	for i, line := range lines {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		ev := protocol.Parse(line)
		if ev.ParseErr != nil {
			t.Fatalf("stream line %d failed to parse: %v\n%s", i, ev.ParseErr, line)
		}
		events = append(events, ev)
	}
	return events
}

func loadFixture(t *testing.T, name string) []protocol.Event {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("reading fixture %s: %v", name, err)
	}
	return parseStream(t, data)
}

// runKortexStream drives RunLoop with the emitter wired against a fake
// Anthropic server scripted with the given turns, and returns the parsed
// emitted stream. Each turn is one upstream model response; the last must
// carry a non-tool_use stop_reason to terminate the loop.
func runKortexStream(t *testing.T, userMessage string, turns []string) []protocol.Event {
	t.Helper()
	call := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// A scripted turn count that RunLoop overruns is a test-authoring bug;
		// fail it cleanly (the client surfaces the 500 as a RunLoop error the
		// caller's error check catches) rather than panicking on turns[call].
		if call >= len(turns) {
			t.Errorf("RunLoop made upstream call %d but only %d turn(s) were scripted", call+1, len(turns))
			http.Error(w, "unexpected extra upstream call", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(turns[call]))
		call++
	}))
	defer server.Close()

	var out bytes.Buffer
	client := anthropic.NewClient("secret-token", server.URL, time.Second)
	_, err := RunLoop(context.Background(), client, NewDispatcher(), LoopRequest{
		Model:       "claude-haiku-4-5-20251001",
		MaxTokens:   1024,
		UserMessage: userMessage,
		Emitter:     NewStreamEmitter(&out, "sess-kortex", sequencedUUIDs()),
	})
	if err != nil {
		t.Fatalf("RunLoop error: %v", err)
	}
	return parseStream(t, out.Bytes())
}

// assertSessionConsistent checks every event carries the same non-empty
// session id — OpenClaw keys the round on it, and a stream that changed or
// dropped it mid-round would break the reader regardless of event shape.
func assertSessionConsistent(t *testing.T, label string, events []protocol.Event) {
	t.Helper()
	var want string
	for i, ev := range events {
		if ev.SessionID == "" {
			// control/aux lines legitimately omit it; the ones OpenClaw keys on
			// (system/assistant/user/result) must carry it.
			switch ev.Type {
			case protocol.TypeSystem, protocol.TypeAssistant, protocol.TypeUser, protocol.TypeResult:
				t.Errorf("%s event %d (%s) has empty session_id", label, i, ev.Type)
			}
			continue
		}
		if want == "" {
			want = ev.SessionID
			continue
		}
		if ev.SessionID != want {
			t.Errorf("%s event %d session_id = %q, want consistent %q", label, i, ev.SessionID, want)
		}
	}
}

// resultOf returns the terminal result event of a stream.
func resultOf(t *testing.T, label string, events []protocol.Event) protocol.Event {
	t.Helper()
	last := events[len(events)-1]
	if last.Type != protocol.TypeResult || last.Result == nil {
		t.Fatalf("%s: last event is not a result: %+v", label, last)
	}
	return last
}

// assertResultParity compares the two terminal results on the fields OpenClaw
// reads, by value where the value is part of the contract (subtype, stop
// reason, num_turns, error flag) and by presence where it is volatile (the
// result text must be non-empty; usage counters exist as ints regardless of
// value).
func assertResultParity(t *testing.T, claude protocol.Event, kortex protocol.Event) {
	t.Helper()
	c, k := claude.Result, kortex.Result
	if c.IsError != k.IsError {
		t.Errorf("result is_error: claude=%v kortex=%v", c.IsError, k.IsError)
	}
	if claude.Subtype != kortex.Subtype {
		t.Errorf("result subtype: claude=%q kortex=%q", claude.Subtype, kortex.Subtype)
	}
	if c.StopReason != k.StopReason {
		t.Errorf("result stop_reason: claude=%q kortex=%q", c.StopReason, k.StopReason)
	}
	if c.NumTurns != k.NumTurns {
		t.Errorf("result num_turns: claude=%d kortex=%d (turn-count semantics diverged)", c.NumTurns, k.NumTurns)
	}
	if k.Text == "" {
		t.Errorf("kortex result text is empty, want non-empty like claude's %q", c.Text)
	}
}

// TestStreamJSONParity_textOnly proves the text-only conversation (one
// assistant turn, no tools) projects to the same skeleton on both sides.
func TestStreamJSONParity_textOnly(t *testing.T) {
	claude := loadFixture(t, "claude_text.ndjson")
	kortex := runKortexStream(t, "Reply with exactly the text: parity-check-ok and nothing else.", []string{
		`{"content":[{"type":"text","text":"parity-check-ok"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":4}}`,
	})

	wantSkeleton := []semEvent{
		{Type: protocol.TypeSystem, Subtype: "init"},
		{Type: protocol.TypeAssistant, BlockTypes: []string{"text"}},
		{Type: protocol.TypeResult, Subtype: "success"},
	}
	cNorm := normalize(claude)
	kNorm := normalize(kortex)
	if !reflect.DeepEqual(cNorm, wantSkeleton) {
		t.Fatalf("claude fixture skeleton drifted:\n got %+v\nwant %+v", cNorm, wantSkeleton)
	}
	if !reflect.DeepEqual(kNorm, cNorm) {
		t.Fatalf("kortex skeleton != claude skeleton:\n kortex %+v\n claude %+v", kNorm, cNorm)
	}

	assertSessionConsistent(t, "claude", claude)
	assertSessionConsistent(t, "kortex", kortex)
	assertResultParity(t, resultOf(t, "claude", claude), resultOf(t, "kortex", kortex))
}

// TestStreamJSONParity_toolCall proves the single-tool-call conversation
// (assistant tool_use -> user tool_result -> assistant text) projects to the
// same skeleton on both sides, including the tool_use<->tool_result linkage
// and the num_turns=2 round-trip count.
func TestStreamJSONParity_toolCall(t *testing.T) {
	claude := loadFixture(t, "claude_tool.ndjson")
	kortex := runKortexStream(t, "Run the shell command: echo parity-tool-ok  — then tell me its exact output.", []string{
		`{"content":[{"type":"tool_use","id":"toolu_k1","name":"bash","input":{"command":"echo parity-tool-ok"}}],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5}}`,
		`{"content":[{"type":"text","text":"The exact output is: parity-tool-ok"}],"stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":6}}`,
	})

	wantSkeleton := []semEvent{
		{Type: protocol.TypeSystem, Subtype: "init"},
		{Type: protocol.TypeAssistant, BlockTypes: []string{"tool_use"}},
		{Type: protocol.TypeUser, BlockTypes: []string{"tool_result"}},
		{Type: protocol.TypeAssistant, BlockTypes: []string{"text"}},
		{Type: protocol.TypeResult, Subtype: "success"},
	}
	cNorm := normalize(claude)
	kNorm := normalize(kortex)
	if !reflect.DeepEqual(cNorm, wantSkeleton) {
		t.Fatalf("claude fixture skeleton drifted:\n got %+v\nwant %+v", cNorm, wantSkeleton)
	}
	if !reflect.DeepEqual(kNorm, cNorm) {
		t.Fatalf("kortex skeleton != claude skeleton:\n kortex %+v\n claude %+v", kNorm, cNorm)
	}

	assertSessionConsistent(t, "claude", claude)
	assertSessionConsistent(t, "kortex", kortex)
	assertResultParity(t, resultOf(t, "claude", claude), resultOf(t, "kortex", kortex))
	assertToolLinkage(t, "claude", claude)
	assertToolLinkage(t, "kortex", kortex)
}

// assertToolLinkage checks the structural invariant OpenClaw relies on to pair
// a tool result with its call: the user turn's tool_result.tool_use_id equals
// the id of a tool_use the assistant emitted, and both ids are non-empty.
func assertToolLinkage(t *testing.T, label string, events []protocol.Event) {
	t.Helper()
	toolUseIDs := map[string]bool{}
	for _, ev := range events {
		if ev.Type != protocol.TypeAssistant || ev.Message == nil {
			continue
		}
		for _, b := range ev.Message.Content {
			if b.Type == "tool_use" {
				if b.ID == "" || b.Name == "" {
					t.Errorf("%s: tool_use block missing id/name: %+v", label, b)
				}
				toolUseIDs[b.ID] = true
			}
		}
	}
	for _, ev := range events {
		if ev.Type != protocol.TypeUser || ev.Message == nil {
			continue
		}
		for _, b := range ev.Message.Content {
			if b.Type != "tool_result" {
				continue
			}
			if b.ToolUseID == "" {
				t.Errorf("%s: tool_result missing tool_use_id", label)
				continue
			}
			if !toolUseIDs[b.ToolUseID] {
				t.Errorf("%s: tool_result.tool_use_id %q matches no emitted tool_use", label, b.ToolUseID)
			}
		}
	}
}
