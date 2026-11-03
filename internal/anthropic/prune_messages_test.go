package anthropic

import (
	"encoding/json"
	"reflect"
	"testing"
)

// thinkingBlock and redactedThinkingBlock are the two raw shapes
// thinkingBlockTypes classifies (see cache_messages.go) — reused here so this
// file's fixtures stay byte-for-byte consistent with what a real echoed
// assistant turn carries.
var (
	thinkingBlock         = json.RawMessage(`{"type":"thinking","thinking":"reasoning...","signature":"sig123"}`)
	redactedThinkingBlock = json.RawMessage(`{"type":"redacted_thinking","data":"opaque"}`)
	textBlock             = json.RawMessage(`{"type":"text","text":"hello"}`)
	toolUseBlock          = json.RawMessage(`{"type":"tool_use","id":"toolu_1","name":"bash","input":{"command":"pwd"}}`)
)

// TestPruneThinkingFromHistory_dropsThinkingBlocks proves the one thing this
// feature exists for: an intermediate assistant turn (thinking followed by
// the tool_use that acts on it) loses the thinking/redacted_thinking blocks
// while keeping the tool_use block intact, byte-identical.
func TestPruneThinkingFromHistory_dropsThinkingBlocks(t *testing.T) {
	in := []Message{
		{Role: "user", Content: "do the thing"},
		{Role: "assistant", Content: []json.RawMessage{thinkingBlock, toolUseBlock}},
		{Role: "user", Content: []ToolResultBlock{NewToolResultBlock("toolu_1", "ok", false)}},
	}
	out := PruneThinkingFromHistory(in)

	gotAssistant, ok := out[1].Content.([]json.RawMessage)
	if !ok {
		t.Fatalf("assistant content = %#v, want []json.RawMessage", out[1].Content)
	}
	if len(gotAssistant) != 1 {
		t.Fatalf("assistant blocks = %d, want 1 (thinking dropped, tool_use kept): %#v", len(gotAssistant), gotAssistant)
	}
	if string(gotAssistant[0]) != string(toolUseBlock) {
		t.Errorf("surviving block = %s, want byte-identical tool_use %s", gotAssistant[0], toolUseBlock)
	}

	// Untouched messages (user text, tool_result) must be byte/value-identical,
	// not just equivalent, since this is the exact shape re-sent on the wire.
	if out[0].Content != "do the thing" {
		t.Errorf("user message changed: %#v", out[0].Content)
	}
	results, ok := out[2].Content.([]ToolResultBlock)
	if !ok || len(results) != 1 || results[0].ToolUseID != "toolu_1" {
		t.Errorf("tool_result message changed: %#v", out[2].Content)
	}
}

// TestPruneThinkingFromHistory_dropsRedactedThinkingToo proves the redacted
// variant is pruned exactly like plain thinking — both are in
// thinkingBlockTypes (cache_messages.go), reused rather than re-checked here.
func TestPruneThinkingFromHistory_dropsRedactedThinkingToo(t *testing.T) {
	in := []Message{
		{Role: "assistant", Content: []json.RawMessage{redactedThinkingBlock, toolUseBlock}},
	}
	out := PruneThinkingFromHistory(in)
	got, ok := out[0].Content.([]json.RawMessage)
	if !ok || len(got) != 1 || string(got[0]) != string(toolUseBlock) {
		t.Errorf("content = %#v, want only the tool_use block to survive", out[0].Content)
	}
}

// TestPruneThinkingFromHistory_doesNotMutateInput proves the input slice and
// its nested raw-block slices are never written through: the caller
// (session.runTurn) hands this function the SAME []anthropic.Message it is
// about to reuse as the base for the next turn's live RunLoop clone, so any
// aliasing here would risk corrupting that base.
func TestPruneThinkingFromHistory_doesNotMutateInput(t *testing.T) {
	in := []Message{
		{Role: "assistant", Content: []json.RawMessage{thinkingBlock, toolUseBlock}},
	}
	original := make([]json.RawMessage, len(in[0].Content.([]json.RawMessage)))
	copy(original, in[0].Content.([]json.RawMessage))

	_ = PruneThinkingFromHistory(in)

	if !reflect.DeepEqual(in[0].Content.([]json.RawMessage), original) {
		t.Errorf("input mutated: %#v", in[0].Content)
	}
}

// TestPruneThinkingFromHistory_noThinkingIsNoop proves a history that
// already carries no thinking blocks (e.g. every message is a plain string,
// or an assistant turn with only a tool_use block) is returned unchanged —
// no unnecessary cloning, no reshaping of content that was never going to
// change. This also backs the no-op case of the determinism test below.
func TestPruneThinkingFromHistory_noThinkingIsNoop(t *testing.T) {
	in := []Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: []json.RawMessage{toolUseBlock}},
	}
	out := PruneThinkingFromHistory(in)
	if !reflect.DeepEqual(out, in) {
		t.Errorf("got %#v, want unchanged %#v", out, in)
	}
}

// TestPruneThinkingFromHistory_emptyIsNoop mirrors markMessagesCachePrefix's
// own empty-history fail-safe (cache_messages_test.go) for the same reason:
// a brand-new session's nil history must pass through untouched.
func TestPruneThinkingFromHistory_emptyIsNoop(t *testing.T) {
	if got := PruneThinkingFromHistory(nil); got != nil {
		t.Errorf("got %#v, want nil unchanged", got)
	}
}

// TestPruneThinkingFromHistory_allBlocksThinkingIsLeftAlone proves the one
// degenerate edge case this function must never produce: an assistant
// message whose content would become an EMPTY block array if thinking were
// stripped. The Messages API's content array cannot legally be empty, so
// rather than ever emit one, this message is left completely unpruned. This
// should not occur in practice (an intermediate assistant turn with no
// tool_use would not have looped), but the function must not assume that and
// must fail safe instead of emitting a shape the API would 400 on.
func TestPruneThinkingFromHistory_allBlocksThinkingIsLeftAlone(t *testing.T) {
	in := []Message{
		{Role: "assistant", Content: []json.RawMessage{thinkingBlock, redactedThinkingBlock}},
	}
	out := PruneThinkingFromHistory(in)
	got, ok := out[0].Content.([]json.RawMessage)
	if !ok || len(got) != 2 {
		t.Errorf("content = %#v, want both blocks left untouched (pruning would empty the array)", out[0].Content)
	}
}

// TestPruneThinkingFromHistory_isDeterministic is the cache-prefix-stability
// load-bearing test (see the 2026-11-03 cache work in cache_messages.go):
// pruning the SAME input history must produce byte-for-byte identical output
// on every call, or the "stable prefix" the prompt-cache breakpoint depends
// on would shift on some calls and never re-stabilize, destroying the
// savings the whole cache feature exists for. Proven by marshaling the
// output of two independent runs over equal (but not aliased) input and
// comparing the resulting JSON bytes — the same comparison the wire
// ultimately performs.
func TestPruneThinkingFromHistory_isDeterministic(t *testing.T) {
	build := func() []Message {
		return []Message{
			{Role: "user", Content: "do the thing"},
			{Role: "assistant", Content: []json.RawMessage{thinkingBlock, toolUseBlock}},
			{Role: "user", Content: []ToolResultBlock{NewToolResultBlock("toolu_1", "ok", false)}},
			{Role: "assistant", Content: []json.RawMessage{redactedThinkingBlock, textBlock}},
		}
	}

	out1 := PruneThinkingFromHistory(build())
	out2 := PruneThinkingFromHistory(build())

	raw1, err := json.Marshal(out1)
	if err != nil {
		t.Fatalf("marshal out1: %v", err)
	}
	raw2, err := json.Marshal(out2)
	if err != nil {
		t.Fatalf("marshal out2: %v", err)
	}
	if string(raw1) != string(raw2) {
		t.Fatalf("non-deterministic prune:\n  run1 = %s\n  run2 = %s", raw1, raw2)
	}

	// Idempotence: pruning already-pruned output must be a true no-op (the
	// second call finds nothing left to remove), which is what lets the
	// cache prefix re-stabilize after the one-time shape change instead of
	// drifting turn over turn.
	out3 := PruneThinkingFromHistory(out1)
	raw3, err := json.Marshal(out3)
	if err != nil {
		t.Fatalf("marshal out3: %v", err)
	}
	if string(raw1) != string(raw3) {
		t.Fatalf("pruning is not idempotent:\n  once   = %s\n  twice  = %s", raw1, raw3)
	}
}
