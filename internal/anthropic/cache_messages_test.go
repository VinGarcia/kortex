package anthropic

import (
	"reflect"
	"testing"
)

func TestMarkMessagesCachePrefix_emptyHistory(t *testing.T) {
	if got := MarkMessagesCachePrefix(nil); got != nil {
		t.Errorf("empty history: got %#v, want nil unchanged", got)
	}
}

func TestMarkMessagesCachePrefix_stringContent(t *testing.T) {
	in := []Message{{Role: "user", Content: "hello"}}
	out := MarkMessagesCachePrefix(in)

	if in[0].Content != "hello" {
		t.Fatalf("input mutated: %+v", in[0])
	}
	blocks, ok := out[0].Content.([]messageContentBlock)
	if !ok || len(blocks) != 1 {
		t.Fatalf("content = %#v, want one messageContentBlock", out[0].Content)
	}
	if blocks[0].Type != "text" || blocks[0].Text != "hello" {
		t.Errorf("block = %+v", blocks[0])
	}
	if blocks[0].CacheControl == nil || blocks[0].CacheControl.Type != "ephemeral" {
		t.Errorf("CacheControl = %+v, want ephemeral", blocks[0].CacheControl)
	}
}

// TestMarkMessagesCachePrefix_emptyStringContentIsFailSafe proves an empty
// user turn (content == "") is reported ineligible rather than wrapped into a
// breakpoint-carrying empty text block, matching markStableSystemPrefix's
// fail-safe contract for degenerate input.
func TestMarkMessagesCachePrefix_emptyStringContentIsFailSafe(t *testing.T) {
	in := []Message{{Role: "user", Content: ""}}
	out := MarkMessagesCachePrefix(in)
	if !reflect.DeepEqual(out, in) {
		t.Errorf("got %#v, want unchanged %#v", out, in)
	}
}

func TestMarkMessagesCachePrefix_toolResultBlocks(t *testing.T) {
	in := []Message{{Role: "user", Content: []ToolResultBlock{
		NewToolResultBlock("tool_1", "first", false),
		NewToolResultBlock("tool_2", "second", false),
	}}}
	out := MarkMessagesCachePrefix(in)

	results, ok := out[0].Content.([]ToolResultBlock)
	if !ok || len(results) != 2 {
		t.Fatalf("content = %#v", out[0].Content)
	}
	if results[0].CacheControl != nil {
		t.Errorf("first tool_result carries a breakpoint, want only the last")
	}
	if results[1].CacheControl == nil || results[1].CacheControl.Type != "ephemeral" {
		t.Errorf("last tool_result CacheControl = %+v, want ephemeral", results[1].CacheControl)
	}

	// The caller's original slice must stay untouched (no aliasing into it).
	origResults := in[0].Content.([]ToolResultBlock)
	if origResults[0].CacheControl != nil || origResults[1].CacheControl != nil {
		t.Errorf("input mutated: %+v", origResults)
	}
}

// TestMarkMessagesCachePrefix_onlyMarksLastMessage proves earlier messages in
// a growing history are never touched — only the single last message ever
// carries the moving breakpoint.
func TestMarkMessagesCachePrefix_onlyMarksLastMessage(t *testing.T) {
	in := []Message{
		{Role: "user", Content: "first"},
		{Role: "assistant", Content: "second"},
	}
	out := MarkMessagesCachePrefix(in)
	if out[0].Content != "first" {
		t.Errorf("non-last message changed: %#v", out[0].Content)
	}
	if _, ok := out[1].Content.([]messageContentBlock); !ok {
		t.Errorf("last message not marked: %#v", out[1].Content)
	}
}

// TestMarkMessagesCachePrefix_unsupportedContentIsFailSafe proves a content
// shape this client never assigns to history (e.g. a raw map, not a slice of
// ToolResultBlock/json.RawMessage) is reported ineligible rather than
// guessed at or panicking.
func TestMarkMessagesCachePrefix_unsupportedContentIsFailSafe(t *testing.T) {
	in := []Message{{Role: "user", Content: map[string]any{"unexpected": true}}}
	out := MarkMessagesCachePrefix(in)
	if !reflect.DeepEqual(out, in) {
		t.Errorf("got %#v, want unchanged %#v", out, in)
	}
}
