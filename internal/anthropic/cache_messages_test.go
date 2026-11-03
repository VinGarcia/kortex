package anthropic

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// bigEnoughForFloor clears every model's cacheable floor (opus/sonnet/
// fable's 1024 tokens ~4096 bytes, haiku's 4096 tokens ~16384 bytes) on its
// own, so a test proving marking happens doesn't have to reason about exact
// byte/token math.
var bigEnoughForFloor = strings.Repeat("prompt cache history content. ", 1000) // ~30k bytes

func TestMarkMessagesCachePrefix_emptyHistory(t *testing.T) {
	if got := markMessagesCachePrefix("claude-opus-4-8", nil); got != nil {
		t.Errorf("empty history: got %#v, want nil unchanged", got)
	}
}

func TestMarkMessagesCachePrefix_stringContent(t *testing.T) {
	in := []Message{{Role: "user", Content: bigEnoughForFloor}}
	out := markMessagesCachePrefix("claude-opus-4-8", in)

	if in[0].Content != bigEnoughForFloor {
		t.Fatalf("input mutated: %+v", in[0])
	}
	blocks, ok := out[0].Content.([]messageContentBlock)
	if !ok || len(blocks) != 1 {
		t.Fatalf("content = %#v, want one messageContentBlock", out[0].Content)
	}
	if blocks[0].Type != "text" || blocks[0].Text != bigEnoughForFloor {
		t.Errorf("block = %+v", blocks[0])
	}
	if blocks[0].CacheControl == nil || blocks[0].CacheControl.Type != "ephemeral" {
		t.Errorf("CacheControl = %+v, want ephemeral", blocks[0].CacheControl)
	}
}

// TestMarkMessagesCachePrefix_belowFloorIsFailSafe proves a messages array
// whose estimated size sits under the calling model's cacheable floor is left
// unmarked, mirroring markStableSystemPrefix's floor on the system prefix:
// the API would ignore a breakpoint there anyway, so marking below it is
// pointless.
func TestMarkMessagesCachePrefix_belowFloorIsFailSafe(t *testing.T) {
	in := []Message{{Role: "user", Content: "hello"}}
	out := markMessagesCachePrefix("claude-opus-4-8", in)
	if !reflect.DeepEqual(out, in) {
		t.Errorf("got %#v, want unchanged %#v", out, in)
	}
}

// TestMarkMessagesCachePrefix_floorIsPerModel proves the floor check applies
// minCacheableTokens per model, exactly like the system prefix: the same
// messages array marks on opus (floor 1024 tokens) but stays unmarked on
// haiku (floor 4096 tokens).
func TestMarkMessagesCachePrefix_floorIsPerModel(t *testing.T) {
	// ~1200 tokens: clears the 1024 floor, stays under haiku's 4096 floor.
	midContent := strings.Repeat("system ", 700)
	in := []Message{{Role: "user", Content: midContent}}

	if out := markMessagesCachePrefix("claude-opus-4-8", in); out[0].Content == midContent {
		t.Errorf("opus (floor 1024): want marked, got unchanged")
	}
	if out := markMessagesCachePrefix("claude-haiku-4-5-20251001", in); out[0].Content != midContent {
		t.Errorf("haiku (floor 4096): want unmarked below its floor, got %#v", out[0].Content)
	}
}

// TestMarkMessagesCachePrefix_floorAppliesToWholeArray proves the floor is
// evaluated over the WHOLE messages array (like estimatedSystemTokens sums
// every system block), not just the message being marked: an early bulky
// message clears the floor even though the last (marked) message is tiny.
func TestMarkMessagesCachePrefix_floorAppliesToWholeArray(t *testing.T) {
	in := []Message{
		{Role: "user", Content: bigEnoughForFloor},
		{Role: "assistant", Content: "tiny"},
	}
	out := markMessagesCachePrefix("claude-opus-4-8", in)
	if _, ok := out[1].Content.([]messageContentBlock); !ok {
		t.Errorf("last message not marked despite the array clearing the floor: %#v", out[1].Content)
	}
}

func TestMarkMessagesCachePrefix_toolResultBlocks(t *testing.T) {
	in := []Message{{Role: "user", Content: []ToolResultBlock{
		NewToolResultBlock("tool_1", bigEnoughForFloor, false),
		NewToolResultBlock("tool_2", "second", false),
	}}}
	out := markMessagesCachePrefix("claude-opus-4-8", in)

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

// TestMarkMessagesCachePrefix_rawAssistantBlocks proves an echoed assistant
// turn (Message.Content == []json.RawMessage, the shape MessageResponse.
// RawContent produces — session.textOnlyRedraft's canonical history always
// ends in exactly this shape, since it is the core's own final draft turn
// echoed back) is eligible: the breakpoint lands on the last raw block,
// additively (every pre-existing field, including an unmodeled one like
// tool_use's "caller", survives), and only the last block of the slice is
// touched.
func TestMarkMessagesCachePrefix_rawAssistantBlocks(t *testing.T) {
	first := json.RawMessage(`{"type":"text","text":"` + bigEnoughForFloor + `"}`)
	second := json.RawMessage(`{"type":"tool_use","id":"toolu_1","name":"bash","input":{"command":"pwd"},"caller":{"type":"direct"}}`)
	in := []Message{{Role: "assistant", Content: []json.RawMessage{first, second}}}
	out := markMessagesCachePrefix("claude-opus-4-8", in)

	blocks, ok := out[0].Content.([]json.RawMessage)
	if !ok || len(blocks) != 2 {
		t.Fatalf("content = %#v, want two raw blocks", out[0].Content)
	}
	if string(blocks[0]) != string(first) {
		t.Errorf("first block changed: %s", blocks[0])
	}
	var marked map[string]any
	if err := json.Unmarshal(blocks[1], &marked); err != nil {
		t.Fatalf("second block did not decode: %v", err)
	}
	if cc, _ := marked["cache_control"].(map[string]any); cc == nil || cc["type"] != "ephemeral" {
		t.Errorf("second block cache_control = %v, want ephemeral", marked["cache_control"])
	}
	if marked["type"] != "tool_use" || marked["id"] != "toolu_1" || marked["name"] != "bash" {
		t.Errorf("marked block lost a field: %v", marked)
	}
	if caller, _ := marked["caller"].(map[string]any); caller == nil || caller["type"] != "direct" {
		t.Errorf("marked block lost the unmodeled caller field: %v", marked)
	}

	// The input slice itself must stay untouched (no aliasing).
	origBlocks := in[0].Content.([]json.RawMessage)
	if strings.Contains(string(origBlocks[1]), "cache_control") {
		t.Errorf("input mutated: %s", origBlocks[1])
	}
}

func TestMarkMessagesCachePrefix_onlyMarksLastMessage(t *testing.T) {
	in := []Message{
		{Role: "user", Content: "first " + bigEnoughForFloor},
		{Role: "assistant", Content: "second"},
	}
	out := markMessagesCachePrefix("claude-opus-4-8", in)
	if out[0].Content != "first "+bigEnoughForFloor {
		t.Errorf("non-last message changed: %#v", out[0].Content)
	}
	if _, ok := out[1].Content.([]messageContentBlock); !ok {
		t.Errorf("last message not marked: %#v", out[1].Content)
	}
}

func TestMarkMessagesCachePrefix_unsupportedContentIsFailSafe(t *testing.T) {
	in := []Message{{Role: "user", Content: map[string]any{"unexpected": true, "padding": bigEnoughForFloor}}}
	out := markMessagesCachePrefix("claude-opus-4-8", in)
	if !reflect.DeepEqual(out, in) {
		t.Errorf("got %#v, want unchanged %#v", out, in)
	}
}

// The tests below call markLastEligibleBlock and addCacheControlToRawBlock
// directly, bypassing markMessagesCachePrefix's floor gate, so each
// eligibility/fail-safe branch is proven in isolation rather than conflated
// with "the array was too small to mark anyway".

func TestMarkLastEligibleBlock_emptyStringIsIneligible(t *testing.T) {
	if _, ok := markLastEligibleBlock(Message{Role: "user", Content: ""}); ok {
		t.Error("empty string content should be ineligible")
	}
}

func TestMarkLastEligibleBlock_emptyToolResultBlocksIsIneligible(t *testing.T) {
	if _, ok := markLastEligibleBlock(Message{Role: "user", Content: []ToolResultBlock{}}); ok {
		t.Error("empty tool_result slice should be ineligible")
	}
}

func TestMarkLastEligibleBlock_emptyRawBlocksIsIneligible(t *testing.T) {
	if _, ok := markLastEligibleBlock(Message{Role: "assistant", Content: []json.RawMessage{}}); ok {
		t.Error("empty raw-block slice should be ineligible")
	}
}

func TestMarkLastEligibleBlock_malformedRawBlockIsIneligible(t *testing.T) {
	in := Message{Role: "assistant", Content: []json.RawMessage{json.RawMessage("not json")}}
	_, ok := markLastEligibleBlock(in)
	if ok {
		t.Error("a raw block that is not valid JSON should be ineligible")
	}
}

func TestMarkLastEligibleBlock_nilContentIsIneligible(t *testing.T) {
	if _, ok := markLastEligibleBlock(Message{Role: "user", Content: nil}); ok {
		t.Error("nil content should be ineligible")
	}
}

// TestMarkLastEligibleBlock_thinkingRawBlockIsIneligible proves a raw block
// whose "type" is "thinking" or "redacted_thinking" is never marked: the
// Messages API rejects a cache_control marker on those block types outright
// (a 400, not a no-op), unlike the rest of the API's broader eligible set
// (text, tool_use, image, document) that addCacheControlToRawBlock marks
// without needing to know the specific type.
func TestMarkLastEligibleBlock_thinkingRawBlockIsIneligible(t *testing.T) {
	for _, blockType := range []string{"thinking", "redacted_thinking"} {
		t.Run(blockType, func(t *testing.T) {
			in := Message{Role: "assistant", Content: []json.RawMessage{
				json.RawMessage(`{"type":"` + blockType + `","thinking":"reasoning...","signature":"sig"}`),
			}}
			if _, ok := markLastEligibleBlock(in); ok {
				t.Errorf("a %q raw block should be ineligible for a cache breakpoint", blockType)
			}
		})
	}
}

// TestMarkLastEligibleBlock_nullRawBlockIsIneligible proves the literal JSON
// value "null" as a raw block (decodes successfully into a nil
// map[string]json.RawMessage, not a decode error) is reported ineligible
// rather than panicking on the nil-map field assignment inside
// addCacheControlToRawBlock.
func TestMarkLastEligibleBlock_nullRawBlockIsIneligible(t *testing.T) {
	in := Message{Role: "assistant", Content: []json.RawMessage{json.RawMessage("null")}}
	if _, ok := markLastEligibleBlock(in); ok {
		t.Error("a null raw block should be ineligible")
	}
}

func TestAddCacheControlToRawBlock(t *testing.T) {
	tests := []struct {
		desc   string
		raw    json.RawMessage
		wantOK bool
	}{
		{desc: "JSON object gets the field added", raw: json.RawMessage(`{"type":"text","text":"hi"}`), wantOK: true},
		{desc: "a JSON string (not an object) is ineligible", raw: json.RawMessage(`"just a string"`), wantOK: false},
		{desc: "invalid JSON is ineligible", raw: json.RawMessage(`not json`), wantOK: false},
		{desc: "JSON null is ineligible", raw: json.RawMessage(`null`), wantOK: false},
		{desc: "thinking block is ineligible", raw: json.RawMessage(`{"type":"thinking","thinking":"x","signature":"s"}`), wantOK: false},
		{desc: "redacted_thinking block is ineligible", raw: json.RawMessage(`{"type":"redacted_thinking","data":"x"}`), wantOK: false},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			out, ok := addCacheControlToRawBlock(test.raw)
			if ok != test.wantOK {
				t.Fatalf("ok = %v, want %v", ok, test.wantOK)
			}
			if !ok {
				return
			}
			var decoded map[string]any
			if err := json.Unmarshal(out, &decoded); err != nil {
				t.Fatalf("marked block did not decode: %v", err)
			}
			cc, _ := decoded["cache_control"].(map[string]any)
			if cc == nil || cc["type"] != "ephemeral" {
				t.Errorf("cache_control = %v, want ephemeral", decoded["cache_control"])
			}
			if decoded["type"] != "text" || decoded["text"] != "hi" {
				t.Errorf("marked block lost a pre-existing field: %v", decoded)
			}
		})
	}
}
