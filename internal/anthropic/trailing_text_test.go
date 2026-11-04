package anthropic

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestWithTrailingUserText(t *testing.T) {
	tests := []struct {
		name       string
		content    any
		wantBlocks int
		wantLen    int
	}{
		{name: "string content", content: "hi", wantBlocks: 2, wantLen: 1},
		{name: "tool results", content: []ToolResultBlock{{Type: "tool_result", ToolUseID: "t1"}}, wantBlocks: 2, wantLen: 1},
		{name: "raw blocks", content: []json.RawMessage{json.RawMessage(`{"type":"text","text":"a"}`)}, wantBlocks: 2, wantLen: 1},
		{name: "unknown shape appends a user turn", content: 42, wantBlocks: 0, wantLen: 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := []Message{{Role: "user", Content: tt.content}}
			got, err := WithTrailingUserText(input, "ask")
			if err != nil {
				t.Fatalf("WithTrailingUserText: %v", err)
			}
			if len(got) != tt.wantLen {
				t.Fatalf("len = %d, want %d", len(got), tt.wantLen)
			}
			last := got[len(got)-1]
			if last.Role != "user" {
				t.Errorf("last role = %q, want user", last.Role)
			}
			raw, err := json.Marshal(last.Content)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(raw), "ask") {
				t.Errorf("last content = %s, want the appended text", raw)
			}
			if tt.wantBlocks > 0 {
				blocks, ok := last.Content.([]json.RawMessage)
				if !ok || len(blocks) != tt.wantBlocks {
					t.Errorf("content = %#v, want %d raw blocks", last.Content, tt.wantBlocks)
				}
			}
			original, err := json.Marshal(input[0].Content)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(string(original), "ask") {
				t.Errorf("input was mutated: %s", original)
			}
		})
	}
}
