package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vingarcia/kortex/internal/anthropic"
)

func TestDispatcher_Execute(t *testing.T) {
	dir := t.TempDir()
	readPath := filepath.Join(dir, "read.txt")
	if err := os.WriteFile(readPath, []byte("read me"), 0o644); err != nil {
		t.Fatal(err)
	}
	writePath := filepath.Join(dir, "write.txt")
	editPath := filepath.Join(dir, "edit.txt")
	if err := os.WriteFile(editPath, []byte("hello world"), 0o644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		desc        string
		block       anthropic.ContentBlock
		wantErr     bool
		wantContain string
	}{
		{
			desc:        "bash dispatches to RunBash",
			block:       toolUseBlock("toolu_1", "bash", `{"command":"echo hi"}`),
			wantErr:     false,
			wantContain: "hi",
		},
		{
			desc:        "read dispatches to RunRead",
			block:       toolUseBlock("toolu_2", "read", `{"path":"`+readPath+`"}`),
			wantErr:     false,
			wantContain: "read me",
		},
		{
			desc:        "write dispatches to RunWrite",
			block:       toolUseBlock("toolu_3", "write", `{"path":"`+writePath+`","content":"written"}`),
			wantErr:     false,
			wantContain: writePath,
		},
		{
			desc:        "edit dispatches to RunEdit",
			block:       toolUseBlock("toolu_4", "edit", `{"path":"`+editPath+`","old_string":"world","new_string":"kortex"}`),
			wantErr:     false,
			wantContain: "replaced 1",
		},
		{
			desc:        "unknown tool name is an error result",
			block:       toolUseBlock("toolu_5", "nonexistent", `{}`),
			wantErr:     true,
			wantContain: `unknown tool "nonexistent"`,
		},
		{
			desc:        "malformed input JSON is an error result",
			block:       toolUseBlock("toolu_6", "bash", `not json`),
			wantErr:     true,
			wantContain: "decoding bash input",
		},
		{
			desc:        "non tool_use block is rejected",
			block:       anthropic.ContentBlock{Type: "text", ID: "toolu_7"},
			wantErr:     true,
			wantContain: "want tool_use",
		},
	}

	d := NewDispatcher()
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			result := d.Execute(context.Background(), test.block)
			if result.ToolUseID != test.block.ID {
				t.Errorf("tool_use_id = %q, want %q", result.ToolUseID, test.block.ID)
			}
			if result.IsError != test.wantErr {
				t.Errorf("is_error = %v, want %v (content=%q)", result.IsError, test.wantErr, result.Content)
			}
			if test.wantContain != "" && !strings.Contains(result.Content, test.wantContain) {
				t.Errorf("content = %q, want it to contain %q", result.Content, test.wantContain)
			}
		})
	}
}

func toolUseBlock(id string, name string, inputJSON string) anthropic.ContentBlock {
	return anthropic.ContentBlock{Type: "tool_use", ID: id, Name: name, Input: []byte(inputJSON)}
}
