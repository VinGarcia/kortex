package tools

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/mcp"
)

func TestMCPToolDef(t *testing.T) {
	tests := []struct {
		desc string
		tool mcp.Tool
		want anthropic.ToolDef
	}{
		{
			desc: "prefixes the name and copies description/schema",
			tool: mcp.Tool{
				Name:        "sessions_send",
				Description: "Send a message to a session.",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"message":{"type":"string"}},"required":["message"]}`),
			},
			want: anthropic.ToolDef{
				Name:        "mcp__openclaw__sessions_send",
				Description: "Send a message to a session.",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"message":{"type":"string"}},"required":["message"]}`),
			},
		},
		{
			desc: "empty tool still gets prefixed",
			tool: mcp.Tool{},
			want: anthropic.ToolDef{Name: "mcp__openclaw__"},
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			got := MCPToolDef(test.tool)
			if !reflect.DeepEqual(got, test.want) {
				t.Errorf("MCPToolDef = %+v, want %+v", got, test.want)
			}
		})
	}
}

func TestMCPToolDefs(t *testing.T) {
	tools := []mcp.Tool{
		{Name: "memory_search", Description: "search"},
		{Name: "memory_get", Description: "get"},
	}
	got := MCPToolDefs(tools)
	want := []anthropic.ToolDef{
		{Name: "mcp__openclaw__memory_search", Description: "search"},
		{Name: "mcp__openclaw__memory_get", Description: "get"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("MCPToolDefs = %+v, want %+v", got, want)
	}
}

func TestMCPToolDefs_empty(t *testing.T) {
	got := MCPToolDefs(nil)
	if len(got) != 0 {
		t.Errorf("MCPToolDefs(nil) = %+v, want empty", got)
	}
}

func TestMCPToolResultBlock(t *testing.T) {
	tests := []struct {
		desc   string
		toolID string
		result mcp.CallToolResult
		want   anthropic.ToolResultBlock
	}{
		{
			desc:   "single text block, success",
			toolID: "toolu_1",
			result: mcp.CallToolResult{
				Content: []mcp.Content{{Type: "text", Text: `{"status":"missing"}`}},
				IsError: false,
			},
			want: anthropic.NewToolResultBlock("toolu_1", `{"status":"missing"}`, false),
		},
		{
			desc:   "single text block, tool-level error",
			toolID: "toolu_2",
			result: mcp.CallToolResult{
				Content: []mcp.Content{{Type: "text", Text: "boom: invalid argument"}},
				IsError: true,
			},
			want: anthropic.NewToolResultBlock("toolu_2", "boom: invalid argument", true),
		},
		{
			desc:   "multiple blocks join with newline",
			toolID: "toolu_3",
			result: mcp.CallToolResult{
				Content: []mcp.Content{{Type: "text", Text: "line one"}, {Type: "text", Text: "line two"}},
			},
			want: anthropic.NewToolResultBlock("toolu_3", "line one\nline two", false),
		},
		{
			desc:   "no content blocks",
			toolID: "toolu_4",
			result: mcp.CallToolResult{},
			want:   anthropic.NewToolResultBlock("toolu_4", "", false),
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			got := MCPToolResultBlock(test.toolID, test.result)
			if got != test.want {
				t.Errorf("MCPToolResultBlock = %+v, want %+v", got, test.want)
			}
		})
	}
}
