package tools

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/mcp"
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

// fakeMCPCaller records the last call and returns a scripted result/error,
// standing in for *mcp.Client so the dispatcher's MCP routing is tested
// without a live loopback server.
type fakeMCPCaller struct {
	result  mcp.CallToolResult
	err     error
	gotName string
	gotArgs map[string]any
	calledN int
}

func (f *fakeMCPCaller) CallTool(ctx context.Context, name string, arguments map[string]any) (mcp.CallToolResult, error) {
	f.calledN++
	f.gotName = name
	f.gotArgs = arguments
	return f.result, f.err
}

func TestDispatcher_ToolDefs(t *testing.T) {
	t.Run("builtins only when no MCP", func(t *testing.T) {
		got := NewDispatcher().ToolDefs()
		if !reflect.DeepEqual(got, Definitions()) {
			t.Errorf("ToolDefs = %+v, want the builtins %+v", got, Definitions())
		}
	})
	t.Run("builtins then prefixed MCP tools", func(t *testing.T) {
		d := &Dispatcher{mcp: &fakeMCPCaller{}, mcpTools: []mcp.Tool{{Name: "sessions_send"}}}
		got := d.ToolDefs()
		want := append(Definitions(), anthropic.ToolDef{Name: "mcp__openclaw__sessions_send"})
		if !reflect.DeepEqual(got, want) {
			t.Errorf("ToolDefs = %+v, want %+v", got, want)
		}
	})
}

func TestDispatcher_ExecuteMCP(t *testing.T) {
	t.Run("routes to CallTool with the prefix stripped and args decoded", func(t *testing.T) {
		fake := &fakeMCPCaller{result: mcp.CallToolResult{Content: []mcp.Content{{Type: "text", Text: "sent"}}}}
		d := &Dispatcher{mcp: fake}
		got := d.Execute(context.Background(), toolUseBlock("toolu_1", "mcp__openclaw__sessions_send", `{"message":"hi"}`))

		if fake.gotName != "sessions_send" {
			t.Errorf("CallTool name = %q, want the bare %q", fake.gotName, "sessions_send")
		}
		if !reflect.DeepEqual(fake.gotArgs, map[string]any{"message": "hi"}) {
			t.Errorf("CallTool args = %+v, want the decoded input object", fake.gotArgs)
		}
		if got.IsError || got.Content != "sent" {
			t.Errorf("result = %+v, want the tool's text with is_error false", got)
		}
	})
	t.Run("tool-level error is forwarded as is_error", func(t *testing.T) {
		fake := &fakeMCPCaller{result: mcp.CallToolResult{Content: []mcp.Content{{Type: "text", Text: "bad arg"}}, IsError: true}}
		got := (&Dispatcher{mcp: fake}).Execute(context.Background(), toolUseBlock("toolu_2", "mcp__openclaw__memory_get", `{}`))
		if !got.IsError || got.Content != "bad arg" {
			t.Errorf("result = %+v, want is_error with the tool's message", got)
		}
	})
	t.Run("transport error becomes an is_error result, not a panic", func(t *testing.T) {
		fake := &fakeMCPCaller{err: errors.New("connection refused")}
		got := (&Dispatcher{mcp: fake}).Execute(context.Background(), toolUseBlock("toolu_3", "mcp__openclaw__memory_get", `{}`))
		if !got.IsError || !strings.Contains(got.Content, "connection refused") {
			t.Errorf("result = %+v, want is_error carrying the transport error", got)
		}
	})
	t.Run("malformed input is an is_error result and never calls the server", func(t *testing.T) {
		fake := &fakeMCPCaller{}
		got := (&Dispatcher{mcp: fake}).Execute(context.Background(), toolUseBlock("toolu_4", "mcp__openclaw__memory_get", `not json`))
		if !got.IsError || !strings.Contains(got.Content, "decoding") {
			t.Errorf("result = %+v, want a decode is_error", got)
		}
		if fake.calledN != 0 {
			t.Errorf("CallTool invoked %d times on malformed input, want 0", fake.calledN)
		}
	})
	t.Run("MCP tool with no client configured is an is_error result", func(t *testing.T) {
		got := NewDispatcher().Execute(context.Background(), toolUseBlock("toolu_5", "mcp__openclaw__sessions_send", `{}`))
		if !got.IsError || !strings.Contains(got.Content, "no MCP client") {
			t.Errorf("result = %+v, want an is_error about the missing client", got)
		}
	})
}
