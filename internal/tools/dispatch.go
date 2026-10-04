package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/mcp"
)

// Definitions returns the ToolDef declarations for every builtin this
// package implements, in the order the tool-loop should declare them in a
// request's "tools" field.
func Definitions() []anthropic.ToolDef {
	return []anthropic.ToolDef{BashDef, ReadDef, WriteDef, EditDef}
}

// mcpCaller is the slice of the MCP client the dispatcher needs: invoking a
// tool by its bare (unprefixed) name. The concrete *mcp.Client satisfies it;
// the interface keeps the dispatcher testable without a live HTTP server.
type mcpCaller interface {
	CallTool(ctx context.Context, name string, arguments map[string]any) (mcp.CallToolResult, error)
}

// Dispatcher executes tool_use content blocks by name, routing the native
// builtins (bash, read, write, edit) to their executors and the
// mcp__openclaw__* tools to the OpenClaw loopback server, producing the
// matching tool_result either way.
type Dispatcher struct {
	// mcp is the loopback client for mcp__openclaw__* tools; nil when kortex
	// was spawned without an --mcp-config (builtins-only).
	mcp mcpCaller
	// mcpTools is the tools/list catalog ToolDefs() declares to the model
	// alongside the builtins; empty when mcp is nil.
	mcpTools []mcp.Tool
}

// NewDispatcher builds a builtins-only Dispatcher. Builtins read their
// configuration (cwd, timeout) from the tool_use input itself, so it holds no
// other state.
func NewDispatcher() *Dispatcher {
	return &Dispatcher{}
}

// NewDispatcherWithMCP builds a Dispatcher that also exposes and executes the
// gateway's MCP tools. tools is the tools/list catalog (bare names) the
// handshake returned; client invokes them. A nil client (or empty catalog)
// degrades to the builtins-only behavior of NewDispatcher.
func NewDispatcherWithMCP(client *mcp.Client, tools []mcp.Tool) *Dispatcher {
	if client == nil {
		return NewDispatcher()
	}
	return &Dispatcher{mcp: client, mcpTools: tools}
}

// ToolDefs returns every tool the loop declares in a request's "tools" field:
// the native builtins first, then the gateway's MCP tools (prefixed for the
// model). Builtins-only when no MCP client is wired.
func (d *Dispatcher) ToolDefs() []anthropic.ToolDef {
	return append(Definitions(), MCPToolDefs(d.mcpTools)...)
}

// Execute runs the builtin named by block.Name with block.Input and returns
// the tool_result block keyed to block.ID. An unknown tool name, or input
// that fails to decode against the builtin's schema, comes back as an
// is_error:true tool_result rather than a Go error — same convention as the
// builtins themselves — so the tool-loop can always feed the result
// straight back to the model without a special failure path.
func (d *Dispatcher) Execute(ctx context.Context, block anthropic.ContentBlock) anthropic.ToolResultBlock {
	if block.Type != "tool_use" {
		return anthropic.NewToolResultBlock(block.ID, fmt.Sprintf("error: dispatcher given a %q block, want tool_use", block.Type), true)
	}

	if strings.HasPrefix(block.Name, MCPToolPrefix) {
		return d.executeMCP(ctx, block)
	}

	switch block.Name {
	case "bash":
		var input BashInput
		if err := json.Unmarshal(block.Input, &input); err != nil {
			return errorResult(block.ID, "bash", err)
		}
		output, isError := RunBash(ctx, input)
		return anthropic.NewToolResultBlock(block.ID, output, isError)

	case "read":
		var input ReadInput
		if err := json.Unmarshal(block.Input, &input); err != nil {
			return errorResult(block.ID, "read", err)
		}
		output, isError := RunRead(input)
		return anthropic.NewToolResultBlock(block.ID, output, isError)

	case "write":
		var input WriteInput
		if err := json.Unmarshal(block.Input, &input); err != nil {
			return errorResult(block.ID, "write", err)
		}
		output, isError := RunWrite(input)
		return anthropic.NewToolResultBlock(block.ID, output, isError)

	case "edit":
		var input EditInput
		if err := json.Unmarshal(block.Input, &input); err != nil {
			return errorResult(block.ID, "edit", err)
		}
		output, isError := RunEdit(input)
		return anthropic.NewToolResultBlock(block.ID, output, isError)

	default:
		return anthropic.NewToolResultBlock(block.ID, fmt.Sprintf("error: unknown tool %q", block.Name), true)
	}
}

// executeMCP routes an mcp__openclaw__* tool_use to the loopback server: it
// strips the prefix back to the bare name the server's catalog uses, decodes
// the input object into the arguments map CallTool expects, and maps the
// result (or a transport failure) to an is_error tool_result the loop can feed
// straight back to the model.
func (d *Dispatcher) executeMCP(ctx context.Context, block anthropic.ContentBlock) anthropic.ToolResultBlock {
	if d.mcp == nil {
		return anthropic.NewToolResultBlock(block.ID, fmt.Sprintf("error: MCP tool %q requested but no MCP client is configured", block.Name), true)
	}
	// An absent/null input is a valid no-argument call; only a present-but-
	// malformed input is an error.
	arguments := map[string]any{}
	if len(block.Input) > 0 {
		if err := json.Unmarshal(block.Input, &arguments); err != nil {
			return errorResult(block.ID, block.Name, err)
		}
	}
	name := strings.TrimPrefix(block.Name, MCPToolPrefix)
	result, err := d.mcp.CallTool(ctx, name, arguments)
	if err != nil {
		return anthropic.NewToolResultBlock(block.ID, fmt.Sprintf("error: calling MCP tool %q: %v", name, err), true)
	}
	return MCPToolResultBlock(block.ID, result)
}

// errorResult builds the tool_result for a tool_use whose input JSON failed
// to decode against the named builtin's input struct (e.g. the model sent a
// field of the wrong type).
func errorResult(toolUseID string, toolName string, decodeErr error) anthropic.ToolResultBlock {
	return anthropic.NewToolResultBlock(toolUseID, fmt.Sprintf("error: decoding %s input: %v", toolName, decodeErr), true)
}
