package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/vingarcia/kortex/internal/anthropic"
)

// Definitions returns the ToolDef declarations for every builtin this
// package implements, in the order the tool-loop should declare them in a
// request's "tools" field.
func Definitions() []anthropic.ToolDef {
	return []anthropic.ToolDef{BashDef, ReadDef, WriteDef, EditDef}
}

// Dispatcher executes tool_use content blocks by name against the native
// builtins (bash, read, write, edit) and produces the matching tool_result.
type Dispatcher struct{}

// NewDispatcher builds a Dispatcher. It holds no state today; builtins read
// their configuration (cwd, timeout) from the tool_use input itself, not
// from the Dispatcher.
func NewDispatcher() *Dispatcher {
	return &Dispatcher{}
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

// errorResult builds the tool_result for a tool_use whose input JSON failed
// to decode against the named builtin's input struct (e.g. the model sent a
// field of the wrong type).
func errorResult(toolUseID string, toolName string, decodeErr error) anthropic.ToolResultBlock {
	return anthropic.NewToolResultBlock(toolUseID, fmt.Sprintf("error: decoding %s input: %v", toolName, decodeErr), true)
}
