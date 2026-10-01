// Package tools implements the F3b native builtins the kortex tool-loop
// executes itself (bash, read, write, edit), plus the Dispatcher that maps
// a tool_use block to the right builtin and produces its tool_result.
package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
)

// DefaultBashTimeoutSeconds bounds a bash call when the tool_use input does
// not set timeout_seconds: long enough for a typical cron/git/script
// command, short enough that one runaway command can't block the whole
// tool-loop turn indefinitely.
const DefaultBashTimeoutSeconds = 120

// bashMaxOutputBytes caps the combined stdout+stderr returned in the
// tool_result. The cap keeps a runaway command (e.g. one that floods
// stdout) from ballooning the next request's token cost; real shell output
// for the targeted workloads (crons, git, scripts) is a few KB.
const bashMaxOutputBytes = 256 << 10

// BashInput is the bash tool's input_schema, decoded from a tool_use
// block's Input.
type BashInput struct {
	// Command is the shell command to run via `sh -c`.
	Command string `json:"command"`
	// Cwd is the working directory; empty means the kortex process's cwd.
	Cwd string `json:"cwd,omitempty"`
	// TimeoutSeconds bounds the command; 0 means DefaultBashTimeoutSeconds.
	TimeoutSeconds int `json:"timeout_seconds,omitempty"`
}

// BashDef is the bash tool declaration for the Messages API "tools" field.
var BashDef = mustToolDef(
	"bash",
	"Run a shell command via `sh -c` and return its combined stdout+stderr and exit status. Optionally set a working directory and a timeout.",
	map[string]any{
		"type": "object",
		"properties": map[string]any{
			"command":         map[string]any{"type": "string", "description": "The shell command to execute."},
			"cwd":             map[string]any{"type": "string", "description": "Working directory for the command; defaults to the kortex process's cwd."},
			"timeout_seconds": map[string]any{"type": "integer", "description": "Seconds to allow the command to run before it is killed; defaults to 120."},
		},
		"required": []string{"command"},
	},
)

// RunBash executes input.Command and returns the tool_result content plus
// whether it counts as an error turn (is_error). A nonzero exit status, a
// timeout, or a failure to start the command are all reported as
// tool_result content (never a Go error) — the model needs to see the
// failure to react to it, matching the F3a spike's convention of returning
// failures as a is_error:true tool_result rather than aborting the loop.
func RunBash(ctx context.Context, input BashInput) (output string, isError bool) {
	if input.Command == "" {
		return "error: command is required", true
	}
	timeout := time.Duration(input.TimeoutSeconds) * time.Second
	if input.TimeoutSeconds <= 0 {
		timeout = DefaultBashTimeoutSeconds * time.Second
	}
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(runCtx, "sh", "-c", input.Command)
	if input.Cwd != "" {
		cmd.Dir = input.Cwd
	}
	var combined bytes.Buffer
	cmd.Stdout = &combined
	cmd.Stderr = &combined

	runErr := cmd.Run()

	out, truncated := truncateUTF8(combined.Bytes(), bashMaxOutputBytes)
	result := string(out)
	if truncated {
		result += fmt.Sprintf("\n[output truncated at %d bytes]", bashMaxOutputBytes)
	}

	if errors.Is(runCtx.Err(), context.DeadlineExceeded) {
		return result + fmt.Sprintf("\n[command timed out after %ds]", int(timeout.Seconds())), true
	}
	if runErr != nil {
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			return result + fmt.Sprintf("\n[exit status %d]", exitErr.ExitCode()), true
		}
		return result + fmt.Sprintf("\n[failed to run command: %v]", runErr), true
	}
	return result, false
}

// mustToolDef builds an anthropic.ToolDef from a Go value input schema,
// panicking on a marshal failure — schemas here are small static literals,
// so a failure can only be a programmer error caught immediately at package
// init.
func mustToolDef(name string, description string, inputSchema map[string]any) anthropic.ToolDef {
	raw, err := json.Marshal(inputSchema)
	if err != nil {
		panic(fmt.Sprintf("tools: building schema for %s: %v", name, err))
	}
	return anthropic.ToolDef{Name: name, Description: description, InputSchema: raw}
}
