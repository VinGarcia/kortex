package tools

import (
	"context"
	"fmt"

	"github.com/vingarcia/kortex/internal/anthropic"
)

// DefaultMaxTurns bounds how many tool_use round trips RunLoop makes before
// giving up. This is a safety cap against a runaway loop (a model that
// keeps calling tools without ever reaching end_turn), not an expected
// depth — most tool-loop turns in the workloads F3b targets (crons, git,
// scripts) resolve in a handful of calls.
const DefaultMaxTurns = 25

// LoopRequest configures one RunLoop call.
type LoopRequest struct {
	Model       string
	System      string
	MaxTokens   int
	UserMessage string
	// MaxTurns bounds tool_use round trips; 0 means DefaultMaxTurns.
	MaxTurns int
	// Emitter, when non-nil, receives the loop's turns as stream-json (NDJSON)
	// events: system/init, one assistant event per turn, one user event per
	// tool_result batch, and the terminal result. Nil disables emission, in
	// which case RunLoop behaves exactly as it did before F3d.
	Emitter *StreamEmitter
}

// LoopResult is the outcome of a completed tool-loop.
type LoopResult struct {
	// FinalText is the assistant's last text response once the loop
	// resolves (stop_reason other than tool_use).
	FinalText string
	// StopReason is the final response's stop_reason.
	StopReason string
	// Turns is how many request/response round trips the loop made
	// (always >= 1).
	Turns int
}

// RunLoop ports the F3a spike's tool_use/tool_result cycle
// (hack/toolloop.sh's run_tool_loop) into kortex proper: it calls the
// Messages API with the native builtins declared in Tools, and for every
// tool_use block the model requests, executes it with dispatcher and sends
// the tool_result back as the next turn's user message, repeating until the
// model stops asking for tools (or MaxTurns is hit). The message array it
// builds follows the exact shape the spike proved the API accepts: user ->
// assistant (echoed verbatim via ContentBlock.Raw, preserving unmodeled
// fields like tool_use.caller) -> user (tool_result) -> ...
func RunLoop(ctx context.Context, client *anthropic.Client, dispatcher *Dispatcher, req LoopRequest) (LoopResult, error) {
	maxTurns := req.MaxTurns
	if maxTurns <= 0 {
		maxTurns = DefaultMaxTurns
	}

	messages := []anthropic.Message{{Role: "user", Content: req.UserMessage}}
	tools := Definitions()

	if err := req.Emitter.systemInit(); err != nil {
		return LoopResult{}, err
	}

	for turn := 1; turn <= maxTurns; turn++ {
		resp, err := client.CreateMessage(ctx, anthropic.MessageRequest{
			Model:     req.Model,
			System:    req.System,
			MaxTokens: req.MaxTokens,
			Messages:  messages,
			Tools:     tools,
		})
		if err != nil {
			return LoopResult{}, fmt.Errorf("tools: tool-loop turn %d: %w", turn, err)
		}
		if err := req.Emitter.assistant(resp); err != nil {
			return LoopResult{}, err
		}

		toolUses := resp.ToolUseBlocks()
		// stop_reason other than tool_use — or tool_use with no decodable block,
		// which we treat as resolved rather than looping forever on the same
		// state — ends the loop with the terminal result event.
		if resp.StopReason != "tool_use" || len(toolUses) == 0 {
			result := LoopResult{FinalText: resp.Text, StopReason: resp.StopReason, Turns: turn}
			if err := req.Emitter.result(result, resp.Usage); err != nil {
				return LoopResult{}, err
			}
			return result, nil
		}

		results := make([]anthropic.ToolResultBlock, len(toolUses))
		for i, block := range toolUses {
			results[i] = dispatcher.Execute(ctx, block)
		}
		if err := req.Emitter.toolResults(results); err != nil {
			return LoopResult{}, err
		}

		messages = append(messages,
			anthropic.Message{Role: "assistant", Content: resp.RawContent()},
			anthropic.Message{Role: "user", Content: results},
		)
	}

	return LoopResult{}, fmt.Errorf("tools: tool-loop exceeded MaxTurns (%d) without reaching a non-tool_use stop_reason", maxTurns)
}
