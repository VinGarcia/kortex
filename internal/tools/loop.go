package tools

import (
	"context"
	"fmt"
	"slices"

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
	// History is the prior conversation this turn continues, in the exact
	// anthropic.Message shape a previous LoopResult returned (assistant turns
	// echoed verbatim via raw content, user turns carrying tool_result blocks).
	// Nil for the first turn of a session. RunLoop never mutates it: the new
	// user message and this turn's exchanges are appended to a clone.
	History []anthropic.Message
	// MaxTurns bounds tool_use round trips; 0 means DefaultMaxTurns.
	MaxTurns int
	// Emitter, when non-nil, receives the loop's turns as stream-json (NDJSON)
	// events: system/init, one assistant event per turn, one user event per
	// tool_result batch, and the terminal result. Nil disables emission, in
	// which case RunLoop behaves exactly as it did before F3d.
	Emitter *StreamEmitter
	// DeferFinal, when true, suppresses the emission of the FINAL turn's
	// assistant event and the terminal result: RunLoop returns the resolved
	// LoopResult without putting the final draft on the wire, leaving the caller
	// (the active superego loop) to govern the draft and re-emit the delivered
	// text via StreamEmitter.EmitFinalTurn. Intermediate tool_use assistant
	// events and their tool_result turns are still emitted live. With DeferFinal
	// false the emission is byte-for-byte what it was before this field existed.
	DeferFinal bool
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
	// Messages is the full canonical conversation after this turn: the input
	// History plus the new user message, every intermediate assistant/tool_result
	// exchange, and the final assistant turn. Feed it back as the next turn's
	// History to continue the conversation (in-process or persisted across a
	// restart).
	Messages []anthropic.Message
	// Usage is this turn's usage summed across every round trip — the same total
	// the terminal result reports. A DeferFinal caller needs it to bill the round
	// on its own re-emitted result (EmitFinalTurn), since RunLoop did not emit
	// the result itself.
	Usage anthropic.Usage
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

	messages := append(slices.Clone(req.History), anthropic.Message{Role: "user", Content: req.UserMessage})
	tools := dispatcher.ToolDefs()

	// OpenClaw's native tool-authority capture reads the tool names kortex
	// declares to the model from system/init's "tools" field; emit the same
	// set the loop declares in each request's Tools so the gateway sees the
	// runtime's real tool list rather than rejecting an absent one.
	toolNames := make([]string, len(tools))
	for i, def := range tools {
		toolNames[i] = def.Name
	}
	if err := req.Emitter.systemInit(toolNames); err != nil {
		return LoopResult{}, err
	}

	// The terminal result reports usage summed across every turn, not just the
	// last one: OpenClaw bills the round on the result's usage, so a multi-turn
	// tool loop that reported only the final turn would under-count the round.
	var totalUsage anthropic.Usage
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
		totalUsage = totalUsage.Add(resp.Usage)

		toolUses := resp.ToolUseBlocks()
		// stop_reason other than tool_use — or tool_use with no decodable block,
		// which we treat as resolved rather than looping forever on the same
		// state — ends the loop with the terminal result event.
		isFinal := resp.StopReason != "tool_use" || len(toolUses) == 0

		// The final turn's assistant event is deferred when DeferFinal is set (the
		// caller re-emits the governed text); every intermediate tool_use assistant
		// event is always emitted live. With DeferFinal false this fires on every
		// turn exactly as before.
		if !(isFinal && req.DeferFinal) {
			if err := req.Emitter.assistant(resp); err != nil {
				return LoopResult{}, err
			}
		}

		if isFinal {
			messages = append(messages, anthropic.Message{Role: "assistant", Content: resp.RawContent()})
			result := LoopResult{FinalText: resp.Text, StopReason: resp.StopReason, Turns: turn, Messages: messages, Usage: totalUsage}
			// The terminal result is deferred alongside the final assistant event:
			// the caller governs the draft and emits both via EmitFinalTurn.
			if !req.DeferFinal {
				if err := req.Emitter.result(result, totalUsage); err != nil {
					return LoopResult{}, err
				}
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
