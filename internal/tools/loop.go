package tools

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
)

// DefaultMaxTurns bounds how many tool_use round trips RunLoop makes before
// giving up. This is a safety cap against a runaway loop (a model that
// keeps calling tools without ever reaching end_turn), not an expected
// depth — most tool-loop turns in the workloads F3b targets (crons, git,
// scripts) resolve in a handful of calls. Raised 25->100 (2026-10-21, Vini):
// 25 was killing legitimate long turns (multi-step implementation in-session)
// recurrently; the kill was a serious recurring bug. 100 keeps the runaway
// guard while giving real deep turns room. A refined finalize-on-budget path
// for the core turn is deferred to a background run.
const DefaultMaxTurns = 100

// DefaultCallTimeout is the per-call deadline RunLoop uses when
// LoopRequest.CallTimeout is 0. The core client carries no HTTP timeout, so a
// single hung CreateMessage would otherwise block the whole turn forever; this
// safety net bounds each call generously enough to let normal long generations
// finish while still catching a truly stuck call. Held equal to the composition
// root's main.defaultLoopCallTimeout (the two are per-layer fallbacks for the
// same bound); raised 120s->300s with it (2026-10-31 incident) now that each
// call is keepalive-bracketed and so can safely outlast the 180s watchdog.
const DefaultCallTimeout = 300 * time.Second

// DefaultKeepaliveInterval is how often RunLoop emits a system/keepalive line
// while a core-model call is in flight, when LoopRequest.KeepaliveInterval is
// 0. It must stay well under OpenClaw's 180s no-output watchdog; 45s matches
// the governor ladder's cadence (session.DefaultGovernKeepaliveInterval, the
// same watchdog — the two share the "beat under the watchdog" concept, so they
// use the same value). Without this bracket a single slow call on a
// large-context session goes silent for its whole duration and the watchdog
// kills the CLI: the 2026-10-31 incident, and what makes a raised CallTimeout
// safe only once paired with the keepalive.
const DefaultKeepaliveInterval = 45 * time.Second

// LoopRequest configures one RunLoop call.
type LoopRequest struct {
	Model     string
	System    string
	MaxTokens int
	// Effort is forwarded to each core-model call as output_config.effort (see
	// anthropic.MessageRequest.Effort); "" lets the API default stand. Bounding
	// effort keeps a thinking model from spending the whole MaxTokens budget on
	// thinking and returning no text.
	Effort      string
	UserMessage string
	// History is the prior conversation this turn continues, in the exact
	// anthropic.Message shape a previous LoopResult returned (assistant turns
	// echoed verbatim via raw content, user turns carrying tool_result blocks).
	// Nil for the first turn of a session. RunLoop never mutates it: the new
	// user message and this turn's exchanges are appended to a clone.
	History []anthropic.Message
	// MaxTurns bounds tool_use round trips; 0 means DefaultMaxTurns.
	MaxTurns int
	// CallTimeout bounds each individual CreateMessage call to the core model
	// inside the loop; 0 means DefaultCallTimeout. MaxTurns caps how many
	// round trips run, but a single hung call blocks the turn indefinitely
	// because the core client has no HTTP timeout, so this per-call deadline is
	// the only bound on one stuck call. Unlike the superego redraft (which has a
	// raw draft to fall back to), the loop has no fallback: a timed-out call
	// surfaces as a fail-closed error return from RunLoop.
	CallTimeout time.Duration
	// KeepaliveInterval is how often a benign system/keepalive line is emitted
	// (via Emitter) while each core-model call inside the loop is in flight; 0
	// means DefaultKeepaliveInterval. A single core call on a large-context
	// session can run for minutes with nothing else on the wire, and OpenClaw
	// kills a CLI silent past its no-output watchdog, so each call is bracketed
	// by keepalives. Ignored when Emitter is nil (there is no wire to keep
	// alive).
	KeepaliveInterval time.Duration
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
	// EphemeralContext, when non-empty, is appended as ONE extra trailing user
	// message to the array sent to the core model on EVERY call in the loop, but
	// is NEVER added to the accumulating/returned Messages. It is ephemeral
	// injected context — the tone digest — recomputed per turn and kept out of
	// canonical history, exactly like the superego's ephemeral tail. Placed last
	// so it is the final item of the core's context; placed in the message tail
	// (never the system prefix) so it never disturbs the cacheable prefix
	// (anti-nonce rule: mutable per-turn content goes at the end of messages).
	EphemeralContext string
	// FinalizeOnBudget changes what happens when the loop reaches MaxTurns while
	// the model is still asking for tools. The default (false) fails closed with
	// an error, which is right for a user turn that has no fallback output. When
	// true, RunLoop instead makes ONE final call with no tools offered, so the
	// round ends gracefully with a deliverable text answer that still sees every
	// tool_result gathered so far — the model is forced to stop acting and
	// answer. This is the bounded superego-redraft path: the redraft may run a
	// small number of tool rounds to resolve the superego's finding, and when it
	// exhausts that budget it answers with the evidence it has rather than
	// erroring the turn. Only this opt-in path finalizes; every other caller
	// keeps the fail-closed behavior.
	FinalizeOnBudget bool
	// IsTerminalTool, when non-nil, is consulted once per turn, after a tool_use
	// batch of exactly ONE call has been executed: if it reports true for that
	// call's name and result, RunLoop ends the turn immediately with a synthetic
	// end_turn terminal (LoopResult.Yielded) instead of feeding the result back
	// and calling the model again. Nil means no tool_use ever ends a turn this
	// way — the existing behavior for any caller that leaves this unset. This
	// seam keeps the tools package agnostic of any specific tool's name or
	// result semantics; the composition root (main) injects the concrete
	// predicate, e.g. OpenClaw's sessions_yield, whose {"status":"yielded"}
	// result means the model chose to END the turn and await events rather than
	// request an ordinary follow-up call. Never consulted when the batch has
	// zero or more than one tool_use.
	IsTerminalTool func(name string, result anthropic.ToolResultBlock) bool
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
	// Yielded reports that the loop ended because the model's only tool_use was a
	// call the injected IsTerminalTool predicate recognized as ending the turn —
	// the native OpenClaw sessions_yield semantic (the model chose to await
	// events), not an ordinary tool result to loop back on. RunLoop has already
	// emitted the full terminal round for this case (assistant tool_use,
	// tool_result, and result), so a DeferFinal consumer must NOT also run the
	// governor or EmitFinalTurn on it: a yield carries no user-facing draft for
	// the superego to review, and FinalText is legitimately empty. StopReason is
	// forced to "end_turn" so the round closes cleanly instead of spinning to the
	// MaxTurns kill.
	Yielded bool
}

// withEphemeralContext returns messages with one extra trailing user message
// carrying ephemeral injected context (the tone digest), or messages unchanged
// when ephemeral is "". The result is only ever handed to a single CreateMessage
// call; the caller's accumulating messages slice is never mutated (a fresh slice
// is allocated), so the ephemeral content never enters the returned LoopResult,
// canonical history, or the stream.
func withEphemeralContext(messages []anthropic.Message, ephemeral string) []anthropic.Message {
	if ephemeral == "" {
		return messages
	}
	out := make([]anthropic.Message, 0, len(messages)+1)
	out = append(out, messages...)
	return append(out, anthropic.Message{Role: "user", Content: ephemeral})
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

	callTimeout := req.CallTimeout
	if callTimeout <= 0 {
		callTimeout = DefaultCallTimeout
	}

	keepaliveInterval := req.KeepaliveInterval
	if keepaliveInterval <= 0 {
		keepaliveInterval = DefaultKeepaliveInterval
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
		// Bound each call with its own deadline (see CallTimeout). cancel() and
		// stopKeepalive() are called explicitly after the call, not deferred, so
		// the per-turn contexts do not pile up across loop iterations and — the
		// invariant that matters — the keepalive goroutine is fully stopped BEFORE
		// any assistant/result/tool_result line is emitted below, keeping the wire
		// order deterministic (see StreamEmitter.startCallKeepalive).
		cctx, cancel := context.WithTimeout(ctx, callTimeout)
		stopKeepalive := req.Emitter.startCallKeepalive(keepaliveInterval)
		resp, err := client.CreateMessage(cctx, anthropic.MessageRequest{
			Model:     req.Model,
			System:    req.System,
			MaxTokens: req.MaxTokens,
			Effort:    req.Effort,
			Messages:  withEphemeralContext(messages, req.EphemeralContext),
			Tools:     tools,
		})
		cancel()
		stopKeepalive()
		if err != nil {
			// Fail-closed: the loop has no fallback output, so a call that blew the
			// per-call deadline surfaces as a clear error naming the bound. A
			// deadline hit is reported with just the turn and configured timeout,
			// which already fully explain it; the raw transport cause ("...context
			// deadline exceeded") is redundant here, and its wording also misled a
			// downstream error classifier into reporting a plain timeout as a
			// context overflow (#5994). Any other call error keeps the existing
			// turn-scoped wrapping.
			if cctx.Err() == context.DeadlineExceeded && ctx.Err() == nil {
				return LoopResult{}, fmt.Errorf("tools: tool-loop turn %d: call exceeded per-call deadline (%s)", turn, callTimeout)
			}
			return LoopResult{}, fmt.Errorf("tools: tool-loop turn %d: %w", turn, err)
		}
		totalUsage = totalUsage.Add(resp.Usage)

		toolUses := resp.ToolUseBlocks()
		// stop_reason other than tool_use — or tool_use with no decodable block,
		// which we treat as resolved rather than looping forever on the same
		// state — ends the loop with the terminal result event.
		isFinal := resp.StopReason != "tool_use" || len(toolUses) == 0

		// A final turn with no deliverable text cannot be sent: OpenClaw rejects an
		// empty assistant turn. Two cases produce one. (1) A thinking model spends
		// its whole MaxTokens budget on thinking and stops with zero text blocks
		// (stop_reason "max_tokens"); it cannot be continued server-side, since
		// echoing the partial assistant turn back is a last-assistant prefill, which
		// fable rejects with a 400. (2) The text is entirely tool-call markup that
		// StripToolMarkup removes before delivery, leaving nothing — so the check
		// strips first, matching what result()/EmitFinalTurn actually deliver. Either
		// way fail closed before any emission with an error naming the stop_reason; it
		// surfaces as a terminal is_error result in session.runTurn. A larger
		// MaxTokens and a bounded Effort (see main) keep case (1) rare; the guard is
		// the backstop for when they are not enough.
		if isFinal && strings.TrimSpace(StripToolMarkup(resp.Text)) == "" {
			return LoopResult{}, fmt.Errorf("tools: tool-loop turn %d: model returned no deliverable text (stop_reason %q); raise max_tokens or lower effort", turn, resp.StopReason)
		}

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

		// Injected terminal tool: when the turn's ONLY tool_use is a call the
		// composition root's IsTerminalTool predicate recognizes (e.g. OpenClaw's
		// sessions_yield executing to "yielded"), the model has chosen to END the
		// turn — it is NOT asking us to feed a tool_result back and call it again.
		// Treating a yield as an ordinary tool result was the original bug this
		// guards: the loop re-called the model, which (seeing only the tone digest)
		// yielded again every ~4-5s until it hit the MaxTurns kill, destroying the
		// real reply. So end the loop here with a synthetic end_turn terminal
		// instead of looping. tools itself carries no knowledge of which tool name
		// or result shape qualifies — that is entirely the injected predicate's
		// call, keeping this package agnostic of any specific tool's semantics.
		//
		// Detected AFTER executing the batch (not before) for two reasons: the
		// tool's real-world effect must still run exactly once, and reusing the
		// batch result means the tool is never executed twice. The assistant
		// tool_use turn and the tool_result are already emitted and appended above,
		// so the call's full round is on the wire and in canonical history; we only
		// add the terminal result. A single-call batch the predicate does NOT
		// recognize as terminal (e.g. a yield whose result was an error payload)
		// falls through to a normal next turn.
		if len(results) == 1 && req.IsTerminalTool != nil && req.IsTerminalTool(toolUses[0].Name, results[0]) {
			// StopReason is forced to end_turn: the raw response said "tool_use", but
			// the predicate says this call IS the end of the turn, and the terminal
			// result must report a non-tool_use stop_reason for OpenClaw to close the
			// round. FinalText is whatever text the model sent alongside the call
			// (usually empty for a yield), which is a legitimate empty final turn, so
			// the empty-text guard above is intentionally not applied on this path.
			result := LoopResult{
				FinalText:  resp.Text,
				StopReason: "end_turn",
				Turns:      turn,
				Messages:   messages,
				Usage:      totalUsage,
				Yielded:    true,
			}
			// Emitted unconditionally (even under DeferFinal): a yield has no
			// user-facing draft for the governor to substitute, so there is nothing
			// to defer. The consumer keys on LoopResult.Yielded to skip the governor
			// and avoid a second terminal (see session.runTurn).
			if err := req.Emitter.result(result, totalUsage); err != nil {
				return LoopResult{}, err
			}
			return result, nil
		}
	}

	// The budget is exhausted with a tool_use still pending. Fail closed by
	// default; the FinalizeOnBudget caller instead gets one graceful toolless
	// call so the round still ends with a deliverable answer.
	if req.FinalizeOnBudget {
		return finalizeWithoutTools(ctx, client, req, messages, maxTurns, callTimeout, keepaliveInterval, totalUsage)
	}
	return LoopResult{}, fmt.Errorf("tools: tool-loop exceeded MaxTurns (%d) without reaching a non-tool_use stop_reason", maxTurns)
}

// finalizeWithoutTools makes one last core call with NO tools offered after the
// tool-loop hit its round budget, so a FinalizeOnBudget loop ends with a
// deliverable text answer (the model sees every tool_result gathered so far)
// instead of erroring. It mirrors the loop's own final-turn handling: the
// empty-text guard, the deferred-vs-live emission, the appended assistant turn,
// and the usage-summed result. messages already ends with the last turn's
// user(tool_result), which is a valid prefix for a toolless call; turns is the
// budget that was consumed (reported as LoopResult.Turns).
func finalizeWithoutTools(
	ctx context.Context,
	client *anthropic.Client,
	req LoopRequest,
	messages []anthropic.Message,
	turns int,
	callTimeout time.Duration,
	keepaliveInterval time.Duration,
	totalUsage anthropic.Usage,
) (LoopResult, error) {
	cctx, cancel := context.WithTimeout(ctx, callTimeout)
	stopKeepalive := req.Emitter.startCallKeepalive(keepaliveInterval)
	resp, err := client.CreateMessage(cctx, anthropic.MessageRequest{
		Model:     req.Model,
		System:    req.System,
		MaxTokens: req.MaxTokens,
		Effort:    req.Effort,
		Messages:  messages,
		// No Tools: force a text answer so the budgeted round closes gracefully.
	})
	cancel()
	stopKeepalive()
	if err != nil {
		// Same as the RunLoop branch above (#5994): report only the bound, not
		// the redundant transport cause.
		if cctx.Err() == context.DeadlineExceeded && ctx.Err() == nil {
			return LoopResult{}, fmt.Errorf("tools: tool-loop finalize call exceeded per-call deadline (%s)", callTimeout)
		}
		return LoopResult{}, fmt.Errorf("tools: tool-loop finalize call: %w", err)
	}
	totalUsage = totalUsage.Add(resp.Usage)
	if strings.TrimSpace(StripToolMarkup(resp.Text)) == "" {
		return LoopResult{}, fmt.Errorf("tools: tool-loop finalize returned no deliverable text (stop_reason %q)", resp.StopReason)
	}
	if !req.DeferFinal {
		if err := req.Emitter.assistant(resp); err != nil {
			return LoopResult{}, err
		}
	}
	messages = append(messages, anthropic.Message{Role: "assistant", Content: resp.RawContent()})
	result := LoopResult{FinalText: resp.Text, StopReason: resp.StopReason, Turns: turns, Messages: messages, Usage: totalUsage}
	if !req.DeferFinal {
		if err := req.Emitter.result(result, totalUsage); err != nil {
			return LoopResult{}, err
		}
	}
	return result, nil
}
