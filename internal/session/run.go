package session

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"slices"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/history"
	"github.com/vingarcia/kortex/internal/protocol"
	"github.com/vingarcia/kortex/internal/tools"
)

// Annotator is the facet seam on the incoming side: it may rewrite the user's
// text before the turn runs (the native-path parallel of the proxy's
// to-backend Interceptor). Defined here, not imported from the facet package,
// for the same decoupling reason the proxy defines its own ports. A nil
// Config.Annotator skips annotation.
type Annotator interface {
	Annotate(text string) string
}

// TurnEvaluator is the facet seam on the completion side: it is notified of
// each completed turn with the session's accumulated history (the native-path
// parallel of the proxy's from-backend TurnEvaluator; identical signature, so
// facet.OutputEvaluator satisfies both). The implementation returns promptly
// by contract (it dispatches its own goroutine). A nil Config.TurnEvaluator
// skips evaluation.
type TurnEvaluator interface {
	EvaluateCompletedTurn(turnIndex int, snapshot history.Snapshot)
}

// OutputGovernor is the facet seam for the active superego loop: it runs
// SYNCHRONOUSLY before the turn's final assistant/result events are emitted and
// decides the text to deliver. Given the core's draft, it runs the blocking
// evaluate→review→revise ladder and returns the governed text (an approved
// draft, a later redraft, or — when held is true — a hold-and-ask message
// delivered instead of the draft). redraft lets the governor ask the CORE model
// for a new draft, given an ephemeral message tail (the rejected drafts and the
// superego critiques) appended to the turn's canonical messages; that tail is
// never persisted and never emitted. This interface is defined here with an
// inline func type and only shared types (context/history/anthropic), so
// facet.ActiveSuperego satisfies it structurally without session importing
// facet. A nil Config.Governor leaves the pre-F2d (shadow/none) path unchanged.
type OutputGovernor interface {
	GovernOutput(
		ctx context.Context,
		turnIndex int,
		snapshot history.Snapshot,
		draft string,
		redraft func(ctx context.Context, tail []anthropic.Message) (string, error),
	) (text string, held bool)
}

// maxLineBytes bounds one NDJSON input line. The default bufio.Scanner cap
// (64KiB) is far too small for the initialize control_request: it carries the
// full agent system prompt (appendSystemPrompt), which routinely runs to tens
// of KiB. 16MiB leaves generous headroom without risking an unbounded read.
const maxLineBytes = 16 << 20

// DefaultRedraftTimeout is the safe floor Run applies when Config.RedraftTimeout
// is 0, mirroring tools.DefaultCallTimeout. It guarantees a zero value can never
// collapse into an already-expired deadline that would make every redraft fail
// instantly and silently disable the superego (fail open on every turn). 60s
// matches main.go's defaultRedraftTimeout.
const DefaultRedraftTimeout = 60 * time.Second

// Config is everything Run needs, supplied by the composition root (main): Run
// reads no env and resolves no paths of its own. Every field except Diag is
// required.
type Config struct {
	// Stdin is the OpenClaw NDJSON input stream (the real claude-cli's stdin).
	Stdin io.Reader
	// Stdout is where stream-json events and the control_response are written
	// (the real claude-cli's stdout).
	Stdout io.Writer
	// Client calls the Anthropic Messages API for every turn.
	Client *anthropic.Client
	// Dispatcher executes the tool_use blocks a turn requests (builtins + MCP).
	Dispatcher *tools.Dispatcher
	// Store persists and loads the canonical history for SessionID.
	Store *Store
	// SessionID is the argv-supplied session id this process owns; it keys the
	// persisted history and stamps every emitted event.
	SessionID string
	// Model is the argv-supplied model id passed to every Messages API call.
	Model string
	// MaxTokens caps each turn's response length.
	MaxTokens int
	// RedraftTimeout bounds one redraft call to the core model inside the active
	// superego loop. The core client carries no HTTP timeout and the redraft is
	// the one core-model call in the governor loop with no deadline of its own,
	// so without this bound a hung redraft blocks the whole turn forever; on
	// timeout the governor fails open to delivering the raw draft. 0 means
	// DefaultRedraftTimeout.
	RedraftTimeout time.Duration
	// LoopCallTimeout bounds each individual core-model call inside the
	// tool-loop (RunLoop). The core client carries no HTTP timeout and MaxTurns
	// only caps the number of round trips, so without this a single hung call
	// blocks the whole turn forever; 0 lets RunLoop apply its safe default. The
	// tool loop has no fallback output, so a timed-out call fails closed as a
	// turn error rather than fail-open like the redraft.
	LoopCallTimeout time.Duration
	// NewUUID mints one v4 message uuid per emitted assistant/user event.
	NewUUID func() string
	// Diag, when non-nil, receives one-line diagnostics for non-fatal problems
	// (e.g. a history save that failed after the turn already completed). It is
	// never the stdout wire. Nil discards them.
	Diag io.Writer
	// Annotator, when non-nil, rewrites each real user message before its turn
	// runs (the input-annotator facet). Nil leaves the user text untouched.
	Annotator Annotator
	// TurnEvaluator, when non-nil, is notified of each completed turn over the
	// session's accumulated history (the output-evaluator facet, which in turn
	// fires the superego on a gate hit). Nil skips evaluation. In active-superego
	// mode this is left nil: the governor already evaluates synchronously, so the
	// async shadow evaluation would be redundant work on the same draft.
	TurnEvaluator TurnEvaluator
	// Governor, when non-nil, puts the session on the active-superego path: the
	// final turn is deferred (RunLoop DeferFinal), the governor decides the text
	// to deliver, and the delivered text — not the core's raw draft — is emitted
	// and persisted. Nil keeps the plain path (final turn emitted directly by
	// RunLoop). Governor and the async TurnEvaluator are mutually exclusive by
	// construction in the composition root.
	Governor OutputGovernor
}

// Run drives a native kortex session: it reads the OpenClaw NDJSON protocol
// from cfg.Stdin, answers the initialize handshake, and for each user turn runs
// the tool-loop with the session's accumulated history, emits the turn's
// stream-json on cfg.Stdout, and persists the updated history. One process
// serves every turn of a session until stdin closes; a later process resumes
// the same conversation by loading the persisted history under the same session
// id.
//
// A turn whose tool-loop fails (e.g. the Messages API call errored) does not
// abort the session: Run emits a terminal error result so OpenClaw completes
// that round rather than hanging, logs the cause to Diag, and continues reading
// the next turn. Run returns only on stdin EOF, a context cancellation, or a
// failure writing to stdout (which breaks the wire and cannot be recovered).
func Run(ctx context.Context, cfg Config) error {
	msgs, err := cfg.Store.Load(cfg.SessionID)
	if err != nil {
		// A corrupt persisted history is fatal: continuing with a silently
		// empty conversation would drop the context OpenClaw assumes is intact.
		return err
	}

	emitter := tools.NewStreamEmitter(cfg.Stdout, cfg.SessionID, cfg.NewUUID)
	var systemPrompt string
	// turns accumulates one history.Turn per completed turn, feeding the
	// TurnEvaluator the same per-process view the passthrough recorder gives
	// it: turns handled by this process, not the cross-resume history on disk.
	var turns []history.Turn

	scanner := bufio.NewScanner(cfg.Stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), maxLineBytes)
	for scanner.Scan() {
		if err := ctx.Err(); err != nil {
			return err
		}
		ev := protocol.Parse(scanner.Bytes())
		switch ev.Type {
		case protocol.TypeControlRequest:
			if ev.Subtype == "initialize" {
				// OpenClaw ships the system prompt here, not via argv; capture it
				// for every subsequent turn and acknowledge the handshake.
				systemPrompt = ev.AppendSystemPrompt
				line, err := protocol.EmitControlResponse(ev.RequestID)
				if err != nil {
					return fmt.Errorf("session: building control response: %w", err)
				}
				if _, err := cfg.Stdout.Write(line); err != nil {
					return fmt.Errorf("session: writing control response: %w", err)
				}
			}
			// Other control requests (interrupt, can_use_tool, hook_callback)
			// are out of v0 scope: kortex drives its own tools and never asks
			// permission, so there is nothing to answer. Ignored intentionally.

		case protocol.TypeUser:
			// --replay-user-messages echoes the user's own turns back on stdout;
			// those are not new input to act on.
			if ev.IsReplay || ev.Message == nil {
				continue
			}
			// The input-annotator facet rewrites the user's text before the
			// turn runs, so the annotated text is what the core model sees and
			// what the canonical history persists — exactly the passthrough
			// behavior, where the annotated line is what reaches the backend.
			userText := ev.Message.TextContent()
			if cfg.Annotator != nil {
				userText = cfg.Annotator.Annotate(userText)
			}
			var turn *history.Turn
			msgs, turn, err = runTurn(ctx, cfg, emitter, systemPrompt, msgs, turns, userText)
			if err != nil {
				return err
			}
			// A failed turn (turn == nil) emitted its error result and left
			// history unchanged; it has no assistant output to evaluate, so the
			// evaluator fires only on a turn that completed successfully.
			if turn != nil && cfg.TurnEvaluator != nil {
				turns = append(turns, *turn)
				snapshot := history.Snapshot{SessionID: cfg.SessionID, Turns: append([]history.Turn(nil), turns...)}
				cfg.TurnEvaluator.EvaluateCompletedTurn(len(turns)-1, snapshot)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("session: reading stdin: %w", err)
	}
	return nil
}

// runTurn runs one user turn through the tool-loop and returns the updated
// history plus a history.Turn describing the completed turn (nil when the turn
// failed). A tool-loop failure is reported to OpenClaw as a terminal error
// result and the previous history is returned unchanged with a nil turn, so the
// session can carry on with the next turn; only a failure writing that error
// result (a broken wire) propagates as a fatal error.
func runTurn(
	ctx context.Context,
	cfg Config,
	emitter *tools.StreamEmitter,
	systemPrompt string,
	msgs []anthropic.Message,
	priorTurns []history.Turn,
	userText string,
) ([]anthropic.Message, *history.Turn, error) {
	res, err := tools.RunLoop(ctx, cfg.Client, cfg.Dispatcher, tools.LoopRequest{
		Model:       cfg.Model,
		System:      systemPrompt,
		MaxTokens:   cfg.MaxTokens,
		History:     msgs,
		UserMessage: userText,
		CallTimeout: cfg.LoopCallTimeout,
		Emitter:     emitter,
		// The active path defers the final turn so the governor can decide the
		// delivered text before anything reaches the wire.
		DeferFinal: cfg.Governor != nil,
	})
	if err != nil {
		if emitErr := emitter.ErrorResult(fmt.Sprintf("kortex: %v", err)); emitErr != nil {
			return nil, nil, fmt.Errorf("session: %w", emitErr)
		}
		diagf(cfg.Diag, "session: turn failed, round closed with error result: %v", err)
		return msgs, nil, nil
	}

	finalText := res.FinalText
	finalMessages := res.Messages
	if cfg.Governor != nil {
		var govErr error
		finalText, finalMessages, govErr = governTurn(ctx, cfg, emitter, systemPrompt, priorTurns, userText, res)
		if govErr != nil {
			return nil, nil, govErr
		}
	}

	if err := cfg.Store.Save(cfg.SessionID, finalMessages); err != nil {
		// The turn already succeeded and its result reached OpenClaw; a save
		// failure only costs cross-process resume, so it must not fail the turn.
		diagf(cfg.Diag, "session: history not persisted (resume will lose this turn): %v", err)
	}
	// The turn's canonical text for the facets: the user message as sent (the
	// annotator already rewrote it) and the assistant's DELIVERED text (the
	// governed text on the active path, the raw draft otherwise). Unlike the
	// proxy's event-driven history.Recorder, this native construction leaves
	// ToolCalls and IsError unset — every current facet consumer reads only the
	// user/assistant text, so the paths behave identically; a future facet that
	// needs tool-call context would have to populate them from the LoopResult.
	turn := history.Turn{
		UserText:      userText,
		AssistantText: finalText,
		StopReason:    res.StopReason,
		Completed:     true,
	}
	return finalMessages, &turn, nil
}

// governTurn runs the active superego loop over the deferred final draft and
// emits the delivered turn. It returns the delivered text and the canonical
// messages to persist: the loop's rejected drafts and critiques live only in an
// ephemeral tail passed to the redraft callback, so the persisted messages carry
// only the single delivered assistant turn (its content replaced with the
// governed text), never the tail. A failure emitting the final turn (a broken
// wire) propagates; the governor itself never errors (it fails open internally).
func governTurn(
	ctx context.Context,
	cfg Config,
	emitter *tools.StreamEmitter,
	systemPrompt string,
	priorTurns []history.Turn,
	userText string,
	res tools.LoopResult,
) (string, []anthropic.Message, error) {
	turnIndex := len(priorTurns)
	reviewTurns := append(append([]history.Turn(nil), priorTurns...), history.Turn{
		UserText:      userText,
		AssistantText: res.FinalText,
		StopReason:    res.StopReason,
		Completed:     true,
	})
	snapshot := history.Snapshot{SessionID: cfg.SessionID, Turns: reviewTurns}

	// redraft asks the core model for a new draft over the turn's canonical
	// messages plus the ephemeral tail. No tools are offered: a redraft is a pure
	// text revision, never a fresh round of tool execution. The tail is appended
	// to a clone, so res.Messages (the canonical history) is never mutated by the
	// loop. Each redraft's usage is accumulated so the re-emitted result bills the
	// full round, not just the first draft.
	var redraftUsage anthropic.Usage
	redraft := func(ctx context.Context, tail []anthropic.Message) (string, error) {
		redraftTimeout := cfg.RedraftTimeout
		if redraftTimeout <= 0 {
			redraftTimeout = DefaultRedraftTimeout
		}
		ctx, cancel := context.WithTimeout(ctx, redraftTimeout)
		defer cancel()
		messages := append(slices.Clone(res.Messages), tail...)
		resp, err := cfg.Client.CreateMessage(ctx, anthropic.MessageRequest{
			Model:     cfg.Model,
			System:    systemPrompt,
			MaxTokens: cfg.MaxTokens,
			Messages:  messages,
		})
		if err != nil {
			return "", err
		}
		redraftUsage = redraftUsage.Add(resp.Usage)
		return resp.Text, nil
	}

	// The held bool is intentionally not consumed here: when a turn is held,
	// GovernOutput already returns the hold-and-ask text as finalText, which this
	// seam delivers and persists like any other assistant turn (the facet log is
	// where the hold decision is recorded). It is part of the port contract for
	// callers that may want to signal a held turn differently; the native path
	// does not.
	finalText, _ := cfg.Governor.GovernOutput(ctx, turnIndex, snapshot, res.FinalText, redraft)
	finalUsage := res.Usage.Add(redraftUsage)
	if err := emitter.EmitFinalTurn(finalText, res.Turns, res.StopReason, finalUsage); err != nil {
		return "", nil, fmt.Errorf("session: %w", err)
	}

	// Persist only the clean delivered text as the final assistant turn: replace
	// the raw draft content of the last (assistant) message with the governed
	// text. The ephemeral tail was never in res.Messages, so nothing else needs
	// stripping.
	finalMessages := replaceFinalAssistantText(res.Messages, finalText)
	return finalText, finalMessages, nil
}

// replaceFinalAssistantText returns a copy of messages with the last message's
// content replaced by text (a plain string, valid Messages API input). The
// active path calls it to persist the governed delivered text in place of the
// core's raw final draft, so canonical history never records a rejected draft.
func replaceFinalAssistantText(messages []anthropic.Message, text string) []anthropic.Message {
	if len(messages) == 0 {
		return messages
	}
	out := slices.Clone(messages)
	out[len(out)-1] = anthropic.Message{Role: "assistant", Content: text}
	return out
}

func diagf(w io.Writer, format string, args ...any) {
	if w == nil {
		return
	}
	fmt.Fprintf(w, format+"\n", args...)
}
