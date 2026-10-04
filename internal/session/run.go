package session

import (
	"bufio"
	"context"
	"fmt"
	"io"

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

// maxLineBytes bounds one NDJSON input line. The default bufio.Scanner cap
// (64KiB) is far too small for the initialize control_request: it carries the
// full agent system prompt (appendSystemPrompt), which routinely runs to tens
// of KiB. 16MiB leaves generous headroom without risking an unbounded read.
const maxLineBytes = 16 << 20

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
	// fires the superego on a gate hit). Nil skips evaluation.
	TurnEvaluator TurnEvaluator
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
			msgs, turn, err = runTurn(ctx, cfg, emitter, systemPrompt, msgs, userText)
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
	userText string,
) ([]anthropic.Message, *history.Turn, error) {
	res, err := tools.RunLoop(ctx, cfg.Client, cfg.Dispatcher, tools.LoopRequest{
		Model:       cfg.Model,
		System:      systemPrompt,
		MaxTokens:   cfg.MaxTokens,
		History:     msgs,
		UserMessage: userText,
		Emitter:     emitter,
	})
	if err != nil {
		if emitErr := emitter.ErrorResult(fmt.Sprintf("kortex: %v", err)); emitErr != nil {
			return nil, nil, fmt.Errorf("session: %w", emitErr)
		}
		diagf(cfg.Diag, "session: turn failed, round closed with error result: %v", err)
		return msgs, nil, nil
	}
	if err := cfg.Store.Save(cfg.SessionID, res.Messages); err != nil {
		// The turn already succeeded and its result reached OpenClaw; a save
		// failure only costs cross-process resume, so it must not fail the turn.
		diagf(cfg.Diag, "session: history not persisted (resume will lose this turn): %v", err)
	}
	// The turn's canonical text for the facets: the user message as sent (the
	// annotator already rewrote it) and the assistant's final text. Unlike the
	// proxy's event-driven history.Recorder, this native construction leaves
	// ToolCalls and IsError unset — every current facet consumer reads only the
	// user/assistant text, so the paths behave identically; a future facet that
	// needs tool-call context would have to populate them from the LoopResult.
	turn := history.Turn{
		UserText:      userText,
		AssistantText: res.FinalText,
		StopReason:    res.StopReason,
		Completed:     true,
	}
	return res.Messages, &turn, nil
}

func diagf(w io.Writer, format string, args ...any) {
	if w == nil {
		return
	}
	fmt.Fprintf(w, format+"\n", args...)
}
