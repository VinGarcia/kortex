package session

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"strings"
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

// ToneDigester is the facet seam for the ephemeral tone digest: given the
// session's user messages (each already carrying the input annotator's
// `[emoções pN]` lines, oldest first, the current turn's message last), it
// returns a compact, human-readable digest of the conversation's current
// emotional tone to inject as the LAST item of the core's context for the turn.
// The block is EPHEMERAL — recomputed each turn, injected only into that turn's
// core call (never into the canonical history and never onto the stream), exactly
// like the superego's ephemeral tail. "" means inject nothing (fail-open: no
// annotations, or any internal error). Defined here, not imported from facet, for
// the same decoupling reason the other seams are; facet.ToneDigest satisfies it
// structurally. A nil Config.ToneDigester skips the digest.
type ToneDigester interface {
	Digest(userTexts []string) string
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

// DefaultRedraftToolBudget is the safe floor Run applies when
// Config.RedraftToolBudget is 0, mirroring DefaultRedraftTimeout above. It is
// the SINGLE source of truth for the redraft tool-use round budget: session
// cannot import config (see .go-arch-lint.yml — config is not in session's
// mayDependOn list), so the const lives here and main references
// session.DefaultRedraftToolBudget. Clamping here guarantees that any caller
// that builds a Config with RedraftToolBudget 0 WITHOUT main's resolution (a
// test, the native path, a future refactor) still gets this 3-round budget,
// never RunLoop's much larger DefaultMaxTurns (25) — the silent 8x
// safety-budget drift this clamp exists to prevent.
const DefaultRedraftToolBudget = 3

// redraftActionInstruction is the deterministic guard-rail appended to the tail
// of an active-mode redraft prompt. It is read by the CORE model, so it is
// written as explicit, full-sentence Portuguese (the language the core and
// superego already converse in). It tells the core that this redraft may ACT —
// run tools to actually resolve the superego's finding — while bounding that
// action to only what the specific critique requires, so a redraft resolves the
// problem with evidence rather than rewording it out of existence.
const redraftActionInstruction = "Você pode usar ferramentas nesta rodada para RESOLVER de fato os problemas apontados pelo superego — por exemplo, rodando a verificação que você afirmou ter feito, em vez de apenas reescrever o texto. " +
	"Limite-se ao estritamente necessário: faça apenas as modificações de texto e as chamadas de ferramenta indispensáveis para sanar os pontos levantados na crítica acima, sem retrabalho gratuito e sem reabrir partes da resposta que já estavam corretas. " +
	"Quando terminar, responda com a nova versão final da resposta, incorporando as evidências que você obteve."

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
	// Effort is the output_config.effort forwarded to every core-model call (the
	// tool-loop and the governor's redraft), wired from the spawn's --effort. ""
	// lets the API default stand. It bounds how much of MaxTokens a thinking
	// model spends on thinking, so text always has room (see the empty-final-turn
	// guard in tools.RunLoop).
	Effort string
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
	// RedraftToolBudget caps how many tool-use round trips a single active
	// superego redraft may make before it must answer with text. A
	// superego-triggered redraft drives the SAME tool loop the turn uses, so the
	// core can ACT on the critique (e.g. actually run the verification the
	// superego flagged as missing) and answer with evidence instead of rewording
	// the problem away; this budget bounds that loop so a redraft can never open
	// an unbounded tool run. The composition root resolves the configured value
	// (falling back to session.DefaultRedraftToolBudget) and passes the concrete
	// budget in; a 0 here is clamped to DefaultRedraftToolBudget at the
	// consumption point (never handed to RunLoop as 0, which would inherit its
	// far larger MaxTurns default), so the contract holds for any caller.
	// Only consulted on the active (Governor) path.
	RedraftToolBudget int
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
	// ToneDigester, when non-nil, computes the ephemeral tone digest injected as
	// the last item of the core's context on each turn (the session's annotated
	// user messages in, a bounded digest block out). Nil skips the digest, leaving
	// the core context unchanged. The digest never enters canonical history.
	ToneDigester ToneDigester
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
	// It stays gated on TurnEvaluator being wired (see below) so the
	// active-superego path — mutually exclusive with the evaluator — keeps seeing
	// a single-turn view (turnIndex 0), exactly as before this feature existed.
	var turns []history.Turn
	// priorUserTexts is the tone digest's OWN lightweight source of prior turns:
	// just each completed turn's annotated user text, accumulated on every path
	// independently of turns. The digest needs only UserText (not a full
	// history.Turn with tool results), and feeding it from here keeps the digest
	// decoupled from — and cost-neutral to — the governor/superego, which must
	// not begin seeing multi-turn history as a side effect of this feature.
	var priorUserTexts []string

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
			msgs, turn, err = runTurn(ctx, cfg, emitter, systemPrompt, msgs, turns, priorUserTexts, userText)
			if err != nil {
				return err
			}
			// A failed turn (turn == nil) emitted its error result and left
			// history unchanged, so there is nothing to accumulate or evaluate.
			if turn != nil {
				// The tone digest's own accumulator grows on every completed turn,
				// regardless of which facets are wired, so the digest always sees the
				// full prior conversation's annotated user text.
				priorUserTexts = append(priorUserTexts, turn.UserText)
				// turns and the evaluator stay gated on TurnEvaluator exactly as
				// before: the active-superego (Governor) path is mutually exclusive
				// with the evaluator, so turns stays empty there and the governor keeps
				// seeing a single-turn snapshot (turnIndex 0). The digest must not
				// change that, which is why it draws from priorUserTexts above instead.
				if cfg.TurnEvaluator != nil {
					turns = append(turns, *turn)
					snapshot := history.Snapshot{SessionID: cfg.SessionID, Turns: append([]history.Turn(nil), turns...)}
					cfg.TurnEvaluator.EvaluateCompletedTurn(len(turns)-1, snapshot)
				}
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
	priorUserTexts []string,
	userText string,
) ([]anthropic.Message, *history.Turn, error) {
	res, err := tools.RunLoop(ctx, cfg.Client, cfg.Dispatcher, tools.LoopRequest{
		Model:       cfg.Model,
		System:      systemPrompt,
		MaxTokens:   cfg.MaxTokens,
		Effort:      cfg.Effort,
		History:     msgs,
		UserMessage: userText,
		CallTimeout: cfg.LoopCallTimeout,
		Emitter:     emitter,
		// The ephemeral tone digest is injected as the last item of the core's
		// context for this turn and never persisted (see toneDigest). It draws from
		// priorUserTexts, not priorTurns, so it stays independent of the governor's
		// turn view.
		EphemeralContext: toneDigest(cfg, priorUserTexts, userText),
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
	// governed text on the active path, the raw draft otherwise). ToolCalls is
	// populated from this turn's exchanges so the superego can verify provenance
	// (what the assistant actually ran this turn), mirroring the proxy's
	// event-driven history.Recorder. Only the messages appended for THIS turn are
	// scanned (res.Messages is the whole conversation; the prior history is
	// msgs), so each Turn carries only its own tool calls, exactly as the proxy
	// Recorder attaches them to the open turn. IsError is left false: a Turn is
	// built only on the success path here (RunLoop failures emit an error result
	// and return a nil turn above), which matches the proxy's non-error terminal
	// result.
	turn := history.Turn{
		UserText:      userText,
		AssistantText: finalText,
		ToolCalls:     toolCallsFromMessages(res.Messages[len(msgs):]),
		StopReason:    res.StopReason,
		Completed:     true,
	}
	return finalMessages, &turn, nil
}

// toneDigest computes the ephemeral tone digest for the turn about to run: the
// session's annotated user messages (every prior turn's UserText, oldest first,
// then this turn's text) handed to the digester. priorUserTexts is the digest's
// own accumulator (see Run), kept separate from the governor's turn view so the
// digest never changes what the superego receives. Returns "" when no digester
// is wired, so the core context is unchanged on the no-facet path. The digest is
// handed to RunLoop as EphemeralContext and never persisted.
func toneDigest(cfg Config, priorUserTexts []string, userText string) string {
	if cfg.ToneDigester == nil {
		return ""
	}
	userTexts := make([]string, 0, len(priorUserTexts)+1)
	userTexts = append(userTexts, priorUserTexts...)
	userTexts = append(userTexts, userText)
	return cfg.ToneDigester.Digest(userTexts)
}

// toolCallsFromMessages derives this turn's tool calls from the native message
// slice RunLoop appended for the turn (user → assistant tool_use → user
// tool_result → … → final assistant). It mirrors the proxy history.Recorder:
// every tool_use content block opens a ToolCall{ID,Name,Input}, and every
// tool_result block attaches its text and is_error to the matching ToolCall by
// id. The native loop (tools.RunLoop) executes each tool_use exactly once, so
// each id carries exactly one result on this path. The native message shapes
// are the concrete Go values
// RunLoop builds: an assistant turn's Content is []json.RawMessage (the raw
// content blocks), and a tool_result turn's Content is []anthropic.ToolResultBlock.
// Other content shapes (e.g. the plain-string user message) carry no tool calls
// and are skipped. The result preserves tool_use order and is deterministic.
func toolCallsFromMessages(messages []anthropic.Message) []history.ToolCall {
	var calls []history.ToolCall
	for _, msg := range messages {
		switch content := msg.Content.(type) {
		case []json.RawMessage:
			for _, raw := range content {
				var block struct {
					Type  string          `json:"type"`
					ID    string          `json:"id"`
					Name  string          `json:"name"`
					Input json.RawMessage `json:"input"`
				}
				if err := json.Unmarshal(raw, &block); err != nil {
					continue
				}
				if block.Type == "tool_use" {
					calls = append(calls, history.ToolCall{
						ID:    block.ID,
						Name:  block.Name,
						Input: string(block.Input),
					})
				}
			}
		case []anthropic.ToolResultBlock:
			for _, result := range content {
				for i := range calls {
					if calls[i].ID != result.ToolUseID {
						continue
					}
					// The native loop emits exactly one tool_result per tool_use id,
					// so a plain assignment is correct: there is no duplicate result
					// to guard against on this path.
					calls[i].Result = result.Content
					calls[i].IsError = result.IsError
					break
				}
			}
		}
	}
	return calls
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
	// messages plus the ephemeral tail. Unlike the pre-#4753 behavior (a pure
	// text revision with no tools), it drives the SAME bounded tool loop a normal
	// turn uses, so when the superego's finding needs ACTION — e.g. "you did not
	// actually verify X" — the core can run the check and answer with evidence
	// rather than rewording the problem out of existence. The tail's final entry
	// is the critique this redraft must answer; it becomes the loop's user
	// message (with the deterministic bounded-action guard-rail appended), and
	// everything before it continues the conversation as history. The tail is
	// appended to a clone, so res.Messages (the canonical history) is never
	// mutated. Each redraft's usage is accumulated so the re-emitted result bills
	// the full round, not just the first draft.
	//
	// The whole redraft (tool loop plus any fallback) is bounded by RedraftTimeout
	// so a hung or runaway redraft can never block the turn; the loop is further
	// capped at RedraftToolBudget tool-use rounds and finalizes with a text answer
	// when that budget is spent (FinalizeOnBudget). If the tool loop errors for
	// any non-deadline reason, the redraft falls back to the pre-#4753 text-only
	// revision; and GovernOutput itself fails open to the best draft in hand if
	// the redraft returns an error, so acting can never break the turn.
	var redraftUsage anthropic.Usage
	redraft := func(ctx context.Context, tail []anthropic.Message) (string, error) {
		redraftTimeout := cfg.RedraftTimeout
		if redraftTimeout <= 0 {
			redraftTimeout = DefaultRedraftTimeout
		}
		ctx, cancel := context.WithTimeout(ctx, redraftTimeout)
		defer cancel()

		// Clamp at the consumption point, like the RedraftTimeout floor above
		// (see DefaultRedraftToolBudget for why 0 must not reach RunLoop).
		redraftToolBudget := cfg.RedraftToolBudget
		if redraftToolBudget <= 0 {
			redraftToolBudget = DefaultRedraftToolBudget
		}

		// A nil/empty tail carries no critique to answer, so it cannot drive the
		// tool loop; it falls straight to the text-only path, preserving the
		// pre-#4753 behavior exactly for that (never-reached-in-practice) case.
		if len(tail) > 0 {
			loopHistory := append(slices.Clone(res.Messages), tail[:len(tail)-1]...)
			loopRes, err := tools.RunLoop(ctx, cfg.Client, cfg.Dispatcher, tools.LoopRequest{
				Model:       cfg.Model,
				System:      systemPrompt,
				MaxTokens:   cfg.MaxTokens,
				Effort:      cfg.Effort,
				History:     loopHistory,
				UserMessage: redraftToolUserMessage(tail[len(tail)-1]),
				MaxTurns:    redraftToolBudget,
				CallTimeout: cfg.LoopCallTimeout,
				// Budget exhausted ends the round with a text answer that still sees
				// the tool_results gathered, not a turn error.
				FinalizeOnBudget: true,
				// Emitter left nil: the redraft's tool loop is internal deliberation
				// over the ephemeral tail, so its intermediate assistant/tool_result
				// events never reach the wire — only the governed final text does
				// (EmitFinalTurn). The tools' real side effects still run; that is
				// the point of letting the redraft act.
			})
			if err == nil {
				redraftUsage = redraftUsage.Add(loopRes.Usage)
				// A redraft can now run tools with real side effects whose events
				// never reach the wire (Emitter is nil above), so leave a bounded
				// audit trail of what it did. Only the messages this redraft appended
				// (everything past loopHistory) are inspected, so the original turn's
				// tool calls are never re-logged; each line carries the tool id, name,
				// error flag, and a length-capped result marker, never full output.
				for _, call := range toolCallsFromMessages(loopRes.Messages[len(loopHistory):]) {
					diagf(cfg.Diag, "session: redraft tool call id=%s name=%s is_error=%t result=%q",
						call.ID, call.Name, call.IsError, auditResultMarker(call.Result))
				}
				// A redraft that strips to nothing is not a deliverable draft; report
				// it as a failure so GovernOutput fails open to the non-empty draft in
				// hand rather than delivering an empty turn OpenClaw would reject.
				if strings.TrimSpace(tools.StripToolMarkup(loopRes.FinalText)) == "" {
					return "", fmt.Errorf("redraft tool loop returned no deliverable text (stop_reason %q)", loopRes.StopReason)
				}
				return loopRes.FinalText, nil
			}
			// The overall redraft deadline already fired: the text-only fallback
			// cannot succeed on a dead context and would only burn another blocked
			// call, so surface the error and let GovernOutput fail open.
			if ctx.Err() != nil {
				return "", err
			}
			// Fail-safe: the bounded tool loop errored for another reason. Degrade to
			// the pre-#4753 text-only redraft — a reword is worse than acting, but far
			// better than breaking the turn.
			diagf(cfg.Diag, "session: redraft tool loop failed, falling back to text-only redraft: %v", err)
		}
		return textOnlyRedraft(ctx, cfg, systemPrompt, res.Messages, tail, &redraftUsage)
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
	//
	// The flatten is deliberately unconditional, even though the canary core model
	// (fable) returns signature-bearing thinking blocks on every turn. Two reasons:
	//   1. The governor substitutes its own delivered text for the core's draft, so
	//      persisting the draft's structured blocks beside a different delivered
	//      text would record a turn that never happened. Only the delivered text
	//      belongs here.
	//   2. Dropping a terminal turn's thinking blocks does not break resume:
	//      Anthropic requires thinking blocks preserved only WITHIN a tool-use cycle
	//      (a thinking+tool_use turn answered by a tool_result in the same request),
	//      and a terminal turn is followed on resume by a fresh user turn. Confirmed
	//      live 2026-10-04. (Intermediate tool_use turns differ — RunLoop echoes
	//      those verbatim via RawContent, signatures intact.)
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

// textOnlyRedraft is the pre-#4753 redraft: one core-model call with no tools
// offered, over the canonical messages plus the ephemeral tail. It is the
// fail-safe the action-capable redraft degrades to when its bounded tool loop
// errors, and the path a nil/empty tail takes. The caller owns the context
// deadline (the redraft closure already wrapped ctx with RedraftTimeout), so
// this does not re-bound it. Usage is accumulated into the caller's running
// total via the usage pointer so the re-emitted result still bills this call.
func textOnlyRedraft(
	ctx context.Context,
	cfg Config,
	systemPrompt string,
	canonical []anthropic.Message,
	tail []anthropic.Message,
	usage *anthropic.Usage,
) (string, error) {
	messages := append(slices.Clone(canonical), tail...)
	resp, err := cfg.Client.CreateMessage(ctx, anthropic.MessageRequest{
		Model:     cfg.Model,
		System:    systemPrompt,
		MaxTokens: cfg.MaxTokens,
		Effort:    cfg.Effort,
		Messages:  messages,
	})
	if err != nil {
		return "", err
	}
	*usage = usage.Add(resp.Usage)
	if strings.TrimSpace(tools.StripToolMarkup(resp.Text)) == "" {
		return "", fmt.Errorf("redraft returned no deliverable text (stop_reason %q)", resp.StopReason)
	}
	return resp.Text, nil
}

// auditResultMarker bounds a tool result to a short, rune-safe marker for the
// redraft audit log so a large tool payload can never flood the diag stream; it
// records enough to correlate the call, never the full output.
func auditResultMarker(result string) string {
	const maxRunes = 120
	runes := []rune(result)
	if len(runes) > maxRunes {
		return string(runes[:maxRunes]) + "…(+more)"
	}
	return result
}

// redraftToolUserMessage renders the user message that opens the redraft tool
// loop: the superego critique the redraft must answer (the ephemeral tail's
// final entry), followed by the deterministic bounded-action guard-rail that
// tells the core to act only as far as the finding requires. A critique whose
// content is not a plain string (never produced by revisionInstruction today)
// degrades to the guard-rail alone rather than dropping it.
func redraftToolUserMessage(critique anthropic.Message) string {
	text, _ := critique.Content.(string)
	if text == "" {
		return redraftActionInstruction
	}
	return text + "\n\n" + redraftActionInstruction
}

func diagf(w io.Writer, format string, args ...any) {
	if w == nil {
		return
	}
	fmt.Fprintf(w, format+"\n", args...)
}
