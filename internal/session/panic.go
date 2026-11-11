package session

import (
	"context"
	"fmt"
	"runtime/debug"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/history"
	"github.com/vingarcia/kortex/internal/tools"
)

// panicErrorText is the whole user-visible text of a recovered panic. It
// reaches OpenClaw as "kortex: <text>", so it must not leak the panic value.
const panicErrorText = "internal error (panic recovered)"

// PanicError is the error for a panic recovered on the turn path. Error() is
// short and omits the panic value on purpose; the value stays reachable for
// diagnostics through Unwrap. Even when wrapped with fmt.Errorf("x: %w", p) the
// text stays non-leaky, because %w only appends Error().
type PanicError struct {
	cause error
}

// NewPanicError wraps a recovered panic value.
func NewPanicError(value any) *PanicError {
	return &PanicError{cause: fmt.Errorf("panic: %v", value)}
}

func (e *PanicError) Error() string { return panicErrorText }

func (e *PanicError) Unwrap() error { return e.cause }

// runTurnRecovered runs runTurn and converts a panic anywhere in it (history
// pruning, tone digest, RunLoop and the tools beneath it, the governor,
// persistence) into the same outcome as an ordinary turn failure: a terminal
// error result, unchanged history, nil turn, nil error, so the session carries
// on. The panic value and stack go to the diag stream only.
//
// Wire well-formedness: the emitter writes each NDJSON line with one Write
// under a mutex, so a panic cannot leave a half-written line. A panic after the
// round already emitted its final result line (only Store.Save and the code
// after governTurn are past that point) makes the error result a SECOND result
// line for the same round; that is accepted over crashing the process, and the
// persisted history is simply not updated for that turn.
func runTurnRecovered(
	ctx context.Context,
	cfg Config,
	emitter *tools.StreamEmitter,
	systemPrompt string,
	msgs []anthropic.Message,
	priorTurns []history.Turn,
	priorUserTexts []string,
	userText string,
) (outMsgs []anthropic.Message, outTurn *history.Turn, err error) {
	defer func() {
		r := recover()
		if r == nil {
			return
		}
		diagf(cfg.Diag, "session: panic recovered in turn: %v\n%s", r, debug.Stack())
		panicErr := NewPanicError(r)
		outMsgs, outTurn, err = msgs, nil, nil
		if emitErr := emitter.ErrorResult(fmt.Sprintf("kortex: %v", panicErr)); emitErr != nil {
			err = fmt.Errorf("session: %w", emitErr)
			return
		}
		diagf(cfg.Diag, "session: turn failed, round closed with error result: %v", panicErr)
	}()
	return runTurn(ctx, cfg, emitter, systemPrompt, msgs, priorTurns, priorUserTexts, userText)
}
