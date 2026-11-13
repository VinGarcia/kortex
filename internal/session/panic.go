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
// round already emitted its terminal result line (only Store.Save and the code
// after governTurn are past that point) must NOT emit a second result: OpenClaw
// would read it as the answer to the NEXT round, showing the user an error in
// reply to a message that has not run yet. In that case the panic is only
// logged, and the persisted history is simply not updated for that turn.
func runTurnRecovered(
	ctx context.Context,
	cfg Config,
	emitter *tools.StreamEmitter,
	systemPrompt string,
	msgs []anthropic.Message,
	priorTurns []history.Turn,
	priorUserTexts []string,
	userText string,
	delivered string,
) (outMsgs []anthropic.Message, outTurn *history.Turn, err error) {
	resultsBefore := emitter.ResultCount()
	defer func() {
		if r := recover(); r != nil {
			diagf(cfg.Diag, "session: panic recovered in turn: %v\n%s", r, debug.Stack())
			outMsgs, outTurn, err = msgs, nil, closeRoundAfterPanic(cfg, emitter, resultsBefore, NewPanicError(r))
		}
	}()
	return runTurn(ctx, cfg, emitter, systemPrompt, msgs, priorTurns, priorUserTexts, userText, delivered)
}

// closeRoundAfterPanic ends the round of a turn that panicked. It emits the
// short error result only when the round has not already written its terminal
// result line (resultsBefore is the emitter's ResultCount at turn start); a
// second result would be read by OpenClaw as the answer to the next round. The
// returned error is non-nil only when the error result itself could not be
// written (wire presumed broken).
func closeRoundAfterPanic(cfg Config, emitter *tools.StreamEmitter, resultsBefore int, panicErr *PanicError) error {
	if emitter.ResultCount() > resultsBefore {
		diagf(cfg.Diag, "session: panic after the round's result line; not emitting a second result: %v", panicErr)
		return nil
	}
	if err := emitter.ErrorResult(fmt.Sprintf("kortex: %v", panicErr)); err != nil {
		return fmt.Errorf("session: %w", err)
	}
	diagf(cfg.Diag, "session: turn failed, round closed with error result: %v", panicErr)
	return nil
}
