package tools

import (
	"context"
	"errors"
	"fmt"
)

// TurnErrorKind classifies why a turn failed. The set is deliberately small
// and closed: consumers (e.g. a future error-path that formulates a user
// reply) switch on it instead of parsing the error text.
type TurnErrorKind string

const (
	// KindUpstreamAPI: the Messages API call failed (HTTP status, transport).
	KindUpstreamAPI TurnErrorKind = "upstream-api"
	// KindTimeout: a call exceeded the per-call deadline.
	KindTimeout TurnErrorKind = "timeout"
	// KindEmptyReply: the model ended its turn with no deliverable text.
	KindEmptyReply TurnErrorKind = "empty-reply"
	// KindCtxCancel: the turn's context was cancelled.
	KindCtxCancel TurnErrorKind = "ctx-cancel"
	// KindInternal: a Kortex-side failure that is not a panic.
	KindInternal TurnErrorKind = "internal"
	// KindPanic: a panic recovered on the turn path.
	KindPanic TurnErrorKind = "panic"
)

// TurnError is the typed error for a failed turn. Error() returns exactly the
// text the pre-typed ad-hoc errors produced (Msg, then ": " and the cause when
// one is wrapped), so the user-visible "kortex: <err>" line is unchanged.
//
// Unwrap exposes Cause for errors.Is/As. Deadline errors deliberately carry NO
// Cause: the raw "context deadline exceeded" chain misled a downstream error
// classifier into reporting a plain timeout as a context overflow (#5994), so
// it stays out of the chain and the classification lives in Kind instead
// (errors.As to *TurnError, then Kind == KindTimeout).
type TurnError struct {
	Kind  TurnErrorKind
	Msg   string
	Cause error
	// OmitCause keeps Cause out of Error() (it stays reachable via Unwrap).
	// Used when the cause must not reach the user-visible text (panics).
	OmitCause bool
}

func (e *TurnError) Error() string {
	if e.Cause == nil || e.OmitCause {
		return e.Msg
	}
	if e.Msg == "" {
		return e.Cause.Error()
	}
	return e.Msg + ": " + e.Cause.Error()
}

func (e *TurnError) Unwrap() error { return e.Cause }

// KindOf returns the TurnErrorKind of err's chain, or "" if it has none.
func KindOf(err error) TurnErrorKind {
	var te *TurnError
	if errors.As(err, &te) {
		return te.Kind
	}
	return ""
}

// NewPanicError builds the error for a recovered panic. Its text is short and
// non-leaky because it reaches the user as "kortex: <err>"; the panic value
// and stack belong in the diagnostics stream, not here.
func NewPanicError(value any) *TurnError {
	return &TurnError{
		Kind:      KindPanic,
		Msg:       "internal error (panic recovered)",
		Cause:     fmt.Errorf("panic: %v", value),
		OmitCause: true,
	}
}

// wrapTurn adds a prefix to err ("prefix: err" text, same as fmt.Errorf with
// %w) while keeping err's kind; a non-TurnError cause is classified internal.
func wrapTurn(prefix string, err error) *TurnError {
	kind := KindOf(err)
	if kind == "" {
		kind = KindInternal
	}
	return &TurnError{Kind: kind, Msg: prefix, Cause: err}
}

// callError wraps a failed Messages call. A per-call deadline (cctx expired
// while the parent ctx is live) becomes a KindTimeout error with no Cause (see
// TurnError, #5994) and text deadlineMsg; otherwise the error wraps err with
// prefix and is classified ctx-cancel or upstream-api.
func callError(ctx context.Context, cctx context.Context, err error, prefix string, deadlineMsg string) *TurnError {
	if cctx.Err() == context.DeadlineExceeded && ctx.Err() == nil {
		return &TurnError{Kind: KindTimeout, Msg: deadlineMsg}
	}
	kind := KindUpstreamAPI
	if errors.Is(err, context.Canceled) {
		kind = KindCtxCancel
	}
	return &TurnError{Kind: kind, Msg: prefix, Cause: err}
}
