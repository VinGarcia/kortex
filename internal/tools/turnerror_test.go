package tools

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
)

func TestRunLoop_ErrorKinds(t *testing.T) {
	const thinkingOnly = `{"content":[{"type":"thinking","thinking":"","signature":"s"}],"stop_reason":"max_tokens","usage":{"input_tokens":1,"output_tokens":1}}`

	tests := []struct {
		desc         string
		handler      http.HandlerFunc
		hang         bool // handler blocks until the test ends
		callTimeout  time.Duration
		wantKind     TurnErrorKind
		wantText     string // exact Error() text; "" skips the check
		wantDeadline bool   // whether errors.Is(err, context.DeadlineExceeded) must hold
	}{
		{
			desc: "upstream http error",
			handler: func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, "boom", http.StatusInternalServerError)
			},
			wantKind: KindUpstreamAPI,
		},
		{
			desc: "empty reply with non-end_turn stop reason",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte(thinkingOnly))
			},
			wantKind: KindEmptyReply,
			wantText: `tools: tool-loop turn 1: model returned no deliverable text (stop_reason "max_tokens"); raise max_tokens or lower effort`,
		},
		{
			desc:        "per-call deadline",
			hang:        true,
			callTimeout: 50 * time.Millisecond,
			wantKind:    KindTimeout,
			wantText:    "tools: tool-loop turn 1: call exceeded per-call deadline (50ms)",
			// #5994: the deadline chain must stay out of errors.Is.
			wantDeadline: false,
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			teardown := make(chan struct{})
			handler := test.handler
			if test.hang {
				handler = func(w http.ResponseWriter, r *http.Request) {
					select {
					case <-r.Context().Done():
					case <-teardown:
					}
				}
			}
			server := httptest.NewServer(handler)
			t.Cleanup(server.Close)
			// Registered after server.Close so it runs first and releases the handler.
			t.Cleanup(func() { close(teardown) })
			client := anthropic.NewClient("secret-token", server.URL, 0)

			_, err := RunLoop(context.Background(), client, NewDispatcher(), LoopRequest{
				Model: "m", MaxTokens: 10, UserMessage: "x", CallTimeout: test.callTimeout,
			})
			var te *TurnError
			if !errors.As(err, &te) {
				t.Fatalf("err = %v (%T), want *TurnError", err, err)
			}
			if te.Kind != test.wantKind {
				t.Errorf("Kind = %q, want %q", te.Kind, test.wantKind)
			}
			if test.wantText != "" && err.Error() != test.wantText {
				t.Errorf("Error() = %q, want %q", err.Error(), test.wantText)
			}
			if got := errors.Is(err, context.DeadlineExceeded); got != test.wantDeadline {
				t.Errorf("errors.Is(DeadlineExceeded) = %v, want %v", got, test.wantDeadline)
			}
		})
	}
}

func TestRunLoop_UpstreamErrorKeepsAPIErrorChain(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer server.Close()
	client := anthropic.NewClient("secret-token", server.URL, time.Second)

	_, err := RunLoop(context.Background(), client, NewDispatcher(), LoopRequest{Model: "m", MaxTokens: 10, UserMessage: "x"})
	var apiErr *anthropic.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err = %v, want chain containing *anthropic.APIError", err)
	}
}

func TestRunLoop_RecoveryToolCallIsEmptyReplyKind(t *testing.T) {
	var requests [][]map[string]any
	server := emptyEndTurnServer(t, `{"content":[{"type":"tool_use","id":"t9","name":"x","input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`, &requests)
	defer server.Close()
	client := anthropic.NewClient("secret-token", server.URL, time.Second)

	_, err := RunLoop(context.Background(), client, NewDispatcher(), LoopRequest{Model: "m", MaxTokens: 100, UserMessage: "x"})
	if KindOf(err) != KindEmptyReply {
		t.Fatalf("KindOf = %q, want %q (err %v)", KindOf(err), KindEmptyReply, err)
	}
	want := "tools: tool-loop turn 2: model ended its turn with no reply text and called a tool when asked for the reply; this is not a max_tokens problem"
	if err.Error() != want {
		t.Errorf("Error() = %q, want %q", err.Error(), want)
	}
}

func TestRunLoop_CtxCancelKind(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	teardown := make(chan struct{})
	defer close(teardown)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cancel()
		select {
		case <-r.Context().Done():
		case <-teardown:
		}
	}))
	t.Cleanup(server.Close)
	client := anthropic.NewClient("secret-token", server.URL, 0)

	_, err := RunLoop(ctx, client, NewDispatcher(), LoopRequest{Model: "m", MaxTokens: 10, UserMessage: "x"})
	if KindOf(err) != KindCtxCancel {
		t.Fatalf("KindOf = %q, want %q (err %v)", KindOf(err), KindCtxCancel, err)
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("errors.Is(Canceled) = false, want true")
	}
}

const (
	stepToolUse     = `{"content":[{"type":"tool_use","id":"t1","name":"bash","input":{"command":"echo hi"}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`
	stepThinkingEnd = `{"content":[{"type":"thinking","thinking":"","signature":"s"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	stepNoContent   = `{"content":[],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	stepHang        = "HANG"
	stepHTTP500     = "HTTP500"
)

// scriptedServer answers call N with steps[N-1] (the last step repeats). A
// stepHang step blocks until the client gives up or the test ends.
func scriptedServer(t *testing.T, steps []string) *httptest.Server {
	t.Helper()
	teardown := make(chan struct{})
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		step := steps[min(calls, len(steps)-1)]
		calls++
		switch step {
		case stepHang:
			select {
			case <-r.Context().Done():
			case <-teardown:
			}
		case stepHTTP500:
			http.Error(w, "boom", http.StatusInternalServerError)
		default:
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(step))
		}
	}))
	t.Cleanup(server.Close)
	t.Cleanup(func() { close(teardown) }) // runs before server.Close
	return server
}

// TestRunLoop_ErrorTextAndKindPerSite pins the exact Error() text of every
// converted error site to the pre-typed fmt.Errorf formats, plus its Kind.
func TestRunLoop_ErrorTextAndKindPerSite(t *testing.T) {
	const timeout = 50 * time.Millisecond
	tests := []struct {
		desc       string
		steps      []string
		req        LoopRequest
		wantKind   TurnErrorKind
		wantText   string
		wantPrefix string // used instead of wantText when the cause text is not stable
	}{
		{
			desc:     "MaxTurns exceeded",
			steps:    []string{stepToolUse},
			req:      LoopRequest{MaxTurns: 1},
			wantKind: KindInternal,
			wantText: "tools: tool-loop exceeded MaxTurns (1) without reaching a non-tool_use stop_reason",
		},
		{
			desc:     "finalize call deadline",
			steps:    []string{stepToolUse, stepHang},
			req:      LoopRequest{MaxTurns: 1, FinalizeOnBudget: true, CallTimeout: timeout},
			wantKind: KindTimeout,
			wantText: "tools: tool-loop finalize call exceeded per-call deadline (50ms)",
		},
		{
			desc:       "finalize call upstream error",
			steps:      []string{stepToolUse, stepHTTP500},
			req:        LoopRequest{MaxTurns: 1, FinalizeOnBudget: true},
			wantKind:   KindUpstreamAPI,
			wantPrefix: "tools: tool-loop finalize call: ",
		},
		{
			desc:     "finalize empty reply",
			steps:    []string{stepToolUse, stepNoContent},
			req:      LoopRequest{MaxTurns: 1, FinalizeOnBudget: true},
			wantKind: KindEmptyReply,
			wantText: `tools: tool-loop finalize returned no deliverable text (stop_reason "end_turn")`,
		},
		{
			desc:     "recovery call deadline",
			steps:    []string{stepToolUse, stepThinkingEnd, stepHang},
			req:      LoopRequest{CallTimeout: timeout},
			wantKind: KindTimeout,
			wantText: "tools: tool-loop turn 2: empty-reply recovery call exceeded per-call deadline (50ms)",
		},
		{
			desc:       "recovery call upstream error (cause prefix)",
			steps:      []string{stepToolUse, stepThinkingEnd, stepHTTP500},
			req:        LoopRequest{},
			wantKind:   KindUpstreamAPI,
			wantPrefix: "tools: tool-loop turn 2: empty-reply recovery call: ",
		},
		{
			desc:     "recovery asks for a tool",
			steps:    []string{stepToolUse, stepThinkingEnd, stepToolUse},
			req:      LoopRequest{},
			wantKind: KindEmptyReply,
			wantText: "tools: tool-loop turn 2: model ended its turn with no reply text and called a tool when asked for the reply; this is not a max_tokens problem",
		},
		{
			desc:     "recovery produces no text again",
			steps:    []string{stepToolUse, stepThinkingEnd, stepThinkingEnd},
			req:      LoopRequest{},
			wantKind: KindEmptyReply,
			wantText: `tools: tool-loop turn 2: model ended its turn with no reply text and again produced none when asked (stop_reason "end_turn"); this is not a max_tokens problem`,
		},
		{
			desc:       "upstream error on first call",
			steps:      []string{stepHTTP500},
			req:        LoopRequest{},
			wantKind:   KindUpstreamAPI,
			wantPrefix: "tools: tool-loop turn 1: ",
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			server := scriptedServer(t, test.steps)
			client := anthropic.NewClient("secret-token", server.URL, 0)
			req := test.req
			req.Model, req.MaxTokens, req.UserMessage = "m", 10, "x"

			_, err := RunLoop(context.Background(), client, NewDispatcher(), req)
			var te *TurnError
			if !errors.As(err, &te) {
				t.Fatalf("err = %v (%T), want *TurnError", err, err)
			}
			if te.Kind != test.wantKind {
				t.Errorf("Kind = %q, want %q", te.Kind, test.wantKind)
			}
			if test.wantText != "" && err.Error() != test.wantText {
				t.Errorf("Error() = %q, want %q", err.Error(), test.wantText)
			}
			if test.wantPrefix != "" && !strings.HasPrefix(err.Error(), test.wantPrefix) {
				t.Errorf("Error() = %q, want prefix %q", err.Error(), test.wantPrefix)
			}
			if test.wantKind == KindTimeout && errors.Is(err, context.DeadlineExceeded) {
				t.Error("per-call deadline must not expose context.DeadlineExceeded (#5994)")
			}
		})
	}
}

// TestRunLoop_ParentDeadlineIsTimeoutWithCause: when the turn's own ctx
// expires (not the per-call bound) the kind is timeout, and the text/cause are
// the pre-typed "prefix: cause" form.
func TestRunLoop_ParentDeadlineIsTimeoutWithCause(t *testing.T) {
	server := scriptedServer(t, []string{stepHang})
	client := anthropic.NewClient("secret-token", server.URL, 0)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := RunLoop(ctx, client, NewDispatcher(), LoopRequest{Model: "m", MaxTokens: 10, UserMessage: "x"})
	if KindOf(err) != KindTimeout {
		t.Fatalf("KindOf = %q, want %q (err %v)", KindOf(err), KindTimeout, err)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Error("parent-deadline error must keep its cause chain")
	}
	if !strings.HasPrefix(err.Error(), "tools: tool-loop turn 1: ") {
		t.Errorf("Error() = %q, want the prefix: cause form", err.Error())
	}
}

func TestWrapTurn(t *testing.T) {
	inner := &TurnError{Kind: KindEmptyReply, Msg: "inner"}
	got := wrapTurn("outer", inner)
	if got.Error() != "outer: inner" || got.Kind != KindEmptyReply {
		t.Errorf("wrapTurn(TurnError) = %q kind %q, want %q kind %q", got.Error(), got.Kind, "outer: inner", KindEmptyReply)
	}
	if !errors.Is(got, inner) {
		t.Error("wrapTurn must keep the cause in the chain")
	}
	plain := wrapTurn("outer", errors.New("plain"))
	if plain.Error() != "outer: plain" || plain.Kind != KindInternal {
		t.Errorf("wrapTurn(plain) = %q kind %q, want %q kind %q", plain.Error(), plain.Kind, "outer: plain", KindInternal)
	}
}
