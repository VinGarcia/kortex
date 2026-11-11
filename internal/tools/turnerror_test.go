package tools

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
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

func TestNewPanicError(t *testing.T) {
	err := error(NewPanicError("kaboom"))
	if KindOf(err) != KindPanic {
		t.Errorf("KindOf = %q, want %q", KindOf(err), KindPanic)
	}
	if err.Error() != "internal error (panic recovered)" {
		t.Errorf("Error() = %q, want the short non-leaky text", err.Error())
	}
}
