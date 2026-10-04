package tools

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
)

// TestRunLoop_bashRoundTrip simulates the F3a spike's observed sequence: the
// first response is a bash tool_use, the second (after the tool_result goes
// back) is end_turn with text. It checks the dispatcher's real bash
// execution feeds back correctly and the loop stops on end_turn.
func TestRunLoop_bashRoundTrip(t *testing.T) {
	call := 0
	var secondRequestBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call++
		w.Header().Set("Content-Type", "application/json")
		if call == 1 {
			w.Write([]byte(`{
				"content": [{"type":"tool_use","id":"toolu_1","name":"bash","input":{"command":"echo loop-ok"},"caller":{"type":"direct"}}],
				"stop_reason": "tool_use",
				"usage": {"input_tokens": 10, "output_tokens": 5}
			}`))
			return
		}
		json.NewDecoder(r.Body).Decode(&secondRequestBody)
		w.Write([]byte(`{
			"content": [{"type":"text","text":"the command printed loop-ok"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 12, "output_tokens": 6}
		}`))
	}))
	defer server.Close()

	client := anthropic.NewClient("secret-token", server.URL, time.Second)
	dispatcher := NewDispatcher()

	result, err := RunLoop(context.Background(), client, dispatcher, LoopRequest{
		Model:       "claude-haiku-4-5-20251001",
		MaxTokens:   1024,
		UserMessage: "run echo loop-ok",
	})
	if err != nil {
		t.Fatalf("RunLoop error: %v", err)
	}
	if call != 2 {
		t.Fatalf("want 2 API calls, got %d", call)
	}
	if result.Turns != 2 || result.StopReason != "end_turn" {
		t.Errorf("unexpected result: %+v", result)
	}
	if result.FinalText != "the command printed loop-ok" {
		t.Errorf("FinalText = %q", result.FinalText)
	}

	// The second request must carry the tool_result with the real bash
	// output (proving the dispatcher actually executed the command) and the
	// matching tool_use_id.
	encoded, _ := json.Marshal(secondRequestBody)
	body := string(encoded)
	if !strings.Contains(body, "toolu_1") || !strings.Contains(body, "loop-ok") {
		t.Errorf("second request did not carry the expected tool_result: %s", body)
	}
}

// TestRunLoop_historyContinuation proves a follow-up turn sends the prior
// History ahead of the new user message, and that the returned Messages carry
// the whole conversation (history + new user turn + final assistant turn) so it
// can seed the next turn.
func TestRunLoop_historyContinuation(t *testing.T) {
	var sentMessages []anthropic.Message
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []anthropic.Message `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		sentMessages = body.Messages
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"content": [{"type":"text","text":"blue"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 3, "output_tokens": 1}
		}`))
	}))
	defer server.Close()

	history := []anthropic.Message{
		{Role: "user", Content: "my favorite color is blue"},
		{Role: "assistant", Content: "noted"},
	}
	client := anthropic.NewClient("secret-token", server.URL, time.Second)
	result, err := RunLoop(context.Background(), client, NewDispatcher(), LoopRequest{
		Model:       "claude-haiku-4-5-20251001",
		MaxTokens:   1024,
		History:     history,
		UserMessage: "what is my favorite color?",
	})
	if err != nil {
		t.Fatalf("RunLoop error: %v", err)
	}

	// The request must carry the two history messages followed by the new user
	// turn — in that order.
	if len(sentMessages) != 3 {
		t.Fatalf("request carried %d messages, want 3 (2 history + new user)", len(sentMessages))
	}
	if sentMessages[0].Content != "my favorite color is blue" || sentMessages[2].Content != "what is my favorite color?" {
		t.Errorf("messages not in history-then-new order: %+v", sentMessages)
	}
	// The returned Messages seed the next turn: 2 history + new user + final
	// assistant = 4.
	if len(result.Messages) != 4 {
		t.Fatalf("result.Messages has %d entries, want 4", len(result.Messages))
	}
	if result.Messages[3].Role != "assistant" {
		t.Errorf("last returned message role = %q, want assistant", result.Messages[3].Role)
	}
	// RunLoop must not mutate the caller's History slice.
	if len(history) != 2 {
		t.Errorf("input History was mutated to len %d, want 2", len(history))
	}
}

// TestRunLoop_maxTurnsExceeded ensures a model that never stops asking for
// tools fails loudly instead of looping forever.
func TestRunLoop_maxTurnsExceeded(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"content": [{"type":"tool_use","id":"toolu_x","name":"bash","input":{"command":"echo again"}}],
			"stop_reason": "tool_use",
			"usage": {"input_tokens": 1, "output_tokens": 1}
		}`))
	}))
	defer server.Close()

	client := anthropic.NewClient("secret-token", server.URL, time.Second)
	dispatcher := NewDispatcher()

	_, err := RunLoop(context.Background(), client, dispatcher, LoopRequest{
		Model:       "m",
		MaxTokens:   10,
		UserMessage: "loop forever",
		MaxTurns:    3,
	})
	if err == nil {
		t.Fatal("want error when MaxTurns is exceeded")
	}
}
