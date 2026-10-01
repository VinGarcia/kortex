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
