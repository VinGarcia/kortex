package tools

import (
	"context"
	"encoding/json"
	"io"
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

// TestRunLoop_emptyFinalTurnFailsClosed proves the backstop for the live
// 2026-10-04 failure: a thinking model that spends its whole MaxTokens budget on
// thinking returns a final turn (stop_reason "max_tokens") carrying only a
// thinking block and zero text. OpenClaw rejects an empty assistant turn, and
// the turn cannot be continued (fable 400s a last-assistant prefill), so RunLoop
// must fail closed with an error naming the stop_reason — never return an empty
// FinalText that a caller would emit as an empty turn.
func TestRunLoop_emptyFinalTurnFailsClosed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// A thinking-only turn: a signature-bearing thinking block, no text block,
		// stopped because max_tokens was hit mid-thought.
		w.Write([]byte(`{
			"content": [{"type":"thinking","thinking":"","signature":"sig-abc"}],
			"stop_reason": "max_tokens",
			"usage": {"input_tokens": 10, "output_tokens": 8192}
		}`))
	}))
	defer server.Close()

	client := anthropic.NewClient("secret-token", server.URL, time.Second)
	_, err := RunLoop(context.Background(), client, NewDispatcher(), LoopRequest{
		Model:       "claude-fable-5",
		MaxTokens:   8192,
		UserMessage: "think hard",
	})
	if err == nil {
		t.Fatal("want a fail-closed error when the final turn has no text")
	}
	// The error must name the stop_reason so the cause is diagnosable.
	if !strings.Contains(err.Error(), "max_tokens") {
		t.Errorf("error = %q, want it to name the max_tokens stop_reason", err)
	}
}

// TestRunLoop_echoesThinkingBlockVerbatim proves the native path preserves a
// signature-bearing thinking block when it echoes an intermediate tool_use turn
// back to the API. Anthropic requires thinking blocks replayed WITHIN a tool-use
// cycle to keep their signature unchanged; RunLoop echoes raw content, so the
// signature must survive into the follow-up request verbatim.
func TestRunLoop_echoesThinkingBlockVerbatim(t *testing.T) {
	call := 0
	var secondRequestBody string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call++
		w.Header().Set("Content-Type", "application/json")
		if call == 1 {
			// fable's real shape: a thinking block (carrying a signature) precedes
			// the tool_use block in the same assistant turn.
			w.Write([]byte(`{
				"content": [
					{"type":"thinking","thinking":"","signature":"sig-xyz-123"},
					{"type":"tool_use","id":"toolu_1","name":"bash","input":{"command":"echo hi"},"caller":{"type":"direct"}}
				],
				"stop_reason": "tool_use",
				"usage": {"input_tokens": 10, "output_tokens": 5}
			}`))
			return
		}
		body, _ := io.ReadAll(r.Body)
		secondRequestBody = string(body)
		w.Write([]byte(`{
			"content": [{"type":"text","text":"done"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 12, "output_tokens": 6}
		}`))
	}))
	defer server.Close()

	client := anthropic.NewClient("secret-token", server.URL, time.Second)
	_, err := RunLoop(context.Background(), client, NewDispatcher(), LoopRequest{
		Model:       "claude-fable-5",
		MaxTokens:   1024,
		UserMessage: "run echo hi",
	})
	if err != nil {
		t.Fatalf("RunLoop error: %v", err)
	}
	if call != 2 {
		t.Fatalf("want 2 API calls, got %d", call)
	}
	// The echoed assistant turn must carry the thinking block with its signature
	// byte-for-byte: a dropped or altered signature would 400 the resume.
	if !strings.Contains(secondRequestBody, `"signature":"sig-xyz-123"`) {
		t.Errorf("follow-up request dropped the thinking block signature: %s", secondRequestBody)
	}
}

// TestRunLoop_forwardsEffort proves LoopRequest.Effort reaches the wire as
// output_config.effort, so OpenClaw's --effort bounds how much of the token
// budget a thinking model spends before answering.
func TestRunLoop_forwardsEffort(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	client := anthropic.NewClient("secret-token", server.URL, time.Second)
	_, err := RunLoop(context.Background(), client, NewDispatcher(), LoopRequest{
		Model:       "claude-fable-5",
		MaxTokens:   1024,
		Effort:      "medium",
		UserMessage: "hi",
	})
	if err != nil {
		t.Fatalf("RunLoop error: %v", err)
	}
	oc, ok := gotBody["output_config"].(map[string]any)
	if !ok {
		t.Fatalf("request did not carry output_config: %v", gotBody)
	}
	if oc["effort"] != "medium" {
		t.Errorf("output_config.effort = %v, want medium", oc["effort"])
	}
}

// TestRunLoop_ephemeralContextSentbutNotPersisted is the tools-seam assertion
// for the ephemeral tone digest: EphemeralContext must be appended as the LAST
// user message of EVERY core call (so the core sees it as the final item of its
// context), exactly once per call (never accumulating across tool rounds), and
// must NEVER appear in the returned Messages (canonical history). The loop runs
// two rounds (tool_use then end_turn) so both the intermediate and the final
// call are checked.
func TestRunLoop_ephemeralContextSentButNotPersisted(t *testing.T) {
	const digest = "<<<TOM_DA_CONVERSA_INICIO>>>\n- alegria\n<<<TOM_DA_CONVERSA_FIM>>>"
	call := 0
	var sentPerCall [][]anthropic.Message
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call++
		var body struct {
			Messages []anthropic.Message `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		sentPerCall = append(sentPerCall, body.Messages)
		w.Header().Set("Content-Type", "application/json")
		if call == 1 {
			w.Write([]byte(`{
				"content": [{"type":"tool_use","id":"toolu_1","name":"bash","input":{"command":"echo hi"},"caller":{"type":"direct"}}],
				"stop_reason": "tool_use",
				"usage": {"input_tokens": 1, "output_tokens": 1}
			}`))
			return
		}
		w.Write([]byte(`{
			"content": [{"type":"text","text":"done"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 1, "output_tokens": 1}
		}`))
	}))
	defer server.Close()

	client := anthropic.NewClient("secret-token", server.URL, time.Second)
	result, err := RunLoop(context.Background(), client, NewDispatcher(), LoopRequest{
		Model:            "m",
		MaxTokens:        1024,
		UserMessage:      "oi",
		EphemeralContext: digest,
	})
	if err != nil {
		t.Fatalf("RunLoop error: %v", err)
	}
	if call != 2 {
		t.Fatalf("want 2 API calls, got %d", call)
	}

	// Every call must end with the ephemeral digest as its final message, and
	// carry it exactly once (no accumulation as the loop grows the array).
	for i, msgs := range sentPerCall {
		if len(msgs) == 0 {
			t.Fatalf("call %d sent no messages", i+1)
		}
		last := msgs[len(msgs)-1]
		if last.Role != "user" || last.Content != digest {
			t.Errorf("call %d: last message = %+v, want the ephemeral digest as a trailing user message", i+1, last)
		}
		occurrences := 0
		for _, m := range msgs {
			if m.Content == digest {
				occurrences++
			}
		}
		if occurrences != 1 {
			t.Errorf("call %d: digest appeared %d times, want exactly 1 (no accumulation)", i+1, occurrences)
		}
	}

	// The digest must NOT be persisted into canonical history: no returned
	// message may carry it.
	for i, m := range result.Messages {
		if m.Content == digest {
			t.Errorf("returned Messages[%d] carries the ephemeral digest; it must stay out of canonical history", i)
		}
	}
}

// TestRunLoop_emptyEphemeralContextChangesNothing proves the fail-open default:
// an empty EphemeralContext appends no extra message, so the wire shape is
// byte-for-byte what it was before the field existed.
func TestRunLoop_emptyEphemeralContextChangesNothing(t *testing.T) {
	var sent []anthropic.Message
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []anthropic.Message `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		sent = body.Messages
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	client := anthropic.NewClient("secret-token", server.URL, time.Second)
	_, err := RunLoop(context.Background(), client, NewDispatcher(), LoopRequest{
		Model:       "m",
		MaxTokens:   1024,
		UserMessage: "oi",
	})
	if err != nil {
		t.Fatalf("RunLoop error: %v", err)
	}
	// Only the single user message: no extra trailing ephemeral message.
	if len(sent) != 1 {
		t.Fatalf("sent %d messages, want 1 (just the user turn, no ephemeral append)", len(sent))
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

// TestRunLoop_callTimeoutFailsClosed proves a hung CreateMessage is bounded by
// LoopRequest.CallTimeout rather than blocking the turn forever: when the
// endpoint never responds, RunLoop returns a deadline error PROMPTLY (bounded
// completion near the tiny injected timeout) instead of hanging. Fail-closed —
// the loop has no fallback output, so the timeout surfaces as an error return
// naming the per-call deadline, not a silent result. The client is built with
// no HTTP timeout (the production wiring from main.go), so the only bound on
// the hung call is the per-call deadline this test exercises.
func TestRunLoop_callTimeoutFailsClosed(t *testing.T) {
	const callTimeout = 100 * time.Millisecond

	// The handler hangs until the client cancels (the per-call deadline firing)
	// or teardown closes. teardown is closed before server.Close runs (defer
	// beats t.Cleanup) so the hung handler never blocks server shutdown even if
	// it didn't observe the client-side cancellation.
	teardown := make(chan struct{})
	defer close(teardown)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-teardown:
		}
	}))
	t.Cleanup(server.Close)

	client := anthropic.NewClient("secret-token", server.URL, 0)

	start := time.Now()
	_, err := RunLoop(context.Background(), client, NewDispatcher(), LoopRequest{
		Model:       "m",
		MaxTokens:   10,
		UserMessage: "hang please",
		CallTimeout: callTimeout,
	})
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("want an error when the core-model call exceeds CallTimeout")
	}
	// Bounded: RunLoop returned far sooner than the hung endpoint would ever
	// allow. The ceiling is generous versus the 100ms deadline yet nowhere near
	// the forever a missing bound would produce.
	if elapsed > 5*time.Second {
		t.Fatalf("RunLoop took %s, want bounded completion near the %s call deadline", elapsed, callTimeout)
	}
	// Fail-closed: the error names the per-call deadline so the cause is clear.
	if !strings.Contains(err.Error(), "per-call deadline") {
		t.Errorf("error = %q, want it to name the per-call deadline", err)
	}
}
