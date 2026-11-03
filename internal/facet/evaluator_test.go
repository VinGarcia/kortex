package facet

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

// TestAnthropicEvaluator_oneShotMessageNeverMarked proves Evaluate sends its
// single user message unmarked even when it is large enough to otherwise
// clear the model's cacheable floor: Evaluate is a one-shot call with no
// continuing history, so a cache_control breakpoint here would be a pure
// cache write with no matching read, strictly worse than not caching — see
// AnthropicEvaluator.Evaluate's comment on why it uses EphemeralTail instead
// of Messages.
func TestAnthropicEvaluator_oneShotMessageNeverMarked(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"content":[{"type":"text","text":"[]"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	client := anthropic.NewClient("secret", server.URL, time.Second)
	evaluator := NewAnthropicEvaluator(client, "claude-opus-4-8")

	userMessage := "evaluate this: " + strings.Repeat("padding well past the cacheable floor. ", 1000)
	if _, err := evaluator.Evaluate(context.Background(), "system", userMessage); err != nil {
		t.Fatalf("Evaluate error: %v", err)
	}

	messages, ok := gotBody["messages"].([]any)
	if !ok || len(messages) != 1 {
		t.Fatalf("messages = %v, want exactly one", gotBody["messages"])
	}
	encoded, err := json.Marshal(messages[0])
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}
	if strings.Contains(string(encoded), "cache_control") {
		t.Errorf("one-shot evaluator message carries a cache breakpoint, want none: %s", encoded)
	}
	if !strings.Contains(string(encoded), userMessage) {
		t.Errorf("request did not carry the user message verbatim: %s", encoded)
	}
}
