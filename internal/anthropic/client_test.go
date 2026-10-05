package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestCreateMessage_success(t *testing.T) {
	var gotPath string
	var gotHeaders http.Header
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotHeaders = r.Header.Clone()
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"content": [{"type":"text","text":"[{\"investment\":0"}, {"type":"text","text":"}]"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 2, "cache_creation_input_tokens": 1}
		}`))
	}))
	defer server.Close()

	client := NewClient("secret-token", server.URL, time.Second)
	resp, err := client.CreateMessage(context.Background(), MessageRequest{
		Model:     "claude-haiku-4-5-20251001",
		System:    "system prompt",
		MaxTokens: 2048,
		Messages:  []Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatal(err)
	}

	if gotPath != "/v1/messages" {
		t.Errorf("path = %q, want /v1/messages", gotPath)
	}
	headerChecks := map[string]string{
		"Authorization":     "Bearer secret-token",
		"Anthropic-Version": "2023-06-01",
		"Anthropic-Beta":    "oauth-2025-04-20",
		"Content-Type":      "application/json",
	}
	for name, want := range headerChecks {
		if got := gotHeaders.Get(name); got != want {
			t.Errorf("header %s = %q, want %q", name, got, want)
		}
	}
	// The system must be the block-array form with the Claude Code preamble as
	// its OWN first block (subscription-token abuse check; see systemBlock) and
	// the caller's prompt as the second.
	wantSystem := []any{
		map[string]any{"type": "text", "text": claudeCodePreamble},
		map[string]any{"type": "text", "text": "system prompt"},
	}
	if !reflect.DeepEqual(gotBody["system"], wantSystem) {
		t.Errorf("system = %v, want %v", gotBody["system"], wantSystem)
	}
	if gotBody["model"] != "claude-haiku-4-5-20251001" || gotBody["max_tokens"] != float64(2048) {
		t.Errorf("unexpected request body: %v", gotBody)
	}
	if resp.Text != `[{"investment":0}]` {
		t.Errorf("text = %q (text blocks should concatenate)", resp.Text)
	}
	if resp.StopReason != "end_turn" || resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
		t.Errorf("unexpected response: %+v", resp)
	}
}

// TestCreateMessage_toolUse reproduces the F3a spike's round trip: a
// request declaring a bash tool gets back a tool_use block (with the
// observed-but-unmodeled "caller" field), and echoing that block's Raw back
// as the assistant turn alongside a tool_result produces the exact
// messages[2] shape the spike proved the API accepts.
func TestCreateMessage_toolUse(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decoding request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"content": [{"type":"tool_use","id":"toolu_01abc","name":"bash","input":{"command":"pwd"},"caller":{"type":"direct"}}],
			"stop_reason": "tool_use",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`))
	}))
	defer server.Close()

	client := NewClient("secret-token", server.URL, time.Second)
	schema, _ := json.Marshal(map[string]any{
		"type":       "object",
		"properties": map[string]any{"command": map[string]any{"type": "string"}},
		"required":   []string{"command"},
	})
	resp, err := client.CreateMessage(context.Background(), MessageRequest{
		Model:     "claude-haiku-4-5-20251001",
		MaxTokens: 1024,
		Messages:  []Message{{Role: "user", Content: "run pwd"}},
		Tools:     []ToolDef{{Name: "bash", Description: "run a shell command", InputSchema: schema}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotToolsRaw, ok := gotBody["tools"]; !ok || gotToolsRaw == nil {
		t.Fatalf("request did not carry tools: %v", gotBody)
	}

	if resp.StopReason != "tool_use" {
		t.Fatalf("stop_reason = %q, want tool_use", resp.StopReason)
	}
	toolUses := resp.ToolUseBlocks()
	if len(toolUses) != 1 {
		t.Fatalf("want 1 tool_use block, got %d", len(toolUses))
	}
	tu := toolUses[0]
	if tu.ID != "toolu_01abc" || tu.Name != "bash" {
		t.Errorf("unexpected tool_use block: %+v", tu)
	}
	var input struct {
		Command string `json:"command"`
	}
	if err := json.Unmarshal(tu.Input, &input); err != nil || input.Command != "pwd" {
		t.Errorf("tool_use.input.command = %+v, err=%v, want pwd", input, err)
	}

	// Build the follow-up turn the way the tool-loop must: echo the raw
	// assistant content back verbatim, then a user turn with the
	// tool_result keyed to the tool_use id.
	followUp := []Message{
		{Role: "user", Content: "run pwd"},
		{Role: "assistant", Content: resp.RawContent()},
		{Role: "user", Content: []ToolResultBlock{NewToolResultBlock(tu.ID, "/home/vingarcia", false)}},
	}
	encoded, err := json.Marshal(followUp)
	if err != nil {
		t.Fatalf("follow-up turn did not marshal: %v", err)
	}
	if !strings.Contains(string(encoded), `"tool_use_id":"toolu_01abc"`) {
		t.Errorf("follow-up turn missing tool_result with matching tool_use_id: %s", encoded)
	}
	if !strings.Contains(string(encoded), `"caller":{"type":"direct"}`) {
		t.Errorf("follow-up turn did not preserve the unmodeled caller field: %s", encoded)
	}
}

// TestCreateMessage_wireLog proves SetWireLog captures both the request and the
// response body for diagnosis — the native-path equivalent of the passthrough's
// trafficLogger — while never writing the OAuth token, which lives only in the
// Authorization header and never in a logged body.
func TestCreateMessage_wireLog(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	var log strings.Builder
	client := NewClient("super-secret-token", server.URL, time.Second)
	client.SetWireLog(&log)
	_, err := client.CreateMessage(context.Background(), MessageRequest{
		Model:     "claude-fable-5",
		MaxTokens: 1024,
		Messages:  []Message{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatal(err)
	}

	logged := log.String()
	// Both directions must be present, each tagged so the two bodies are
	// distinguishable in the file.
	if !strings.Contains(logged, "request ") || !strings.Contains(logged, `"model":"claude-fable-5"`) {
		t.Errorf("wire log missing the request line: %q", logged)
	}
	if !strings.Contains(logged, "response ") || !strings.Contains(logged, `"stop_reason":"end_turn"`) {
		t.Errorf("wire log missing the response line: %q", logged)
	}
	// The token must never reach the log: it travels in the Authorization header,
	// which is not part of any logged body.
	if strings.Contains(logged, "super-secret-token") {
		t.Errorf("wire log leaked the OAuth token: %q", logged)
	}
	// Two lines, one per direction.
	if n := strings.Count(strings.TrimSpace(logged), "\n"); n != 1 {
		t.Errorf("wire log has %d newlines, want 1 (two lines: request + response)", n)
	}
}

func TestCreateMessage_errors(t *testing.T) {
	tests := []struct {
		desc        string
		status      int
		body        string
		wantType    string
		wantMessage string
	}{
		{
			desc:        "structured api error",
			status:      401,
			body:        `{"type":"error","error":{"type":"authentication_error","message":"invalid bearer token"}}`,
			wantType:    "authentication_error",
			wantMessage: "invalid bearer token",
		},
		{
			desc:   "non-json error body still yields typed error with status",
			status: 529,
			body:   "overloaded",
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(test.status)
				w.Write([]byte(test.body))
			}))
			defer server.Close()

			client := NewClient("t", server.URL, time.Second)
			_, err := client.CreateMessage(context.Background(), MessageRequest{Model: "m", MaxTokens: 10, Messages: []Message{{Role: "user", Content: "x"}}})
			var apiErr *APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("want *APIError, got %v", err)
			}
			if apiErr.Status != test.status || apiErr.Type != test.wantType || apiErr.Message != test.wantMessage {
				t.Errorf("got %+v", apiErr)
			}
		})
	}
}

func TestCreateMessage_contextTimeout(t *testing.T) {
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-release
	}))
	defer server.Close()
	defer close(release)

	client := NewClient("t", server.URL, 0)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err := client.CreateMessage(ctx, MessageRequest{Model: "m", MaxTokens: 10, Messages: []Message{{Role: "user", Content: "x"}}})
	if err == nil || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("want deadline exceeded, got %v", err)
	}
}

func TestCreateMessage_malformedSuccessBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("not json"))
	}))
	defer server.Close()

	client := NewClient("t", server.URL, time.Second)
	_, err := client.CreateMessage(context.Background(), MessageRequest{Model: "m", MaxTokens: 10, Messages: []Message{{Role: "user", Content: "x"}}})
	if err == nil || !strings.Contains(err.Error(), "decoding response") {
		t.Fatalf("want decoding error, got %v", err)
	}
}
