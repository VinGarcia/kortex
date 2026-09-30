package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
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
	if gotBody["model"] != "claude-haiku-4-5-20251001" || gotBody["system"] != "system prompt" || gotBody["max_tokens"] != float64(2048) {
		t.Errorf("unexpected request body: %v", gotBody)
	}
	if resp.Text != `[{"investment":0}]` {
		t.Errorf("text = %q (text blocks should concatenate)", resp.Text)
	}
	if resp.StopReason != "end_turn" || resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
		t.Errorf("unexpected response: %+v", resp)
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
