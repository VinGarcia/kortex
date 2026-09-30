// Package anthropic is a minimal client for the Anthropic Messages API
// authenticated with an OAuth Bearer token (Claude subscription setup-token
// flow). It implements exactly what the facets need today: one
// non-streaming messages call with a system prompt.
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const (
	defaultBaseURL   = "https://api.anthropic.com"
	messagesPath     = "/v1/messages"
	anthropicVersion = "2023-06-01"
	// oauthBeta is the beta header value that makes the API accept
	// subscription OAuth Bearer tokens instead of x-api-key.
	oauthBeta = "oauth-2025-04-20"
)

// Client calls the Messages API. Construct with NewClient.
type Client struct {
	baseURL    string
	token      string
	httpClient *http.Client
}

// NewClient builds a client holding the OAuth token. baseURL "" means the
// real Anthropic API; tests point it at a local server. timeout bounds each
// whole HTTP call (callers can bound tighter per-call via context); 0 means
// no client-level timeout.
func NewClient(token string, baseURL string, timeout time.Duration) *Client {
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Client{
		baseURL:    strings.TrimSuffix(baseURL, "/"),
		token:      token,
		httpClient: &http.Client{Timeout: timeout},
	}
}

// Message is one conversation message of a request.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// MessageRequest is a non-streaming Messages API call.
type MessageRequest struct {
	Model     string
	System    string
	MaxTokens int
	Messages  []Message
}

// Usage carries the token counters of a response.
type Usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

// MessageResponse is the decoded non-streaming response.
type MessageResponse struct {
	// Text is the concatenation of the response's text content blocks.
	Text       string
	StopReason string
	Usage      Usage
}

// APIError is a non-2xx answer from the API. Status is always set; Type and
// Message come from the error body when it decodes.
type APIError struct {
	Status  int
	Type    string
	Message string
}

func (e *APIError) Error() string {
	if e.Type == "" && e.Message == "" {
		return fmt.Sprintf("anthropic: http %d", e.Status)
	}
	return fmt.Sprintf("anthropic: http %d: %s: %s", e.Status, e.Type, e.Message)
}

type wireRequest struct {
	Model     string    `json:"model"`
	System    string    `json:"system,omitempty"`
	MaxTokens int       `json:"max_tokens"`
	Messages  []Message `json:"messages"`
}

type wireResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	Usage      Usage  `json:"usage"`
}

type wireError struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// CreateMessage performs one non-streaming Messages call. Transport
// failures come back wrapped; API failures come back as *APIError.
func (c *Client) CreateMessage(ctx context.Context, req MessageRequest) (MessageResponse, error) {
	body, err := json.Marshal(wireRequest{
		Model:     req.Model,
		System:    req.System,
		MaxTokens: req.MaxTokens,
		Messages:  req.Messages,
	})
	if err != nil {
		return MessageResponse{}, fmt.Errorf("anthropic: encoding request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+messagesPath, bytes.NewReader(body))
	if err != nil {
		return MessageResponse{}, fmt.Errorf("anthropic: building request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.token)
	httpReq.Header.Set("anthropic-version", anthropicVersion)
	httpReq.Header.Set("anthropic-beta", oauthBeta)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return MessageResponse{}, fmt.Errorf("anthropic: request failed: %w", err)
	}
	defer resp.Body.Close()

	// The cap keeps a misbehaving endpoint from ballooning memory; real
	// annotator responses are a few KB of JSON tags.
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return MessageResponse{}, fmt.Errorf("anthropic: reading response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		apiErr := &APIError{Status: resp.StatusCode}
		var we wireError
		if json.Unmarshal(respBody, &we) == nil {
			apiErr.Type = we.Error.Type
			apiErr.Message = we.Error.Message
		}
		return MessageResponse{}, apiErr
	}

	var wr wireResponse
	if err := json.Unmarshal(respBody, &wr); err != nil {
		return MessageResponse{}, fmt.Errorf("anthropic: decoding response: %w", err)
	}
	var text strings.Builder
	for _, block := range wr.Content {
		if block.Type == "text" {
			text.WriteString(block.Text)
		}
	}
	return MessageResponse{
		Text:       text.String(),
		StopReason: wr.StopReason,
		Usage:      wr.Usage,
	}, nil
}
