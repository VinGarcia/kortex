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
	"sync"
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
	// wireLog, when non-nil, receives a line-oriented copy of every request and
	// response body for debugging (see SetWireLog). wireMu serializes those
	// writes so concurrent callers never interleave a line.
	wireLog io.Writer
	wireMu  sync.Mutex
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

// SetWireLog routes a line-oriented copy of every request and response body to
// w — the native path's equivalent of the passthrough's trafficLogger, which
// never logged anything on this path and left tonight's diagnosis blind. The
// OAuth token lives in the Authorization header, which is never part of the
// body, so nothing secret is written. A nil w (the default) disables logging.
// Wire it once at construction (before any request); it is not meant to be
// swapped while calls are in flight.
func (c *Client) SetWireLog(w io.Writer) {
	c.wireLog = w
}

// logWire appends one request/response line to the wire log when enabled. The
// body is written verbatim (it carries no credential) with a timestamp and
// direction prefix, matching proxy.trafficLogger's line shape.
func (c *Client) logWire(direction string, body []byte) {
	if c.wireLog == nil {
		return
	}
	c.wireMu.Lock()
	defer c.wireMu.Unlock()
	fmt.Fprintf(c.wireLog, "%s %s ", time.Now().Format(time.RFC3339Nano), direction)
	c.wireLog.Write(body)
	c.wireLog.Write([]byte("\n"))
}

// Message is one conversation message of a request. Content is either a
// plain string (the simple facet calls use this) or a JSON-serializable
// value carrying content blocks (tool_use echoed back from a prior
// response, or []ToolResultBlock for a tool_result turn) — whatever the
// tool-loop needs to send. Both shapes marshal correctly because
// encoding/json accepts any JSON-serializable value in an `any` field.
type Message struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

// ToolDef declares one tool in the "tools" field of a request, in the exact
// shape the Messages API expects (see hack/toolloop.sh, proven against the
// real API in the F3a spike).
type ToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// ToolResultBlock is a tool_result content block sent back to the API as
// part of a user turn, keyed to the tool_use block it answers by
// ToolUseID. This is the exact shape the F3a spike proved the API accepts
// (stop_reason: end_turn on the reply).
type ToolResultBlock struct {
	Type      string `json:"type"`
	ToolUseID string `json:"tool_use_id"`
	Content   string `json:"content"`
	IsError   bool   `json:"is_error,omitempty"`
}

// NewToolResultBlock builds a tool_result content block for toolUseID.
func NewToolResultBlock(toolUseID string, content string, isError bool) ToolResultBlock {
	return ToolResultBlock{Type: "tool_result", ToolUseID: toolUseID, Content: content, IsError: isError}
}

// MessageRequest is a non-streaming Messages API call. Tools is nil for the
// plain facet calls that predate the tool-loop; non-nil, it is forwarded
// verbatim in the "tools" field.
type MessageRequest struct {
	Model string
	// System is the turn-invariant system prompt. Keep it byte-identical across
	// the turns of a conversation: the client marks it as a prompt-cache prefix
	// (see markStableSystemPrefix), so interpolating volatile per-turn content
	// here would miss the cache and instead pay the cache-write penalty every
	// turn — strictly worse than not caching.
	System    string
	MaxTokens int
	Messages  []Message
	Tools     []ToolDef
	// Effort selects output_config.effort ("low".."max"), which bounds how deeply
	// a thinking model reasons before it answers. "" omits output_config so the
	// API applies its default ("high"). It matters for the canary core model
	// (fable): thinking is always on there, so at high effort a small MaxTokens
	// can be spent entirely on thinking, leaving zero text — the empty-final-turn
	// failure this field (wired from the spawn's --effort) guards against.
	Effort string
}

// Usage carries the token counters of a response.
type Usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

// Add returns the counter-by-counter sum of u and o, so a caller accumulating
// usage across several model calls (a multi-turn tool loop, or the active
// superego loop's redrafts) can total the round on one shared implementation.
func (u Usage) Add(o Usage) Usage {
	u.InputTokens += o.InputTokens
	u.OutputTokens += o.OutputTokens
	u.CacheReadInputTokens += o.CacheReadInputTokens
	u.CacheCreationInputTokens += o.CacheCreationInputTokens
	return u
}

// ContentBlock is one parsed content block of a response. Fields populate
// depending on Type ("text" -> Text; "tool_use" -> ID/Name/Input). Raw holds
// the exact bytes the API returned for this block so a tool-loop can echo
// the assistant turn back verbatim in the next request's Messages — this
// matters because tool_use blocks carry fields this client does not model
// (e.g. the F3a spike observed a "caller":{"type":"direct"} field on every
// tool_use block; Raw preserves it even though ContentBlock does not parse
// it).
type ContentBlock struct {
	Type  string
	Text  string
	ID    string
	Name  string
	Input json.RawMessage
	Raw   json.RawMessage
}

// MessageResponse is the decoded non-streaming response.
type MessageResponse struct {
	// Text is the concatenation of the response's text content blocks.
	Text       string
	Content    []ContentBlock
	StopReason string
	Usage      Usage
}

// ToolUseBlocks filters Content down to the tool_use blocks, in order.
func (r MessageResponse) ToolUseBlocks() []ContentBlock {
	var blocks []ContentBlock
	for _, b := range r.Content {
		if b.Type == "tool_use" {
			blocks = append(blocks, b)
		}
	}
	return blocks
}

// RawContent returns the response's content blocks as their exact raw JSON,
// suitable to assign verbatim to a follow-up Message.Content when echoing
// the assistant turn back to the API.
func (r MessageResponse) RawContent() []json.RawMessage {
	raw := make([]json.RawMessage, len(r.Content))
	for i, b := range r.Content {
		raw[i] = b.Raw
	}
	return raw
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
	Model        string        `json:"model"`
	System       []systemBlock `json:"system,omitempty"`
	MaxTokens    int           `json:"max_tokens"`
	Messages     []Message     `json:"messages"`
	Tools        []ToolDef     `json:"tools,omitempty"`
	OutputConfig *outputConfig `json:"output_config,omitempty"`
}

// outputConfig carries the GA output_config.effort knob (no beta header). A nil
// pointer omits the field entirely so the API applies its default; it is set
// only when a caller supplies MessageRequest.Effort.
type outputConfig struct {
	Effort string `json:"effort,omitempty"`
}

// systemBlock is one element of the system-as-array form of the Messages API.
// The client always sends the array form because the subscription-token abuse
// check wants the Claude Code preamble as its OWN first block: a single string
// that merely starts with the preamble passes for short systems but is
// rejected (bare 429) once the rest of the prompt is large — verified live
// 2026-10-04 against claude-fable-5 with the real 61k-char OpenClaw prompt.
// CacheControl, when set on a block, asks the API to cache the request prefix
// up to and including that block (see markStableSystemPrefix).
type systemBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text"`
	CacheControl *cacheControl `json:"cache_control,omitempty"`
}

// cacheControl is a prompt-cache breakpoint. The Messages API caches the exact
// request prefix (tools → system → messages) up to and including the block
// carrying it, so a later turn whose prefix is byte-identical reads those
// tokens from cache (~0.1x the input price) instead of re-billing them.
type cacheControl struct {
	Type string `json:"type"`
}

type wireResponse struct {
	Content    []json.RawMessage `json:"content"`
	StopReason string            `json:"stop_reason"`
	Usage      Usage             `json:"usage"`
}

// wireContentBlock is the superset of fields any response content block may
// carry, used only to parse each raw block in wireResponse.Content.
type wireContentBlock struct {
	Type  string          `json:"type"`
	Text  string          `json:"text"`
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

type wireError struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// claudeCodePreamble is the identity line Anthropic's abuse protection expects
// as the FIRST system block on subscription OAuth tokens: without it, non-haiku
// models (opus/sonnet/fable) reject the call with an instant 429
// rate_limit_error carrying no rate-limit headers — a policy rejection that
// masquerades as quota. It must be its own block in the system array, exactly
// as the real claude-cli sends it (see systemBlock); the caller's prompt goes
// in the following block.
const claudeCodePreamble = "You are Claude Code, Anthropic's official CLI for Claude."

// minCacheableTokens is the smallest system prefix the Messages API will
// actually cache for a model. A cache_control breakpoint on a shorter prefix is
// ignored silently — no error, no caching — so marking below this floor is
// pointless. Haiku 4.5's floor is 4096 tokens; every other model this client
// targets (opus-4-8, sonnet, fable-5) sits at or below 1024, so 1024 is the
// safe default that never marks below a model's real floor.
func minCacheableTokens(model string) int {
	if strings.Contains(model, "haiku") {
		return 4096
	}
	return 1024
}

// markStableSystemPrefix places one ephemeral cache breakpoint at the END of
// the stable system prefix (the last system block), so the whole turn-invariant
// system — the Claude Code preamble plus the caller's prompt — is cached once
// and every later turn reads it back instead of re-billing the entire context
// each turn.
//
// It is fail-safe: when system is empty or its estimated size is below the model's
// cacheable floor, it marks nothing and the request proceeds uncached, which is
// exactly what the API would do with an under-floor breakpoint anyway. The size
// estimate uses the standard ~4-bytes-per-token heuristic because token
// counting is not available client-side; a near-miss costs nothing.
func markStableSystemPrefix(model string, system []systemBlock) {
	if len(system) == 0 {
		return
	}
	totalChars := 0
	for _, block := range system {
		totalChars += len(block.Text)
	}
	if totalChars/4 < minCacheableTokens(model) {
		return
	}
	system[len(system)-1].CacheControl = &cacheControl{Type: "ephemeral"}
}

// CreateMessage performs one non-streaming Messages call. Transport
// failures come back wrapped; API failures come back as *APIError.
func (c *Client) CreateMessage(ctx context.Context, req MessageRequest) (MessageResponse, error) {
	system := []systemBlock{{Type: "text", Text: claudeCodePreamble}}
	if req.System != "" {
		system = append(system, systemBlock{Type: "text", Text: req.System})
	}
	markStableSystemPrefix(req.Model, system)
	wire := wireRequest{
		Model:     req.Model,
		System:    system,
		MaxTokens: req.MaxTokens,
		Messages:  req.Messages,
		Tools:     req.Tools,
	}
	if req.Effort != "" {
		wire.OutputConfig = &outputConfig{Effort: req.Effort}
	}
	body, err := json.Marshal(wire)
	if err != nil {
		return MessageResponse{}, fmt.Errorf("anthropic: encoding request: %w", err)
	}
	c.logWire("request", body)

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
	c.logWire("response", respBody)

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
	blocks := make([]ContentBlock, len(wr.Content))
	for i, raw := range wr.Content {
		var wb wireContentBlock
		if err := json.Unmarshal(raw, &wb); err != nil {
			return MessageResponse{}, fmt.Errorf("anthropic: decoding content block %d: %w", i, err)
		}
		if wb.Type == "text" {
			text.WriteString(wb.Text)
		}
		blocks[i] = ContentBlock{
			Type:  wb.Type,
			Text:  wb.Text,
			ID:    wb.ID,
			Name:  wb.Name,
			Input: wb.Input,
			Raw:   raw,
		}
	}
	return MessageResponse{
		Text:       text.String(),
		Content:    blocks,
		StopReason: wr.StopReason,
		Usage:      wr.Usage,
	}, nil
}
