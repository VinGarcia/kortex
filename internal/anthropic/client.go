// Package anthropic is a minimal client for the Anthropic Messages API
// authenticated with an OAuth Bearer token (Claude subscription setup-token
// flow). It implements exactly what the facets need today: one
// non-streaming messages call with a system prompt.
package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/vingarcia/krest"
)

const (
	defaultBaseURL   = "https://api.anthropic.com"
	messagesPath     = "/v1/messages"
	anthropicVersion = "2023-06-01"
	// oauthBeta is the beta header value that makes the API accept
	// subscription OAuth Bearer tokens instead of x-api-key.
	oauthBeta = "oauth-2025-04-20"

	// maxResponseBytes caps a success body so a misbehaving endpoint cannot
	// balloon memory; real responses are a few KB to a few hundred KB of JSON.
	maxResponseBytes = 4 << 20

	// The retry budget is small on purpose: it rides out a transient 529
	// overload or a dropped connection, while a persistent outage still fails
	// within a few seconds instead of stalling the turn.
	defaultMaxAttempts    = 3
	defaultRetryBaseDelay = 1 * time.Second
	defaultRetryMaxDelay  = 8 * time.Second
)

// Client calls the Messages API. Construct with NewClient.
type Client struct {
	baseURL string
	token   string
	rest    *krest.Client
	// maxAttempts counts the first try; the delays bound krest's exponential
	// backoff between attempts. Fixed by NewClient; tests shrink the delays.
	maxAttempts    int
	retryBaseDelay time.Duration
	retryMaxDelay  time.Duration
	// wireLog, when non-nil, receives a line-oriented copy of every request and
	// response body for debugging (see SetWireLog). wireMu serializes those
	// writes so concurrent callers never interleave a line.
	wireLog io.Writer
	wireMu  sync.Mutex
	// watchdog, when non-nil, observes each successful response for prompt-cache
	// degradation (see SetCacheWatchdog / CacheWatchdog). It is an observer only:
	// it never blocks, mutates, or errors the request/response flow.
	watchdog *CacheWatchdog
}

// NewClient builds a client holding the OAuth token. baseURL "" means the
// real Anthropic API; tests point it at a local server. timeout bounds each
// HTTP attempt (a retried call may take up to maxAttempts of them plus the
// backoff; callers bound the whole call via context); 0 means no client-level
// timeout.
func NewClient(token string, baseURL string, timeout time.Duration) *Client {
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Client{
		baseURL:        strings.TrimSuffix(baseURL, "/"),
		token:          token,
		rest:           krest.New(timeout),
		maxAttempts:    defaultMaxAttempts,
		retryBaseDelay: defaultRetryBaseDelay,
		retryMaxDelay:  defaultRetryMaxDelay,
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

// SetCacheWatchdog wires the prompt-cache watchdog that inspects each
// successful response for silent cache degradation (see CacheWatchdog). Wire it
// only on the CORE client: it is scoped per-client, so enabling it here (and
// not on the superego/evaluator clients) confines it to the core message path
// the F4 spec targets. A nil watchdog (the default) disables observation. Like
// SetWireLog it is meant to be set once at construction, before any request.
func (c *Client) SetCacheWatchdog(w *CacheWatchdog) {
	c.watchdog = w
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
	// One Write per record so a size-rotating writer never splits a record
	// across the current file and its backup.
	line := fmt.Appendf(nil, "%s %s ", time.Now().Format(time.RFC3339Nano), direction)
	line = append(line, body...)
	line = append(line, '\n')
	c.wireLog.Write(line)
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
	// CacheControl marks this block as a prompt-cache breakpoint when set (see
	// markMessagesCachePrefix); nil omits the field, leaving an ordinary
	// tool_result block unaffected.
	CacheControl *cacheControl `json:"cache_control,omitempty"`
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
	// Messages is the canonical conversation history. CreateMessage places one
	// ephemeral prompt-cache breakpoint on its last eligible content block (see
	// markMessagesCachePrefix) before EphemeralTail is appended, so a caller
	// that must send volatile per-turn content (a tone digest, a critique, an
	// ephemeral draft-under-review message) puts it in EphemeralTail instead of
	// here — appending it to Messages directly would move the breakpoint onto
	// content that is never read back, turning every call into a cache write
	// with no matching read.
	Messages []Message
	// EphemeralTail, when non-empty, is appended to Messages AFTER the
	// cache-prefix breakpoint is placed, so it is sent to the model but never
	// marked and never mutates the caller's Messages slice. Nil/empty omits the
	// field, leaving the wire request's messages equal to (marked) Messages —
	// the exact pre-existing behavior for every caller that does not set it.
	EphemeralTail []Message
	Tools         []ToolDef
	// Effort selects output_config.effort ("low".."max"), which bounds how deeply
	// a thinking model reasons before it answers. "" omits output_config so the
	// API applies its default ("high"); it is also omitted for a model whose
	// traits say it rejects the knob (see modelTraits.SupportsEffort). It matters for the canary core model
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
	if estimatedSystemTokens(system) < minCacheableTokens(model) {
		return
	}
	system[len(system)-1].CacheControl = &cacheControl{Type: "ephemeral"}
}

// estimatedSystemTokens approximates the system prefix size in tokens using the
// standard ~4-bytes-per-token heuristic (token counting is not available
// client-side). It is the single source of the size estimate shared by
// markStableSystemPrefix (deciding whether to mark a breakpoint) and the cache
// watchdog (deciding whether a prefix was large enough that caching was
// expected), so the two never drift apart.
func estimatedSystemTokens(system []systemBlock) int {
	totalChars := 0
	for _, block := range system {
		totalChars += len(block.Text)
	}
	return totalChars / 4
}

// systemHasBreakpoint reports whether the request's system array went out with
// an ephemeral cache breakpoint — i.e. the last block carries cache_control.
// markStableSystemPrefix only ever marks the last block, so this reads the
// exact wire truth the watchdog needs.
func systemHasBreakpoint(system []systemBlock) bool {
	return len(system) > 0 && system[len(system)-1].CacheControl != nil
}

// CreateMessage performs one non-streaming Messages call. Transport
// failures come back wrapped; API failures come back as *APIError.
func (c *Client) CreateMessage(ctx context.Context, req MessageRequest) (MessageResponse, error) {
	system := []systemBlock{{Type: "text", Text: claudeCodePreamble}}
	if req.System != "" {
		system = append(system, systemBlock{Type: "text", Text: req.System})
	}
	markStableSystemPrefix(req.Model, system)

	messages := markMessagesCachePrefix(req.Model, req.Messages)
	if len(req.EphemeralTail) > 0 {
		// The breakpoint is already placed (above); appending the tail now can
		// never move it onto this volatile content. combined is a fresh slice —
		// req.Messages/messages is never mutated, matching markMessagesCachePrefix's
		// own no-mutation contract.
		combined := make([]Message, 0, len(messages)+len(req.EphemeralTail))
		combined = append(combined, messages...)
		combined = append(combined, req.EphemeralTail...)
		messages = combined
	}

	wire := wireRequest{
		Model:     req.Model,
		System:    system,
		MaxTokens: req.MaxTokens,
		Messages:  messages,
		Tools:     req.Tools,
	}
	if req.Effort != "" && traitsFor(req.Model).SupportsEffort {
		wire.OutputConfig = &outputConfig{Effort: req.Effort}
	}
	body, err := json.Marshal(wire)
	if err != nil {
		return MessageResponse{}, fmt.Errorf("anthropic: encoding request: %w", err)
	}
	c.logWire("request", body)

	// retried collects the responses of attempts krest discarded to retry: it
	// leaves their bodies open, so close them here to release the connections.
	var retried []*http.Response
	defer func() {
		for _, r := range retried {
			r.Body.Close()
		}
	}()
	attempts := 0
	resp, err := c.rest.Post(ctx, c.baseURL+messagesPath, krest.RequestData{
		// Pre-marshaled bytes (not the struct) so the wire log above shows the
		// exact payload, and so krest can replay it on every attempt.
		Body: body,
		Headers: map[string]any{
			"Content-Type":      "application/json",
			"Authorization":     "Bearer " + c.token,
			"anthropic-version": anthropicVersion,
			"anthropic-beta":    oauthBeta,
		},
		MaxRetries:     c.maxAttempts,
		BaseRetryDelay: c.retryBaseDelay,
		MaxRetryDelay:  c.retryMaxDelay,
		RetryRule: func(attempt *http.Response, err error) bool {
			// Declining the final attempt changes nothing for krest (it stops
			// there anyway) but keeps that response out of retried and the log.
			attempts++
			if attempts >= c.maxAttempts || !isRetryable(ctx, attempt, err) {
				return false
			}
			if attempt != nil {
				retried = append(retried, attempt)
				c.logWire("retrying", fmt.Appendf(nil, "after http %d", attempt.StatusCode))
			} else {
				c.logWire("retrying", fmt.Appendf(nil, "after %v", err))
			}
			return true
		},
		// Stream hands the success body over unread so the size cap below
		// applies; krest still buffers non-2xx bodies into resp.Body.
		Stream: true,
	})
	if resp.StatusCode == 0 {
		if ctx.Err() != nil && !errors.Is(err, ctx.Err()) {
			// A transport failure was followed by a cancel during the backoff
			// wait: surface the cancellation, as the non-2xx branch below does.
			return MessageResponse{}, fmt.Errorf("anthropic: request failed: %w; last attempt: %w", ctx.Err(), err)
		}
		return MessageResponse{}, fmt.Errorf("anthropic: request failed: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		c.logWire("response", resp.Body)
		apiErr := &APIError{Status: resp.StatusCode}
		var we wireError
		if json.Unmarshal(resp.Body, &we) == nil {
			apiErr.Type = we.Error.Type
			apiErr.Message = we.Error.Message
		}
		if ctx.Err() != nil {
			// The context ended during a backoff wait: report the cancellation
			// first, keeping the last API answer reachable via errors.As.
			return MessageResponse{}, fmt.Errorf("anthropic: request failed: %w; last answer: %w", ctx.Err(), apiErr)
		}
		return MessageResponse{}, apiErr
	}
	defer resp.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp, maxResponseBytes))
	if err != nil {
		return MessageResponse{}, fmt.Errorf("anthropic: reading response: %w", err)
	}
	c.logWire("response", respBody)

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
	c.observeCache(req.Model, system, wr.Usage)
	return MessageResponse{
		Text:       text.String(),
		Content:    blocks,
		StopReason: wr.StopReason,
		Usage:      wr.Usage,
	}, nil
}

// isRetryable is the RetryRule for one attempt. It retries transport failures,
// 5xx (529 overloaded included) and genuine rate limits, and nothing once ctx
// is done. A 429 counts as genuine only when it carries rate-limit headers: the
// OAuth policy rejection (see claudeCodePreamble) is a bare 429 with none, and
// retrying it would just repeat the same rejection.
func isRetryable(ctx context.Context, resp *http.Response, err error) bool {
	if ctx.Err() != nil {
		return false
	}
	if err != nil {
		return true
	}
	if resp.StatusCode == http.StatusTooManyRequests {
		return hasRateLimitHeaders(resp.Header)
	}
	return resp.StatusCode >= 500
}

// hasRateLimitHeaders reports whether h carries the headers Anthropic attaches
// to a real rate-limit answer: retry-after or any anthropic-ratelimit-* field.
func hasRateLimitHeaders(h http.Header) bool {
	if h.Get("Retry-After") != "" {
		return true
	}
	for name := range h {
		if strings.HasPrefix(strings.ToLower(name), "anthropic-ratelimit-") {
			return true
		}
	}
	return false
}

// observeCache runs the cache watchdog over one successful response. It is
// fail-safe by contract (F4 requirement: the watchdog must never break the
// request flow): a nil watchdog is a no-op, and any panic inside the watchdog
// is recovered so the response the caller already holds is always returned
// intact. The recovered panic is reported (not swallowed) — a fail-loud monitor
// must surface its own bug rather than hide it. system is the exact block array
// sent on the wire, so the watchdog can read whether a breakpoint went out.
func (c *Client) observeCache(model string, system []systemBlock, usage Usage) {
	if c.watchdog == nil {
		return
	}
	defer func() {
		if r := recover(); r != nil {
			c.watchdog.reportDefect(r)
		}
	}()
	c.watchdog.observe(model, system, usage)
}
