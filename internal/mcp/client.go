package mcp

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

// protocolVersion is the MCP protocol version kortex requests in
// initialize. The live server accepts ["2025-03-26","2024-11-05"]
// (newest-first per §F3c) and echoed this exact value back unchanged when
// the spike requested it; this client does not implement falling back to an
// older version on a mismatch (not exercised by the observed handshake).
const protocolVersion = "2025-03-26"

const (
	clientName    = "kortex"
	clientVersion = "0"
)

// maxResponseBytes caps how much of a response body this client reads. The
// cap keeps a misbehaving endpoint from ballooning memory; real
// initialize/tools-list responses are a few KB of JSON.
const maxResponseBytes = 4 << 20

// defaultTimeout bounds an HTTP call when NewClient is not given an
// *http.Client. The loopback server is on localhost and answers in
// milliseconds in practice; this is a safety bound, not a tuned budget.
const defaultTimeout = 30 * time.Second

// Client is an MCP Streamable-HTTP client for one server endpoint (the
// OpenClaw loopback server). Construct with NewClient or
// NewClientFromConfigFile.
type Client struct {
	url        string
	headers    map[string]string
	httpClient *http.Client

	// sessionID is the Mcp-Session-Id the server returned on a previous
	// response, if any, echoed on subsequent requests. The live server
	// observed in §F3c does not emit one (stateless mode) — this client
	// still captures and echoes it if a future/different deployment starts
	// sending it, at near-zero cost.
	sessionID string
}

// NewClient builds a Client for server. httpClient nil means a new
// *http.Client with defaultTimeout; tests and callers that need their own
// transport/timeout pass one in.
func NewClient(server ServerConfig, httpClient *http.Client) *Client {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: defaultTimeout}
	}
	return &Client{
		url:        server.URL,
		headers:    server.Headers,
		httpClient: httpClient,
	}
}

// NewClientFromConfigFile loads the "openclaw" server entry from the
// --mcp-config file at path (expanding ${VAR} header placeholders via
// getenv, see LoadServerConfig) and builds a Client for it. getenv is
// supplied by the caller — this package never reads the process
// environment itself (main.go passes os.Getenv; tests pass a fake lookup).
func NewClientFromConfigFile(path string, getenv func(string) string, httpClient *http.Client) (*Client, error) {
	server, err := LoadServerConfig(path, getenv)
	if err != nil {
		return nil, err
	}
	return NewClient(server, httpClient), nil
}

// ServerInfo identifies the MCP server, as reported in initialize's result.
type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// initializeResult is the decoded result of the initialize call.
type initializeResult struct {
	ProtocolVersion string          `json:"protocolVersion"`
	Capabilities    json.RawMessage `json:"capabilities"`
	ServerInfo      ServerInfo      `json:"serverInfo"`
}

// Tool is one entry of the tools/list catalog. Name arrives from the server
// WITHOUT the "mcp__openclaw__" prefix (§F3c observed); prefixing for
// exposure to the model is a later slice's concern, not this transport's.
// InputSchema stays raw JSON Schema — this package does not interpret it.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
}

// HandshakeResult is the outcome of a completed Handshake call.
type HandshakeResult struct {
	ServerInfo ServerInfo
	Tools      []Tool
}

// Handshake performs the slice-1 sequence against the server: initialize,
// notifications/initialized, tools/list. It returns the server identity and
// the typed tool catalog. tools/call is a later slice (see package doc).
func (c *Client) Handshake(ctx context.Context) (HandshakeResult, error) {
	init, err := c.initialize(ctx)
	if err != nil {
		return HandshakeResult{}, fmt.Errorf("mcp: initialize: %w", err)
	}
	if err := c.notifyInitialized(ctx); err != nil {
		return HandshakeResult{}, fmt.Errorf("mcp: notifications/initialized: %w", err)
	}
	tools, err := c.toolsList(ctx)
	if err != nil {
		return HandshakeResult{}, fmt.Errorf("mcp: tools/list: %w", err)
	}
	return HandshakeResult{ServerInfo: init.ServerInfo, Tools: tools}, nil
}

// initialize sends the id=1 "initialize" request and parses its result.
func (c *Client) initialize(ctx context.Context) (initializeResult, error) {
	params := map[string]any{
		"protocolVersion": protocolVersion,
		"capabilities":    map[string]any{},
		"clientInfo": map[string]any{
			"name":    clientName,
			"version": clientVersion,
		},
	}
	resp, err := c.post(ctx, rpcRequest{JSONRPC: "2.0", ID: 1, Method: "initialize", Params: params})
	if err != nil {
		return initializeResult{}, err
	}
	defer resp.Body.Close()

	if sid := resp.Header.Get("Mcp-Session-Id"); sid != "" {
		c.sessionID = sid
	}

	body, rpcResp, err := decodeRPCResponse(resp)
	if err != nil {
		return initializeResult{}, err
	}
	if rpcResp.Error != nil {
		return initializeResult{}, rpcResp.Error
	}

	var result initializeResult
	if err := json.Unmarshal(rpcResp.Result, &result); err != nil {
		return initializeResult{}, fmt.Errorf("decoding result: %w (body=%s)", err, body)
	}
	return result, nil
}

// notifyInitialized sends the id-less "notifications/initialized"
// notification. A notification gets no JSON-RPC response; the observed
// server answers HTTP 202 with an empty body (§F3c LIVE HANDSHAKE DUMP).
// Any 2xx with an empty (or whitespace-only) body counts as success.
func (c *Client) notifyInitialized(ctx context.Context) error {
	resp, err := c.post(ctx, rpcRequest{JSONRPC: "2.0", Method: "notifications/initialized"})
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return fmt.Errorf("reading response body: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("http %d: %s", resp.StatusCode, body)
	}
	if len(bytes.TrimSpace(body)) != 0 {
		return fmt.Errorf("expected an empty body, got %q", body)
	}
	return nil
}

// toolsList sends the id=2 "tools/list" request and parses its result.
func (c *Client) toolsList(ctx context.Context) ([]Tool, error) {
	resp, err := c.post(ctx, rpcRequest{JSONRPC: "2.0", ID: 2, Method: "tools/list"})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, rpcResp, err := decodeRPCResponse(resp)
	if err != nil {
		return nil, err
	}
	if rpcResp.Error != nil {
		return nil, rpcResp.Error
	}

	var result struct {
		Tools []Tool `json:"tools"`
	}
	if err := json.Unmarshal(rpcResp.Result, &result); err != nil {
		return nil, fmt.Errorf("decoding result: %w (body=%s)", err, body)
	}
	return result.Tools, nil
}

// rpcRequest is one JSON-RPC 2.0 request or notification. ID is omitted (via
// omitempty on the any-typed zero value, nil) for a notification, matching
// the wire distinction the server uses to decide whether to answer at all.
type rpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      any    `json:"id,omitempty"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

// rpcResponse is one JSON-RPC 2.0 response envelope.
type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id"`
	Result  json.RawMessage `json:"result"`
	Error   *rpcError       `json:"error"`
}

// rpcError is a JSON-RPC 2.0 error object; it implements error so callers
// can propagate it directly.
type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (e *rpcError) Error() string {
	return fmt.Sprintf("jsonrpc error %d: %s", e.Code, e.Message)
}

// post marshals req and POSTs it to the server with the resolved headers
// plus the fixed Content-Type/Accept the MCP Streamable-HTTP transport
// requires. The caller owns closing the returned response's body.
func (c *Client) post(ctx context.Context, req rpcRequest) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("encoding request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("building request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json, text/event-stream")
	for name, value := range c.headers {
		httpReq.Header.Set(name, value)
	}
	if c.sessionID != "" {
		httpReq.Header.Set("Mcp-Session-Id", c.sessionID)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	return resp, nil
}

// decodeRPCResponse reads a JSON-RPC response body and decodes the
// envelope. It requires status 2xx and Content-Type: application/json; see
// the package doc caveat on why text/event-stream is rejected rather than
// parsed in this slice.
func decodeRPCResponse(resp *http.Response) ([]byte, rpcResponse, error) {
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return nil, rpcResponse{}, fmt.Errorf("reading response body: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return body, rpcResponse{}, fmt.Errorf("http %d: %s", resp.StatusCode, body)
	}

	contentType := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(contentType, "application/json") {
		return body, rpcResponse{}, fmt.Errorf("unsupported response Content-Type %q (slice-1 only decodes application/json, see package doc)", contentType)
	}

	var rr rpcResponse
	if err := json.Unmarshal(body, &rr); err != nil {
		return body, rpcResponse{}, fmt.Errorf("decoding json-rpc envelope: %w (body=%s)", err, body)
	}
	return body, rr, nil
}
