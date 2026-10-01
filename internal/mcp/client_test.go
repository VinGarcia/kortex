package mcp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// decodeRequestMethod reads the "method" field of a json-rpc request body.
// Shared by the table rows below that need to branch per handshake step; a
// malformed body is treated as an empty method name rather than failing the
// test, since these fakes only ever receive well-formed requests from
// Client itself — no row here deliberately sends garbage.
func decodeRequestMethod(r *http.Request) string {
	var req struct {
		Method string `json:"method"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)
	return req.Method
}

// authHandshakeHandler mimics the OBSERVED OpenClaw loopback MCP server
// handshake (sylphie/memory/design-prefrontal-redesign.md §F3c LIVE
// HANDSHAKE DUMP): both the bearer token and the capture-key header are
// required (401 otherwise, mirroring the live bearer-alone rejection);
// initialize answers with the exact observed result shape and no
// Mcp-Session-Id header; notifications/initialized answers 202 empty; and
// tools/list answers with a small typed catalog using names observed
// without the mcp__openclaw__ prefix.
func authHandshakeHandler(t *testing.T, wantToken string, wantCaptureKey string) func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+wantToken || r.Header.Get("x-openclaw-cli-capture-key") != wantCaptureKey {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error":"unauthorized"}`))
			return
		}

		switch decodeRequestMethod(r) {
		case "initialize":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"openclaw","version":"0.1.0"}}}`))
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"jsonrpc":"2.0","id":2,"result":{"tools":[
				{"name":"sessions_send","description":"send a message to a session","inputSchema":{"type":"object","properties":{}}},
				{"name":"memory_search","description":"semantic memory search","inputSchema":{"type":"object","properties":{}}},
				{"name":"ask_user","description":"ask the human user","inputSchema":{"type":"object","properties":{}}}
			]}}`))
		default:
			// Safety net: any RPC outside the handshake sequence (including a
			// method that failed to decode) is a test bug, not a valid case.
			// t.Errorf (not Fatalf) because this runs in the server goroutine,
			// where Fatalf's runtime.Goexit would not stop the test.
			t.Errorf("fake server: unexpected method %q", decodeRequestMethod(r))
			w.WriteHeader(http.StatusInternalServerError)
		}
	}
}

// TestClient_Handshake covers the slice-1 Handshake sequence against a
// table of fake servers. Each row supplies its own serverHandler (so a new
// scenario is a new row, not a new sibling function) and either
// wantErrContains (every listed substring must appear in the returned
// error) or checkResult (run when Handshake succeeds).
func TestClient_Handshake(t *testing.T) {
	// Captured by the "echoes session id" row's serverHandler and read back
	// by its checkResult; scoped to the whole test func but only ever
	// written/read by that one row.
	var gotSessionIDOnToolsList string

	tests := []struct {
		name            string
		headers         map[string]string
		serverHandler   func(w http.ResponseWriter, r *http.Request)
		wantErrContains []string
		checkResult     func(t *testing.T, result HandshakeResult)
	}{
		{
			name: "success",
			headers: map[string]string{
				"Authorization":              "Bearer test-token",
				"x-openclaw-cli-capture-key": "test-capture-key",
			},
			serverHandler: authHandshakeHandler(t, "test-token", "test-capture-key"),
			checkResult: func(t *testing.T, result HandshakeResult) {
				if result.ServerInfo.Name != "openclaw" || result.ServerInfo.Version != "0.1.0" {
					t.Errorf("unexpected server info: %+v", result.ServerInfo)
				}

				wantNames := []string{"sessions_send", "memory_search", "ask_user"}
				if len(result.Tools) != len(wantNames) {
					t.Fatalf("got %d tools, want %d: %+v", len(result.Tools), len(wantNames), result.Tools)
				}
				for i, name := range wantNames {
					if result.Tools[i].Name != name {
						t.Errorf("tool[%d].Name = %q, want %q", i, result.Tools[i].Name, name)
					}
					if strings.HasPrefix(result.Tools[i].Name, "mcp__openclaw__") {
						t.Errorf("tool[%d].Name = %q, should arrive without the mcp__openclaw__ prefix (model-side convention, later slice)", i, result.Tools[i].Name)
					}
					if len(result.Tools[i].InputSchema) == 0 {
						t.Errorf("tool[%d].InputSchema is empty", i)
					}
				}
			},
		},
		{
			// Authorization alone, no x-openclaw-cli-capture-key — mirrors
			// the live 401-on-bearer-alone behavior from the LIVE HANDSHAKE
			// DUMP.
			name: "missing capture key is rejected like the live server",
			headers: map[string]string{
				"Authorization": "Bearer test-token",
			},
			serverHandler:   authHandshakeHandler(t, "test-token", "test-capture-key"),
			wantErrContains: []string{"401", "initialize"},
		},
		{
			// This server behavior (emitting Mcp-Session-Id) is NOT what the
			// live gateway does today (it is stateless, see §F3c) — this row
			// only covers the cheap forward-compat echo path the design doc
			// calls for.
			name: "echoes received Mcp-Session-Id onto subsequent requests",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				switch decodeRequestMethod(r) {
				case "initialize":
					w.Header().Set("Content-Type", "application/json")
					w.Header().Set("Mcp-Session-Id", "sess-123")
					w.Write([]byte(`{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26","capabilities":{},"serverInfo":{"name":"openclaw","version":"0.1.0"}}}`))
				case "notifications/initialized":
					w.WriteHeader(http.StatusAccepted)
				case "tools/list":
					gotSessionIDOnToolsList = r.Header.Get("Mcp-Session-Id")
					w.Header().Set("Content-Type", "application/json")
					w.Write([]byte(`{"jsonrpc":"2.0","id":2,"result":{"tools":[]}}`))
				}
			},
			checkResult: func(t *testing.T, result HandshakeResult) {
				if gotSessionIDOnToolsList != "sess-123" {
					t.Errorf("tools/list Mcp-Session-Id header = %q, want sess-123", gotSessionIDOnToolsList)
				}
			},
		},
		{
			name: "rejects a text/event-stream response (slice-1 caveat)",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.Write([]byte("data: {}\n\n"))
			},
			wantErrContains: []string{"unsupported response Content-Type"},
		},
		{
			name: "rejects a non-empty notifications/initialized body",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				switch decodeRequestMethod(r) {
				case "initialize":
					w.Header().Set("Content-Type", "application/json")
					w.Write([]byte(`{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26","capabilities":{},"serverInfo":{"name":"openclaw","version":"0.1.0"}}}`))
				case "notifications/initialized":
					w.WriteHeader(http.StatusAccepted)
					w.Write([]byte(`{"unexpected":"body"}`))
				}
			},
			wantErrContains: []string{"expected an empty body"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(test.serverHandler))
			defer server.Close()

			client := NewClient(ServerConfig{URL: server.URL, Headers: test.headers}, nil)
			result, err := client.Handshake(context.Background())

			if len(test.wantErrContains) > 0 {
				if err == nil {
					t.Fatal("want an error, got nil")
				}
				for _, want := range test.wantErrContains {
					if !strings.Contains(err.Error(), want) {
						t.Errorf("error = %v, want it to contain %q", err, want)
					}
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if test.checkResult != nil {
				test.checkResult(t, result)
			}
		})
	}
}

// TestNewClientFromConfigFile_expandsEnvAndHandshakes exercises the full
// slice-1 path end to end: write a --mcp-config file with literal ${VAR}
// placeholders (the exact shape the gateway writes, per §F3c), inject a fake
// getenv over those placeholders, and confirm the resolved client completes
// a real handshake against the fake server.
func TestNewClientFromConfigFile_expandsEnvAndHandshakes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(authHandshakeHandler(t, "live-token", "live-capture-key")))
	defer server.Close()

	fakeEnv := map[string]string{
		"OPENCLAW_MCP_TOKEN":           "live-token",
		"OPENCLAW_MCP_CLI_CAPTURE_KEY": "live-capture-key",
	}
	getenv := func(name string) string { return fakeEnv[name] }

	configJSON := `{"mcpServers":{"openclaw":{"type":"http","url":"` + server.URL + `","alwaysLoad":true,"headers":{"Authorization":"Bearer ${OPENCLAW_MCP_TOKEN}","x-openclaw-cli-capture-key":"${OPENCLAW_MCP_CLI_CAPTURE_KEY}"}}}}`
	path := filepath.Join(t.TempDir(), "mcp-config.json")
	if err := os.WriteFile(path, []byte(configJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	client, err := NewClientFromConfigFile(path, getenv, nil)
	if err != nil {
		t.Fatal(err)
	}
	result, err := client.Handshake(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Tools) == 0 {
		t.Error("want at least one tool from the fake server's catalog")
	}
}

// TestClient_CallTool covers the slice-2 tools/call parsing against a table
// of fake servers, mirroring TestClient_Handshake's per-row serverHandler
// pattern.
func TestClient_CallTool(t *testing.T) {
	// Captured by the "builds tools/call params" row's serverHandler and read
	// back by its checkResult; scoped to the whole test func but only ever
	// written/read by that one row (mirrors TestClient_Handshake's
	// gotSessionIDOnToolsList precedent).
	var gotName string
	var gotArguments map[string]any

	tests := []struct {
		name            string
		serverHandler   func(w http.ResponseWriter, r *http.Request)
		wantErrContains []string
		checkResult     func(t *testing.T, result CallToolResult)
	}{
		{
			name: "success with a single text block",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				switch decodeRequestMethod(r) {
				case "tools/call":
					w.Header().Set("Content-Type", "application/json")
					w.Write([]byte(`{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"ok"}],"isError":false}}`))
				default:
					t.Errorf("fake server: unexpected method %q", decodeRequestMethod(r))
					w.WriteHeader(http.StatusInternalServerError)
				}
			},
			checkResult: func(t *testing.T, result CallToolResult) {
				if result.IsError {
					t.Error("IsError = true, want false")
				}
				want := []Content{{Type: "text", Text: "ok"}}
				if len(result.Content) != len(want) || result.Content[0] != want[0] {
					t.Errorf("Content = %+v, want %+v", result.Content, want)
				}
			},
		},
		{
			name: "tool-level failure surfaces as isError, not err",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				switch decodeRequestMethod(r) {
				case "tools/call":
					w.Header().Set("Content-Type", "application/json")
					w.Write([]byte(`{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"boom"}],"isError":true}}`))
				default:
					t.Errorf("fake server: unexpected method %q", decodeRequestMethod(r))
					w.WriteHeader(http.StatusInternalServerError)
				}
			},
			checkResult: func(t *testing.T, result CallToolResult) {
				if !result.IsError {
					t.Error("IsError = false, want true")
				}
				if len(result.Content) != 1 || result.Content[0].Text != "boom" {
					t.Errorf("Content = %+v, want a single %q block", result.Content, "boom")
				}
			},
		},
		{
			name: "success with multiple content blocks",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				switch decodeRequestMethod(r) {
				case "tools/call":
					w.Header().Set("Content-Type", "application/json")
					w.Write([]byte(`{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"first"},{"type":"text","text":"second"}],"isError":false}}`))
				default:
					t.Errorf("fake server: unexpected method %q", decodeRequestMethod(r))
					w.WriteHeader(http.StatusInternalServerError)
				}
			},
			checkResult: func(t *testing.T, result CallToolResult) {
				want := []Content{{Type: "text", Text: "first"}, {Type: "text", Text: "second"}}
				if len(result.Content) != len(want) || result.Content[0] != want[0] || result.Content[1] != want[1] {
					t.Errorf("Content = %+v, want %+v", result.Content, want)
				}
			},
		},
		{
			name: "malformed result is a decode error",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				switch decodeRequestMethod(r) {
				case "tools/call":
					w.Header().Set("Content-Type", "application/json")
					w.Write([]byte(`{"jsonrpc":"2.0","id":3,"result":{"content":"not-an-array"}}`))
				default:
					t.Errorf("fake server: unexpected method %q", decodeRequestMethod(r))
					w.WriteHeader(http.StatusInternalServerError)
				}
			},
			wantErrContains: []string{"decoding result"},
		},
		{
			name: "non-200 is an error",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				switch decodeRequestMethod(r) {
				case "tools/call":
					w.WriteHeader(http.StatusInternalServerError)
					w.Write([]byte(`{"error":"boom"}`))
				default:
					t.Errorf("fake server: unexpected method %q", decodeRequestMethod(r))
					w.WriteHeader(http.StatusInternalServerError)
				}
			},
			wantErrContains: []string{"http 500"},
		},
		{
			name: "rejects a text/event-stream response (same caveat as Handshake)",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.Write([]byte("data: {}\n\n"))
			},
			wantErrContains: []string{"unsupported response Content-Type"},
		},
		{
			// A well-formed JSON-RPC error envelope on HTTP 200 (distinct from a
			// tool-level isError result): CallTool must surface rpcResp.Error as
			// a returned error, not a CallToolResult (client.go:141-143).
			name: "jsonrpc error envelope surfaces as err",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				switch decodeRequestMethod(r) {
				case "tools/call":
					w.Header().Set("Content-Type", "application/json")
					w.Write([]byte(`{"jsonrpc":"2.0","id":3,"error":{"code":-32602,"message":"invalid params"}}`))
				default:
					t.Errorf("fake server: unexpected method %q", decodeRequestMethod(r))
					w.WriteHeader(http.StatusInternalServerError)
				}
			},
			wantErrContains: []string{"jsonrpc error", "-32602", "invalid params"},
		},
		{
			// Round-trip of the request CallTool builds (client.go:127-130): the
			// handler decodes the incoming body and the row's checkResult asserts
			// params.name and params.arguments arrived intact, guarding against a
			// silent param-building regression the method-only rows would miss.
			name: "builds tools/call params with name and arguments",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				var req struct {
					Method string `json:"method"`
					Params struct {
						Name      string         `json:"name"`
						Arguments map[string]any `json:"arguments"`
					} `json:"params"`
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Errorf("fake server: decoding request body: %v", err)
				}
				if req.Method != "tools/call" {
					t.Errorf("fake server: unexpected method %q", req.Method)
					w.WriteHeader(http.StatusInternalServerError)
					return
				}
				gotName = req.Params.Name
				gotArguments = req.Params.Arguments
				w.Header().Set("Content-Type", "application/json")
				w.Write([]byte(`{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"ok"}],"isError":false}}`))
			},
			checkResult: func(t *testing.T, result CallToolResult) {
				if gotName != "some_tool" {
					t.Errorf("tools/call params.name = %q, want %q", gotName, "some_tool")
				}
				if gotArguments["arg"] != "value" {
					t.Errorf("tools/call params.arguments = %+v, want map[arg:value]", gotArguments)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(test.serverHandler))
			defer server.Close()

			client := NewClient(ServerConfig{URL: server.URL}, nil)
			result, err := client.CallTool(context.Background(), "some_tool", map[string]any{"arg": "value"})

			if len(test.wantErrContains) > 0 {
				if err == nil {
					t.Fatal("want an error, got nil")
				}
				for _, want := range test.wantErrContains {
					if !strings.Contains(err.Error(), want) {
						t.Errorf("error = %v, want it to contain %q", err, want)
					}
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if test.checkResult != nil {
				test.checkResult(t, result)
			}
		})
	}
}

func TestNewClientFromConfigFile_malformedConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mcp-config.json")
	if err := os.WriteFile(path, []byte(`{not json`), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := NewClientFromConfigFile(path, func(string) string { return "" }, nil)
	if err == nil || !strings.Contains(err.Error(), "decoding config") {
		t.Fatalf("err = %v, want a decoding config error", err)
	}
}
