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
func authHandshakeHandler(wantToken string, wantCaptureKey string) func(w http.ResponseWriter, r *http.Request) {
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
			serverHandler: authHandshakeHandler("test-token", "test-capture-key"),
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
			serverHandler:   authHandshakeHandler("test-token", "test-capture-key"),
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
	server := httptest.NewServer(http.HandlerFunc(authHandshakeHandler("live-token", "live-capture-key")))
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
