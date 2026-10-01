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

// fakeHandshakeServer mimics the OBSERVED OpenClaw loopback MCP server
// handshake (sylphie/memory/design-prefrontal-redesign.md §F3c LIVE
// HANDSHAKE DUMP): both the bearer token and the capture-key header are
// required (401 otherwise, mirroring the live bearer-alone rejection);
// initialize answers with the exact observed result shape and no
// Mcp-Session-Id header; notifications/initialized answers 202 empty; and
// tools/list answers with a small typed catalog using names observed
// without the mcp__openclaw__ prefix.
func fakeHandshakeServer(t *testing.T, wantToken string, wantCaptureKey string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+wantToken || r.Header.Get("x-openclaw-cli-capture-key") != wantCaptureKey {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error":"unauthorized"}`))
			return
		}

		var req struct {
			Method string `json:"method"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("fake server: decoding request body: %v", err)
		}

		switch req.Method {
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
			t.Fatalf("fake server: unexpected method %q", req.Method)
		}
	}))
}

func TestClient_Handshake_success(t *testing.T) {
	server := fakeHandshakeServer(t, "test-token", "test-capture-key")
	defer server.Close()

	client := NewClient(ServerConfig{
		URL: server.URL,
		Headers: map[string]string{
			"Authorization":              "Bearer test-token",
			"x-openclaw-cli-capture-key": "test-capture-key",
		},
	}, nil)

	result, err := client.Handshake(context.Background())
	if err != nil {
		t.Fatal(err)
	}
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
}

func TestClient_Handshake_missingCaptureKey(t *testing.T) {
	server := fakeHandshakeServer(t, "test-token", "test-capture-key")
	defer server.Close()

	// Authorization alone, no x-openclaw-cli-capture-key — mirrors the live
	// 401-on-bearer-alone behavior from the LIVE HANDSHAKE DUMP.
	client := NewClient(ServerConfig{
		URL: server.URL,
		Headers: map[string]string{
			"Authorization": "Bearer test-token",
		},
	}, nil)

	_, err := client.Handshake(context.Background())
	if err == nil {
		t.Fatal("want an error, got nil")
	}
	if !strings.Contains(err.Error(), "401") {
		t.Errorf("error = %v, want it to mention http 401", err)
	}
	if !strings.Contains(err.Error(), "initialize") {
		t.Errorf("error = %v, want it to name the failing step (initialize)", err)
	}
}

func TestClient_Handshake_echoesSessionID(t *testing.T) {
	var gotSessionIDOnToolsList string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Method string `json:"method"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("fake server: decoding request body: %v", err)
		}
		switch req.Method {
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
	}))
	defer server.Close()

	// This server behavior (emitting Mcp-Session-Id) is NOT what the live
	// gateway does today (it is stateless, see §F3c) — this test only
	// covers the cheap forward-compat echo path the design doc calls for.
	client := NewClient(ServerConfig{URL: server.URL}, nil)
	if _, err := client.Handshake(context.Background()); err != nil {
		t.Fatal(err)
	}
	if gotSessionIDOnToolsList != "sess-123" {
		t.Errorf("tools/list Mcp-Session-Id header = %q, want sess-123", gotSessionIDOnToolsList)
	}
}

func TestClient_Handshake_rejectsEventStreamResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("data: {}\n\n"))
	}))
	defer server.Close()

	client := NewClient(ServerConfig{URL: server.URL}, nil)
	_, err := client.Handshake(context.Background())
	if err == nil || !strings.Contains(err.Error(), "unsupported response Content-Type") {
		t.Fatalf("err = %v, want an unsupported Content-Type error (slice-1 caveat)", err)
	}
}

func TestClient_Handshake_rejectsNonEmptyNotificationBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Method string `json:"method"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("fake server: decoding request body: %v", err)
		}
		switch req.Method {
		case "initialize":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26","capabilities":{},"serverInfo":{"name":"openclaw","version":"0.1.0"}}}`))
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
			w.Write([]byte(`{"unexpected":"body"}`))
		}
	}))
	defer server.Close()

	client := NewClient(ServerConfig{URL: server.URL}, nil)
	_, err := client.Handshake(context.Background())
	if err == nil || !strings.Contains(err.Error(), "expected an empty body") {
		t.Fatalf("err = %v, want an empty-body error", err)
	}
}

// TestNewClientFromConfigFile_expandsEnvAndHandshakes exercises the full
// slice-1 path end to end: write a --mcp-config file with literal ${VAR}
// placeholders (the exact shape the gateway writes, per §F3c), set the env
// those placeholders reference, and confirm the resolved client completes a
// real handshake against the fake server.
func TestNewClientFromConfigFile_expandsEnvAndHandshakes(t *testing.T) {
	server := fakeHandshakeServer(t, "live-token", "live-capture-key")
	defer server.Close()

	t.Setenv("OPENCLAW_MCP_TOKEN", "live-token")
	t.Setenv("OPENCLAW_MCP_CLI_CAPTURE_KEY", "live-capture-key")

	configJSON := `{"mcpServers":{"openclaw":{"type":"http","url":"` + server.URL + `","alwaysLoad":true,"headers":{"Authorization":"Bearer ${OPENCLAW_MCP_TOKEN}","x-openclaw-cli-capture-key":"${OPENCLAW_MCP_CLI_CAPTURE_KEY}"}}}}`
	path := filepath.Join(t.TempDir(), "mcp-config.json")
	if err := os.WriteFile(path, []byte(configJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	client, err := NewClientFromConfigFile(path, nil)
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

	_, err := NewClientFromConfigFile(path, nil)
	if err == nil || !strings.Contains(err.Error(), "decoding config") {
		t.Fatalf("err = %v, want a decoding config error", err)
	}
}
