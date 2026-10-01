package mcp

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadServerConfig(t *testing.T) {
	tests := []struct {
		desc    string
		content string
		env     map[string]string
		wantErr string
		check   func(t *testing.T, server ServerConfig)
	}{
		{
			desc:    "expands ${VAR} placeholders from env",
			content: `{"mcpServers":{"openclaw":{"type":"http","url":"http://127.0.0.1:37439/mcp","alwaysLoad":true,"headers":{"Authorization":"Bearer ${OPENCLAW_MCP_TOKEN}","x-openclaw-cli-capture-key":"${OPENCLAW_MCP_CLI_CAPTURE_KEY}"}}}}`,
			env: map[string]string{
				"OPENCLAW_MCP_TOKEN":           "tok-123",
				"OPENCLAW_MCP_CLI_CAPTURE_KEY": "cap-456",
			},
			check: func(t *testing.T, server ServerConfig) {
				if server.URL != "http://127.0.0.1:37439/mcp" {
					t.Errorf("url = %q", server.URL)
				}
				if server.Headers["Authorization"] != "Bearer tok-123" {
					t.Errorf("Authorization = %q", server.Headers["Authorization"])
				}
				if server.Headers["x-openclaw-cli-capture-key"] != "cap-456" {
					t.Errorf("capture-key header = %q", server.Headers["x-openclaw-cli-capture-key"])
				}
			},
		},
		{
			desc:    "unset env var expands to empty, not an error",
			content: `{"mcpServers":{"openclaw":{"type":"http","url":"http://x/mcp","headers":{"Authorization":"Bearer ${SOME_UNSET_VAR_XYZ}"}}}}`,
			check: func(t *testing.T, server ServerConfig) {
				if server.Headers["Authorization"] != "Bearer " {
					t.Errorf("Authorization = %q, want empty expansion", server.Headers["Authorization"])
				}
			},
		},
		{
			desc:    "missing openclaw entry is an error",
			content: `{"mcpServers":{"other":{"type":"http","url":"http://x/mcp"}}}`,
			wantErr: "no mcpServers.openclaw entry",
		},
		{
			desc:    "missing url is an error",
			content: `{"mcpServers":{"openclaw":{"type":"http"}}}`,
			wantErr: "has no url",
		},
		{
			desc:    "malformed json is an error",
			content: `{not json`,
			wantErr: "decoding config",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			for k, v := range test.env {
				t.Setenv(k, v)
			}
			path := filepath.Join(t.TempDir(), "mcp-config.json")
			if err := os.WriteFile(path, []byte(test.content), 0o644); err != nil {
				t.Fatal(err)
			}

			server, err := LoadServerConfig(path)
			if test.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantErr) {
					t.Fatalf("err = %v, want it to contain %q", err, test.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			test.check(t, server)
		})
	}
}

func TestLoadServerConfig_missingFile(t *testing.T) {
	_, err := LoadServerConfig("/nonexistent/path/mcp-config.json")
	if err == nil || !strings.Contains(err.Error(), "reading config") {
		t.Fatalf("err = %v, want a reading config error", err)
	}
}
