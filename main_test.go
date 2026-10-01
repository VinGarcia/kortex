package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMcpConfigPathFromArgv(t *testing.T) {
	tests := []struct {
		desc string
		argv []string
		want string
	}{
		{
			desc: "space-separated form, as observed live",
			argv: []string{"--strict-mcp-config", "--mcp-config", "/tmp/mcp-config.json"},
			want: "/tmp/mcp-config.json",
		},
		{
			desc: "equals-joined form",
			argv: []string{"--mcp-config=/tmp/mcp-config.json"},
			want: "/tmp/mcp-config.json",
		},
		{
			desc: "flag absent",
			argv: []string{"--strict-mcp-config", "-p"},
			want: "",
		},
		{
			desc: "flag is the last argv element with no value",
			argv: []string{"--mcp-config"},
			want: "",
		},
		{
			desc: "empty argv",
			argv: nil,
			want: "",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			got := mcpConfigPathFromArgv(test.argv)
			if got != test.want {
				t.Errorf("mcpConfigPathFromArgv(%v) = %q, want %q", test.argv, got, test.want)
			}
		})
	}
}

func TestBuildMCPClient(t *testing.T) {
	validConfig := `{"mcpServers":{"openclaw":{"type":"http","url":"http://127.0.0.1:37439/mcp","headers":{"Authorization":"Bearer ${OPENCLAW_MCP_TOKEN}"}}}}`

	tests := []struct {
		desc        string
		argvFromDir func(dir string) []string
		configFile  string // written as mcp-config.json in dir when non-empty
		env         map[string]string
		wantNilCli  bool
		wantErr     string
	}{
		{
			desc:        "no --mcp-config in argv: nil client, nil error",
			argvFromDir: func(dir string) []string { return []string{"-p", "hello"} },
			wantNilCli:  true,
		},
		{
			desc: "--mcp-config present and valid: client built",
			argvFromDir: func(dir string) []string {
				return []string{"--strict-mcp-config", "--mcp-config", filepath.Join(dir, "mcp-config.json")}
			},
			configFile: validConfig,
			wantNilCli: false,
		},
		{
			desc: "--mcp-config present but file missing: error, nil client",
			argvFromDir: func(dir string) []string {
				return []string{"--mcp-config", filepath.Join(dir, "does-not-exist.json")}
			},
			wantErr: "mcp client:",
		},
		{
			desc: "--mcp-config present but malformed: error, nil client",
			argvFromDir: func(dir string) []string {
				return []string{"--mcp-config", filepath.Join(dir, "mcp-config.json")}
			},
			configFile: `{not json`,
			wantErr:    "mcp client:",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			dir := t.TempDir()
			if test.configFile != "" {
				if err := os.WriteFile(filepath.Join(dir, "mcp-config.json"), []byte(test.configFile), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			getenv := func(name string) string { return test.env[name] }

			client, err := buildMCPClient(test.argvFromDir(dir), getenv)

			if test.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantErr) {
					t.Fatalf("err = %v, want it to contain %q", err, test.wantErr)
				}
				if client != nil {
					t.Errorf("client = %v, want nil on error", client)
				}
				return
			}
			if err != nil {
				t.Fatalf("err = %v, want nil", err)
			}
			if test.wantNilCli && client != nil {
				t.Errorf("client = %v, want nil", client)
			}
			if !test.wantNilCli && client == nil {
				t.Error("client = nil, want non-nil")
			}
		})
	}
}
