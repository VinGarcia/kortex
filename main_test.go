package main

import (
	"bytes"
	"os"
	"path/filepath"
	"regexp"
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

func TestResolveSuperegoModel(t *testing.T) {
	tests := []struct {
		desc       string
		configured string
		want       string
	}{
		{
			desc:       "empty model falls back to the compiled per-facet default",
			configured: "",
			want:       "claude-opus-4-8",
		},
		{
			desc:       "explicit model overrides the default",
			configured: "claude-sonnet-5",
			want:       "claude-sonnet-5",
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			got := resolveSuperegoModel(test.configured)
			if got != test.want {
				t.Errorf("resolveSuperegoModel(%q) = %q, want %q", test.configured, got, test.want)
			}
		})
	}
}

func TestStreamEmitterEnabled(t *testing.T) {
	tests := []struct {
		gate string
		want bool
	}{
		{gate: "", want: false},
		{gate: "0", want: false},
		{gate: "off", want: false},
		{gate: "no", want: false},
		{gate: "anything-else", want: false},
		{gate: "1", want: true},
		{gate: "true", want: true},
		{gate: "TRUE", want: true},
		{gate: "yes", want: true},
		{gate: " on ", want: true},
	}
	for _, test := range tests {
		t.Run(test.gate, func(t *testing.T) {
			if got := streamEmitterEnabled(test.gate); got != test.want {
				t.Errorf("streamEmitterEnabled(%q) = %v, want %v", test.gate, got, test.want)
			}
		})
	}
}

func TestBuildStreamEmitter(t *testing.T) {
	newUUID := func() string { return "uuid-fixed" }

	if got := buildStreamEmitter("", &bytes.Buffer{}, newUUID); got != nil {
		t.Errorf("gate off: emitter = %v, want nil (dormant by default)", got)
	}
	if got := buildStreamEmitter("1", &bytes.Buffer{}, newUUID); got == nil {
		t.Error("gate on: emitter = nil, want constructed")
	}
}

func TestNewUUIDv4(t *testing.T) {
	v4 := regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)
	first := newUUIDv4()
	if !v4.MatchString(first) {
		t.Fatalf("newUUIDv4() = %q, not a v4 uuid", first)
	}
	if second := newUUIDv4(); second == first {
		t.Errorf("newUUIDv4() returned the same id twice: %q", first)
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
