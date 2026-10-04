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

func TestFirstArgvValue(t *testing.T) {
	tests := []struct {
		desc string
		argv []string
		flag string
		want string
	}{
		{desc: "space-separated", argv: []string{"--model", "claude-opus-4-8"}, flag: "--model", want: "claude-opus-4-8"},
		{desc: "equals-joined", argv: []string{"--model=claude-sonnet-5"}, flag: "--model", want: "claude-sonnet-5"},
		{desc: "absent", argv: []string{"-p", "hi"}, flag: "--model", want: ""},
		{desc: "flag is last element, no value", argv: []string{"--model"}, flag: "--model", want: ""},
		{desc: "first occurrence wins", argv: []string{"--model", "a", "--model", "b"}, flag: "--model", want: "a"},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			if got := firstArgvValue(test.argv, test.flag); got != test.want {
				t.Errorf("firstArgvValue(%v, %q) = %q, want %q", test.argv, test.flag, got, test.want)
			}
		})
	}
}

func TestNativeSessionID(t *testing.T) {
	const minted = "minted-uuid"
	newUUID := func() string { return minted }
	tests := []struct {
		desc string
		argv []string
		want string
	}{
		{desc: "--session-id for a fresh session", argv: []string{"--session-id", "sess-1"}, want: "sess-1"},
		{desc: "--resume for a continuation", argv: []string{"--resume", "sess-2"}, want: "sess-2"},
		{desc: "--session-id wins over --resume", argv: []string{"--resume", "r", "--session-id", "s"}, want: "s"},
		{desc: "neither present: minted", argv: []string{"-p", "hi"}, want: minted},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			if got := nativeSessionID(test.argv, newUUID); got != test.want {
				t.Errorf("nativeSessionID(%v) = %q, want %q", test.argv, got, test.want)
			}
		})
	}
}

func TestRunNative_guardsFailFast(t *testing.T) {
	// Each guard returns before any Messages call, so empty stdin is enough.
	tests := []struct {
		desc string
		env  map[string]string
		argv []string
	}{
		{
			desc: "missing token",
			env:  map[string]string{},
			argv: []string{"--model", "claude-opus-4-8"},
		},
		{
			desc: "missing model",
			env:  map[string]string{"CODECOMPANION_OAUTH_TOKEN": "secret"},
			argv: []string{"-p", "hi"},
		},
		{
			// A present-but-broken KORTEX_CONFIG must fail fast the same way the
			// passthrough path does: buildFacets errors before any turn runs.
			desc: "broken KORTEX_CONFIG",
			env: map[string]string{
				"CODECOMPANION_OAUTH_TOKEN": "secret",
				"KORTEX_CONFIG":             filepath.Join(t.TempDir(), "does-not-exist.json"),
			},
			argv: []string{"--model", "claude-opus-4-8"},
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			getenv := func(name string) string { return test.env[name] }
			code := runNative(test.argv, getenv, strings.NewReader(""), &bytes.Buffer{}, &bytes.Buffer{})
			if code != 1 {
				t.Errorf("runNative exit code = %d, want 1", code)
			}
		})
	}
}

func TestResolveNativeToken(t *testing.T) {
	tokenFile := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(tokenFile, []byte("  file-token\n"), 0o600); err != nil {
		t.Fatalf("writing token file: %v", err)
	}

	tests := []struct {
		desc    string
		env     map[string]string
		want    string
		wantErr bool
	}{
		{
			desc: "token file trimmed and preferred over env",
			env:  map[string]string{"KORTEX_TOKEN_FILE": tokenFile, "CODECOMPANION_OAUTH_TOKEN": "env-token"},
			want: "file-token",
		},
		{
			desc: "env token when no file set",
			env:  map[string]string{"CODECOMPANION_OAUTH_TOKEN": "env-token"},
			want: "env-token",
		},
		{
			desc:    "unreadable token file is a hard error",
			env:     map[string]string{"KORTEX_TOKEN_FILE": filepath.Join(t.TempDir(), "missing"), "CODECOMPANION_OAUTH_TOKEN": "env-token"},
			wantErr: true,
		},
		{
			desc: "neither set yields empty",
			env:  map[string]string{},
			want: "",
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			getenv := func(name string) string { return test.env[name] }
			got, err := resolveNativeToken(getenv)
			if test.wantErr {
				if err == nil {
					t.Fatalf("resolveNativeToken err = nil, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("resolveNativeToken: %v", err)
			}
			if got != test.want {
				t.Errorf("resolveNativeToken = %q, want %q", got, test.want)
			}
		})
	}
}

func TestNativeStateDir(t *testing.T) {
	tests := []struct {
		desc string
		env  map[string]string
		want string
	}{
		{
			desc: "explicit KORTEX_STATE_DIR wins",
			env:  map[string]string{"KORTEX_STATE_DIR": "/custom/state"},
			want: "/custom/state",
		},
		{
			desc: "XDG_STATE_HOME base",
			env:  map[string]string{"XDG_STATE_HOME": "/xdg"},
			want: "/xdg/kortex/sessions",
		},
		{
			desc: "HOME fallback",
			env:  map[string]string{"HOME": "/home/u"},
			want: "/home/u/.local/state/kortex/sessions",
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			getenv := func(name string) string { return test.env[name] }
			if got := nativeStateDir(getenv); got != test.want {
				t.Errorf("nativeStateDir = %q, want %q", got, test.want)
			}
		})
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
