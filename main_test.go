package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/mcp"
	"github.com/vingarcia/kortex/internal/session"
	"github.com/vingarcia/kortex/internal/tools"
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

// TestIsSessionsYield pins the name-matching half of main's OpenClaw-semantics
// seam (isTerminalTool / isSessionsYield, see openclaw_terminal_tool.go): the
// bare name, the real MCP-prefixed form the model actually calls, and
// unrelated tool names (bare or prefixed) must each match the doc comment's
// claim.
func TestIsSessionsYield(t *testing.T) {
	tests := []struct {
		desc string
		name string
		want bool
	}{
		{desc: "bare name matches", name: "sessions_yield", want: true},
		{desc: "MCP-prefixed name matches (the form the model actually calls)", name: tools.MCPToolPrefix + "sessions_yield", want: true},
		{desc: "unrelated bare name does not match", name: "bash", want: false},
		{desc: "unrelated MCP-prefixed name does not match", name: tools.MCPToolPrefix + "sessions_announce", want: false},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			if got := isSessionsYield(test.name); got != test.want {
				t.Errorf("isSessionsYield(%q) = %v, want %v", test.name, got, test.want)
			}
		})
	}
}

// TestIsYieldedResult pins the result-decoding half of main's OpenClaw-
// semantics seam: only a clean (non-error) result whose JSON body decodes to
// exactly {"status":"yielded"} counts — an error result, a different status,
// and a malformed/non-JSON body must all report false rather than panic or
// false-positive.
func TestIsYieldedResult(t *testing.T) {
	tests := []struct {
		desc   string
		result anthropic.ToolResultBlock
		want   bool
	}{
		{desc: "clean yielded status", result: anthropic.ToolResultBlock{Content: `{"status":"yielded"}`}, want: true},
		{desc: "clean yielded status alongside an acknowledgment field", result: anthropic.ToolResultBlock{Content: `{"status":"yielded","acknowledgment":"ok"}`}, want: true},
		{desc: "tool-level error is never yielded, even with a yielded-looking body", result: anthropic.ToolResultBlock{IsError: true, Content: `{"status":"yielded"}`}, want: false},
		{desc: "a different status is not yielded", result: anthropic.ToolResultBlock{Content: `{"status":"queued"}`}, want: false},
		{desc: "non-JSON body is not yielded", result: anthropic.ToolResultBlock{Content: "error: yield refused"}, want: false},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			if got := isYieldedResult(test.result); got != test.want {
				t.Errorf("isYieldedResult(%+v) = %v, want %v", test.result, got, test.want)
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

func TestOpenInboundTap(t *testing.T) {
	t.Run("KORTEX_LOG unset creates no file and returns nil", func(t *testing.T) {
		dir := t.TempDir()
		getenv := func(string) string { return "" }
		var stderr bytes.Buffer

		if tap := openInboundTap(getenv, &stderr); tap != nil {
			tap.Close()
			t.Fatalf("openInboundTap must return nil when KORTEX_LOG is unset")
		}
		// No file: the no-log path must leave the filesystem untouched.
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatalf("reading temp dir: %v", err)
		}
		if len(entries) != 0 {
			t.Errorf("no-log path must create no file, found %d entries", len(entries))
		}
		if stderr.Len() != 0 {
			t.Errorf("no-log path must warn nothing, got: %s", stderr.String())
		}
	})

	t.Run("KORTEX_LOG set opens sibling .inbound file 0600", func(t *testing.T) {
		logPath := filepath.Join(t.TempDir(), "kortex.log")
		getenv := func(name string) string {
			if name == "KORTEX_LOG" {
				return logPath
			}
			return ""
		}
		var stderr bytes.Buffer

		tap := openInboundTap(getenv, &stderr)
		if tap == nil {
			t.Fatalf("openInboundTap returned nil with KORTEX_LOG set; stderr: %s", stderr.String())
		}
		if _, err := tap.Write([]byte("hello\n")); err != nil {
			t.Fatalf("writing to tap: %v", err)
		}
		if err := tap.Close(); err != nil {
			t.Fatalf("closing tap: %v", err)
		}

		// Sibling naming and the exact perms the .native/.facets artifacts use.
		info, err := os.Stat(logPath + ".inbound")
		if err != nil {
			t.Fatalf("stat .inbound file: %v", err)
		}
		if perm := info.Mode().Perm(); perm != 0o600 {
			t.Errorf("inbound tap perms = %o, want 600", perm)
		}
		got, err := os.ReadFile(logPath + ".inbound")
		if err != nil {
			t.Fatalf("reading .inbound file: %v", err)
		}
		if string(got) != "hello\n" {
			t.Errorf("inbound file content = %q, want %q", string(got), "hello\n")
		}
	})
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

// bypassRecordingGovernor is a session.OutputGovernor that records whether it was
// ever invoked (the draft and review array it saw) and otherwise delivers a fixed
// text. A yield turn must bypass the governor entirely, so a non-empty gotDraft or
// non-nil gotMessages after the run means the bypass failed.
type bypassRecordingGovernor struct {
	gotDraft    string
	gotMessages []anthropic.Message
	deliver     string
}

func (g *bypassRecordingGovernor) SystemPromptNote() string { return "" }

func (g *bypassRecordingGovernor) StripArchiveRefs(text string) string { return text }

func (g *bypassRecordingGovernor) GovernOutput(
	ctx context.Context,
	turnIndex int,
	messages []anthropic.Message,
	ephemeralDigest string,
	draft string,
	redraft func(ctx context.Context, tail []anthropic.Message) (string, []anthropic.Message, error),
	finalize func(ctx context.Context, tail []anthropic.Message) (string, error),
) (string, string) {
	g.gotDraft = draft
	g.gotMessages = messages
	return g.deliver, ""
}

// wireResult is the subset of the stream-json terminal result line this test
// asserts on. main may not import internal/protocol (it is not in main's
// mayDependOn), so the integration test parses the wire JSON directly instead of
// reusing the session package's protocol-typed helpers.
type wireResult struct {
	Type       string  `json:"type"`
	IsError    bool    `json:"is_error"`
	Result     string  `json:"result"`
	StopReason *string `json:"stop_reason"`
}

// TestNativeSession_SessionsYieldBypassesGovernorAndEndsTurn is the end-to-end
// (integration) half of the yield fix, placed at the main level because it wires a
// real MCP-backed dispatcher: internal/session may not depend on internal/mcp
// (see .go-arch-lint.yml — session's mayDependOn is protocol/tools/anthropic/
// history), so a session-package test cannot build the mcp.Client this exercise
// needs. main is the composition root and already depends on mcp/session/tools/
// anthropic, so the full wiring belongs here.
//
// When the turn's only tool_use is a sessions_yield that the gateway answers with
// {"status":"yielded"}, RunLoop ends the turn (LoopResult.Yielded) and runTurn must
// SKIP the governor: a yield has no user-facing draft to review, and routing its
// empty FinalText through the superego/EmitFinalTurn would do nothing useful and
// trip the empty-text guard into a spurious error result. The round must still
// close with exactly one success result (stop_reason end_turn), never the
// governor's text and never an error.
func TestNativeSession_SessionsYieldBypassesGovernorAndEndsTurn(t *testing.T) {
	// The core model's only action this turn is to yield.
	coreServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"content": [{"type":"tool_use","id":"toolu_y","name":"mcp__openclaw__sessions_yield","input":{}}],
			"stop_reason": "tool_use",
			"usage": {"input_tokens": 1, "output_tokens": 1}
		}`))
	}))
	defer coreServer.Close()

	// The MCP loopback answers the yield with the {"status":"yielded"} payload the
	// real gateway returns.
	mcpServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\"status\":\"yielded\"}"}],"isError":false}}`))
	}))
	defer mcpServer.Close()

	client := anthropic.NewClient("secret", coreServer.URL, time.Second)
	mcpClient := mcp.NewClient(mcp.ServerConfig{URL: mcpServer.URL}, nil)
	dispatcher := tools.NewDispatcherWithMCP(mcpClient, []mcp.Tool{{Name: "sessions_yield"}})
	gov := &bypassRecordingGovernor{deliver: "GOVERNED-TEXT-MUST-NOT-APPEAR"}

	initialize := `{"type":"control_request","request_id":"req-1","request":{"subtype":"initialize","appendSystemPrompt":"sys"}}`
	userMsg := `{"type":"user","session_id":"sess-1","message":{"role":"user","content":"do it then yield"}}`

	uuidN := 0
	newUUID := func() string {
		uuidN++
		return fmt.Sprintf("uuid-%d", uuidN)
	}

	var out strings.Builder
	err := session.Run(context.Background(), session.Config{
		Stdin:      strings.NewReader(initialize + "\n" + userMsg + "\n"),
		Stdout:     &out,
		Client:     client,
		Dispatcher: dispatcher,
		Store:      session.NewStore(t.TempDir()),
		SessionID:  "sess-1",
		Model:      "m",
		MaxTokens:  1024,
		NewUUID:    newUUID,
		Governor:   gov,
		// Built manually here rather than via runNative, so the yield-terminal
		// predicate must be wired explicitly too, exactly as runNative wires it —
		// this is what makes the yield actually bypass the governor below.
		IsTerminalTool: isTerminalTool,
	})
	if err != nil {
		t.Fatalf("session.Run: %v", err)
	}

	// The governor must never have been invoked: it records the draft and review
	// array it saw, so a non-empty gotDraft or non-nil gotMessages means it ran.
	if gov.gotDraft != "" || gov.gotMessages != nil {
		t.Errorf("governor ran on a yield turn (draft=%q, messages=%+v); it must be bypassed", gov.gotDraft, gov.gotMessages)
	}

	var results []wireResult
	for _, line := range strings.Split(strings.TrimSpace(out.String()), "\n") {
		if line == "" {
			continue
		}
		var res wireResult
		if err := json.Unmarshal([]byte(line), &res); err != nil {
			continue
		}
		if res.Type == "result" {
			results = append(results, res)
		}
	}
	if len(results) != 1 {
		t.Fatalf("want exactly 1 terminal result, got %d:\n%s", len(results), out.String())
	}
	res := results[0]
	if res.IsError {
		t.Errorf("yield closed with an error result: %q", res.Result)
	}
	if res.StopReason == nil || *res.StopReason != "end_turn" {
		t.Errorf("result stop_reason = %v, want end_turn", res.StopReason)
	}
	if strings.Contains(res.Result, "GOVERNED-TEXT-MUST-NOT-APPEAR") {
		t.Errorf("governed text leaked into a yield result: %q", res.Result)
	}
}

func TestAuthFDsFromEnv(t *testing.T) {
	tests := []struct {
		desc string
		env  map[string]string
		want []int
	}{
		{desc: "none set", env: map[string]string{}, want: nil},
		{
			desc: "oauth fd only",
			env:  map[string]string{"CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR": "3"},
			want: []int{3},
		},
		{
			desc: "api key fd only",
			env:  map[string]string{"CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR": "4"},
			want: []int{4},
		},
		{
			desc: "both honored in authFDEnvVars order",
			env: map[string]string{
				"CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR": "3",
				"CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR":     "4",
			},
			want: []int{3, 4},
		},
		{
			desc: "empty values are treated as unset",
			env: map[string]string{
				"CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR": "",
				"CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR":     "",
			},
			want: nil,
		},
		{
			desc: "non-numeric value is skipped, valid sibling kept",
			env: map[string]string{
				"CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR": "abc",
				"CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR":     "5",
			},
			want: []int{5},
		},
		{
			desc: "negative value is skipped, valid sibling kept",
			env: map[string]string{
				"CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR": "-1",
				"CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR":     "5",
			},
			want: []int{5},
		},
		{
			desc: "stdio fd 2 is skipped, valid sibling kept",
			env: map[string]string{
				"CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR": "2",
				"CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR":     "5",
			},
			want: []int{5},
		},
		{
			desc: "lowest forwardable fd 3 is kept",
			env: map[string]string{
				"CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR": "3",
			},
			want: []int{3},
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			// Pin every var first so the host environment cannot leak in.
			for _, name := range authFDEnvVars {
				t.Setenv(name, "")
			}
			for name, value := range test.env {
				t.Setenv(name, value)
			}

			got := authFDsFromEnv()

			if fmt.Sprint(got) != fmt.Sprint(test.want) {
				t.Errorf("authFDsFromEnv() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestBuildNativeDispatcher(t *testing.T) {
	// A fake loopback MCP server exposing one tool; initialize/tools/list
	// answered per method, the initialized notification with 202.
	newServer := func(t *testing.T, failToolsList bool) *httptest.Server {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req struct {
				Method string `json:"method"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			switch req.Method {
			case "initialize":
				w.Header().Set("Content-Type", "application/json")
				w.Write([]byte(`{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26","capabilities":{},"serverInfo":{"name":"openclaw","version":"0.1.0"}}}`))
			case "notifications/initialized":
				w.WriteHeader(http.StatusAccepted)
			case "tools/list":
				if failToolsList {
					w.WriteHeader(http.StatusInternalServerError)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				w.Write([]byte(`{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"memory_search","description":"d","inputSchema":{"type":"object","properties":{}}}]}}`))
			default:
				w.WriteHeader(http.StatusInternalServerError)
			}
		}))
		t.Cleanup(srv.Close)
		return srv
	}
	writeConfig := func(t *testing.T, url string) []string {
		path := filepath.Join(t.TempDir(), "mcp.json")
		cfg := fmt.Sprintf(`{"mcpServers":{"openclaw":{"type":"http","url":%q}}}`, url)
		if err := os.WriteFile(path, []byte(cfg), 0o600); err != nil {
			t.Fatal(err)
		}
		return []string{"--mcp-config", path}
	}
	const mcpToolName = "mcp__openclaw__memory_search"
	builtinCount := len(tools.NewDispatcher().ToolDefs())

	tests := []struct {
		desc        string
		argv        func(t *testing.T) []string
		wantMCPTool bool
		wantStderr  string
	}{
		{
			desc: "no --mcp-config: builtins only, silent",
			argv: func(t *testing.T) []string { return []string{"--model", "m"} },
		},
		{
			desc: "unreadable --mcp-config: builtins only, warns",
			argv: func(t *testing.T) []string {
				return []string{"--mcp-config", filepath.Join(t.TempDir(), "missing.json")}
			},
			wantStderr: "reading config",
		},
		{
			desc: "handshake fails: builtins only, warns",
			argv: func(t *testing.T) []string {
				return writeConfig(t, newServer(t, true).URL)
			},
			wantStderr: "handshake failed",
		},
		{
			desc: "handshake ok: MCP tools exposed after builtins",
			argv: func(t *testing.T) []string {
				return writeConfig(t, newServer(t, false).URL)
			},
			wantMCPTool: true,
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			var stderr bytes.Buffer

			dispatcher := buildNativeDispatcher(context.Background(), test.argv(t), func(string) string { return "" }, &stderr)

			if dispatcher == nil {
				t.Fatal("dispatcher = nil, want non-nil (fail-open)")
			}
			defs := dispatcher.ToolDefs()
			hasMCP := false
			for _, def := range defs {
				if def.Name == mcpToolName {
					hasMCP = true
				}
			}
			if hasMCP != test.wantMCPTool {
				t.Errorf("has %s = %v, want %v", mcpToolName, hasMCP, test.wantMCPTool)
			}
			wantLen := builtinCount
			if test.wantMCPTool {
				wantLen++
			}
			if len(defs) != wantLen {
				t.Errorf("len(ToolDefs) = %d, want %d", len(defs), wantLen)
			}
			if test.wantStderr != "" && !strings.Contains(stderr.String(), test.wantStderr) {
				t.Errorf("stderr = %q, want it to contain %q", stderr.String(), test.wantStderr)
			}
			if test.wantStderr == "" && stderr.Len() != 0 {
				t.Errorf("stderr = %q, want empty", stderr.String())
			}
		})
	}
}

func TestBuildFacetDeps(t *testing.T) {
	dir := t.TempDir()
	writeFile := func(name string, content string) string {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
		return path
	}
	prompt := writeFile("prompt.md", "header\n---\nBODY\n")
	memory := writeFile("memory.md", "MEM")
	noBody := writeFile("nobody.md", "no separator here")

	tests := []struct {
		desc        string
		token       string
		promptPath  string
		memoryPath  string
		timeoutSecs int
		wantErr     string
		wantTimeout time.Duration
		wantPrompt  []string // substrings
	}{
		{
			desc:        "default timeout applied when config is 0",
			token:       "tok",
			promptPath:  prompt,
			timeoutSecs: 0,
			wantTimeout: 7 * time.Second,
			wantPrompt:  []string{"BODY"},
		},
		{
			desc:        "explicit timeout wins; memory appended to prompt",
			token:       "tok",
			promptPath:  prompt,
			memoryPath:  memory,
			timeoutSecs: 3,
			wantTimeout: 3 * time.Second,
			wantPrompt:  []string{"BODY", "MEM"},
		},
		{desc: "empty token", token: "", promptPath: prompt, wantErr: "token env FACET_TEST_TOKEN is empty or unset"},
		{desc: "missing prompt file", token: "tok", promptPath: filepath.Join(dir, "nope.md"), wantErr: "facet testFacet: reading facet system prompt"},
		{desc: "prompt without body", token: "tok", promptPath: noBody, wantErr: "has no body"},
		{desc: "missing memory file", token: "tok", promptPath: prompt, memoryPath: filepath.Join(dir, "nomem.md"), wantErr: "reading emotional memory"},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			t.Setenv("FACET_TEST_TOKEN", test.token)

			client, systemPrompt, timeout, err := buildFacetDeps("testFacet", "FACET_TEST_TOKEN", test.promptPath, test.memoryPath, test.timeoutSecs, 7)

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
			if client == nil {
				t.Error("client = nil, want non-nil")
			}
			if timeout != test.wantTimeout {
				t.Errorf("timeout = %v, want %v", timeout, test.wantTimeout)
			}
			for _, want := range test.wantPrompt {
				if !strings.Contains(systemPrompt, want) {
					t.Errorf("systemPrompt = %q, want it to contain %q", systemPrompt, want)
				}
			}
		})
	}
}

func TestBuildFacets(t *testing.T) {
	dir := t.TempDir()
	prompt := filepath.Join(dir, "prompt.md")
	if err := os.WriteFile(prompt, []byte("---\nBODY\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FACET_TEST_TOKEN", "tok")

	facetJSON := func(name string, extra string) string {
		return fmt.Sprintf(`%q:{"enabled":true,"model":"m","tokenEnv":"FACET_TEST_TOKEN","systemPromptPath":%q%s}`, name, prompt, extra)
	}
	writeConfig := func(t *testing.T, facets ...string) string {
		path := filepath.Join(t.TempDir(), "kortex.json")
		body := `{"facets":{` + strings.Join(facets, ",") + `}}`
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		return path
	}

	tests := []struct {
		desc            string
		cfg             func(t *testing.T) string
		withLog         bool
		wantErr         bool
		wantAnnotator   bool
		wantEvaluator   bool
		wantSuperego    bool
		wantGovernor    bool
		wantBudget      int
		wantConflictLog bool
	}{
		{desc: "empty config path: nothing built", cfg: func(t *testing.T) string { return "" }},
		{desc: "unreadable config", cfg: func(t *testing.T) string { return filepath.Join(t.TempDir(), "nope.json") }, wantErr: true},
		{
			desc: "no facet enabled: nothing built",
			cfg: func(t *testing.T) string {
				return writeConfig(t, strings.Replace(facetJSON("inputAnnotator", ""), `"enabled":true`, `"enabled":false`, 1))
			},
		},
		{
			desc:          "annotator only",
			cfg:           func(t *testing.T) string { return writeConfig(t, facetJSON("inputAnnotator", "")) },
			withLog:       true,
			wantAnnotator: true,
		},
		{
			desc:          "evaluator only",
			cfg:           func(t *testing.T) string { return writeConfig(t, facetJSON("outputEvaluator", "")) },
			wantEvaluator: true,
		},
		{
			desc: "shadow superego: evaluator triggers it, no governor",
			cfg: func(t *testing.T) string {
				return writeConfig(t, facetJSON("outputEvaluator", ""), facetJSON("superego", `,"mode":"shadow"`))
			},
			wantEvaluator: true,
			wantSuperego:  true,
		},
		{
			desc: "active superego: governor, default redraft budget, conflicts log",
			cfg: func(t *testing.T) string {
				return writeConfig(t, facetJSON("outputEvaluator", ""), facetJSON("superego", `,"mode":"active"`))
			},
			withLog:         true,
			wantEvaluator:   true,
			wantSuperego:    true,
			wantGovernor:    true,
			wantBudget:      session.DefaultRedraftToolBudget,
			wantConflictLog: true,
		},
		{
			desc: "active superego: configured redraft budget honored, no log path",
			cfg: func(t *testing.T) string {
				return writeConfig(t, facetJSON("outputEvaluator", ""), facetJSON("superego", `,"mode":"active","redraftToolBudget":9`))
			},
			wantEvaluator: true,
			wantSuperego:  true,
			wantGovernor:  true,
			wantBudget:    9,
		},
		{
			desc: "unusable facet deps (prompt missing) surface as error",
			cfg: func(t *testing.T) string {
				path := filepath.Join(t.TempDir(), "kortex.json")
				body := fmt.Sprintf(`{"facets":{"inputAnnotator":{"enabled":true,"model":"m","tokenEnv":"FACET_TEST_TOKEN","systemPromptPath":%q}}}`, filepath.Join(dir, "gone.md"))
				if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
					t.Fatal(err)
				}
				return path
			},
			wantErr: true,
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			logPath := ""
			if test.withLog {
				logPath = filepath.Join(t.TempDir(), "kortex.log")
			}

			built, err := buildFacets(test.cfg(t), logPath, "sess-1")
			t.Cleanup(func() {
				if built.logFile != nil {
					built.logFile.Close()
				}
				if built.conflictsLogFile != nil {
					built.conflictsLogFile.Close()
				}
			})

			if test.wantErr {
				if err == nil {
					t.Fatal("err = nil, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("err = %v, want nil", err)
			}
			if got := built.annotator != nil; got != test.wantAnnotator {
				t.Errorf("annotator built = %v, want %v", got, test.wantAnnotator)
			}
			if got := built.outputEvaluator != nil; got != test.wantEvaluator {
				t.Errorf("outputEvaluator built = %v, want %v", got, test.wantEvaluator)
			}
			if got := built.superego != nil; got != test.wantSuperego {
				t.Errorf("superego built = %v, want %v", got, test.wantSuperego)
			}
			if got := built.governor != nil; got != test.wantGovernor {
				t.Errorf("governor built = %v, want %v", got, test.wantGovernor)
			}
			if built.redraftToolBudget != test.wantBudget {
				t.Errorf("redraftToolBudget = %d, want %d", built.redraftToolBudget, test.wantBudget)
			}
			if got := built.conflictsLogFile != nil; got != test.wantConflictLog {
				t.Errorf("conflictsLogFile opened = %v, want %v", got, test.wantConflictLog)
			}
			wantFacetLog := test.withLog && (test.wantAnnotator || test.wantEvaluator)
			if got := built.logFile != nil; got != wantFacetLog {
				t.Errorf("logFile opened = %v, want %v", got, wantFacetLog)
			}
			if wantFacetLog {
				if _, err := os.Stat(logPath + ".facets"); err != nil {
					t.Errorf("facets log not created: %v", err)
				}
			}
		})
	}
}

func TestRunNative_emptyStdinCleanExit(t *testing.T) {
	// EOF on stdin ends session.Run cleanly with no model call, so this drives
	// the whole composition root (token file, wire log, inbound tap, facets,
	// dispatcher) without touching the network.
	dir := t.TempDir()
	tokenFile := filepath.Join(dir, "token")
	if err := os.WriteFile(tokenFile, []byte("secret\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	prompt := filepath.Join(dir, "prompt.md")
	if err := os.WriteFile(prompt, []byte("---\nBODY\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	cfgPath := filepath.Join(dir, "kortex.json")
	cfg := fmt.Sprintf(`{"facets":{
		"inputAnnotator":{"enabled":true,"model":"m","tokenEnv":"FACET_TEST_TOKEN","systemPromptPath":%[1]q},
		"outputEvaluator":{"enabled":true,"model":"m","tokenEnv":"FACET_TEST_TOKEN","systemPromptPath":%[1]q},
		"superego":{"enabled":true,"mode":"active","tokenEnv":"FACET_TEST_TOKEN","systemPromptPath":%[1]q}}}`, prompt)
	if err := os.WriteFile(cfgPath, []byte(cfg), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FACET_TEST_TOKEN", "tok")
	logPath := filepath.Join(dir, "kortex.log")
	env := map[string]string{
		"KORTEX_TOKEN_FILE": tokenFile,
		"KORTEX_CONFIG":     cfgPath,
		"KORTEX_LOG":        logPath,
		"KORTEX_STATE_DIR":  filepath.Join(dir, "state"),
	}
	getenv := func(name string) string { return env[name] }
	var stdout, stderr bytes.Buffer

	code := runNative([]string{"--model", "m", "--effort", "low", "--session-id", "sess-1"}, getenv, strings.NewReader(""), &stdout, &stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q", code, stderr.String())
	}
	if stdout.Len() != 0 {
		t.Errorf("stdout = %q, want empty (no input, no output)", stdout.String())
	}
	// History is written lazily (first turn), so no turn means no state; any
	// state that does appear must be under the injected KORTEX_STATE_DIR.
	if entries, err := os.ReadDir(env["KORTEX_STATE_DIR"]); err == nil && len(entries) != 0 {
		t.Errorf("unexpected state written with no turns: %v", entries)
	}
	for _, suffix := range []string{".native", ".inbound", ".facets", ".conflicts"} {
		if _, err := os.Stat(logPath + suffix); err != nil {
			t.Errorf("expected %s artifact: %v", suffix, err)
		}
	}
}
