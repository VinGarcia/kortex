package main

import (
	"context"
	"crypto/rand"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/config"
	"github.com/vingarcia/kortex/internal/facet"
	"github.com/vingarcia/kortex/internal/mcp"
	"github.com/vingarcia/kortex/internal/proxy"
	"github.com/vingarcia/kortex/internal/session"
	"github.com/vingarcia/kortex/internal/tools"
)

// Env vars OpenClaw uses to announce the FD carrying the auth secret
// (spec F0 §2). Only one is set per spawn; both are honored if present.
var authFDEnvVars = []string{
	"CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
	"CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
}

func main() {
	// F3d flip: when the operator opts in through KORTEX_STREAM_EMITTER, kortex
	// does NOT exec the real claude-cli. It runs the session natively — reading
	// the NDJSON protocol from stdin, driving the tool-loop, and producing
	// stream-json on stdout — so OpenClaw cannot tell it apart from the real CLI.
	// Unset/non-affirmative keeps the byte-identical passthrough below untouched.
	if streamEmitterEnabled(os.Getenv("KORTEX_STREAM_EMITTER")) {
		os.Exit(runNative(os.Args[1:], os.Getenv, os.Stdin, os.Stdout, os.Stderr))
	}

	selfPath, err := os.Executable()
	if err != nil {
		selfPath = os.Args[0]
	}
	searchDirs := filepath.SplitList(os.Getenv("PATH"))
	claudeBin, err := proxy.ResolveClaudeBin(os.Getenv("KORTEX_CLAUDE_BIN"), selfPath, searchDirs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "kortex: %v\n", err)
		os.Exit(1)
	}
	logPath := os.Getenv("KORTEX_LOG")
	eventLogPath := ""
	if logPath != "" {
		eventLogPath = logPath + ".events"
	}

	// KORTEX_CONFIG is the facet feature flag: absent means no config, no
	// facets, pure passthrough — F1 behavior untouched. A present but
	// broken config fails fast at startup (before any exec): silently
	// degrading a flag the operator explicitly set would hide the
	// misconfiguration; a startup failure is immediately visible.
	facets, err := buildFacets(os.Getenv("KORTEX_CONFIG"), logPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "kortex: %v\n", err)
		os.Exit(1)
	}
	// --mcp-config is unconditionally injected by OpenClaw on the live path
	// (F3c), whether or not kortex has a consumer for it yet — unlike
	// KORTEX_CONFIG, a parse failure here is fail-open (warn, nil client),
	// not fatal: nothing depends on the MCP client in Run() yet (see
	// buildMCPClient's doc comment), so refusing to start over it would put
	// today's passthrough/facets behavior at risk for zero benefit.
	mcpClient, err := buildMCPClient(os.Args[1:], os.Getenv)
	if err != nil {
		fmt.Fprintf(os.Stderr, "kortex: mcp client disabled: %v\n", err)
	} else {
		facets.mcpClient = mcpClient
	}
	var interceptor proxy.Interceptor
	if facets.annotator != nil {
		interceptor = facets.annotator
	}
	var turnEvaluator proxy.TurnEvaluator
	if facets.outputEvaluator != nil {
		turnEvaluator = facets.outputEvaluator
	}

	// os.Exit skips defers, so the facet log is closed explicitly after Run.
	code := proxy.Run(proxy.Config{
		ClaudeBin:     claudeBin,
		Argv:          os.Args[1:],
		AuthFDs:       authFDsFromEnv(),
		LogPath:       logPath,
		EventLogPath:  eventLogPath,
		Interceptor:   interceptor,
		TurnEvaluator: turnEvaluator,
		Stdin:         os.Stdin,
		Stdout:        os.Stdout,
		Stderr:        os.Stderr,
	})
	// The child already exited and its output is long delivered; waiting
	// here only keeps the process alive until the final turn's async
	// evaluation lands in the log (bounded by the facet timeout), instead
	// of killing it by exiting.
	if facets.outputEvaluator != nil {
		facets.outputEvaluator.WaitInFlight()
	}
	// After the evaluators drained, every superego dispatch already
	// happened; now wait for the reviews themselves (this order matters:
	// waiting on the superego first could miss a review dispatched by a
	// still-running evaluation).
	if facets.superego != nil {
		facets.superego.WaitInFlight()
	}
	if facets.logFile != nil {
		facets.logFile.Close()
	}
	os.Exit(code)
}

// builtFacets is what the composition root assembled: the constructed
// facets (each nil when absent or disabled) plus the facet-log file they
// share (nil when logging is off; the caller owns closing it), plus the MCP
// client (nil when --mcp-config was absent or unparseable; see
// buildMCPClient).
type builtFacets struct {
	annotator       *facet.InputAnnotator
	outputEvaluator *facet.OutputEvaluator
	superego        *facet.Superego
	// governor is non-nil only in superego mode "active": the blocking
	// evaluate→review→revise loop that governs the delivered text. In that mode
	// the output evaluator is built WITHOUT its shadow trigger (the governor
	// drives the superego directly), so the session wires the governor instead
	// of the async TurnEvaluator.
	governor  *facet.ActiveSuperego
	logFile   *os.File
	mcpClient *mcp.Client
}

// buildFacets loads the config at cfgPath and constructs every enabled
// facet. The zero builtFacets when cfgPath is empty. All env reading stays
// here in the composition root: a facet receives the token, never the env
// name.
func buildFacets(cfgPath string, logPath string) (builtFacets, error) {
	var built builtFacets
	if cfgPath == "" {
		return built, nil
	}
	cfg, err := config.Load(cfgPath)
	if err != nil {
		return built, err
	}
	annCfg := cfg.Facets.InputAnnotator
	evalCfg := cfg.Facets.OutputEvaluator
	seCfg := cfg.Facets.Superego
	annEnabled := annCfg != nil && annCfg.Enabled
	evalEnabled := evalCfg != nil && evalCfg.Enabled
	seEnabled := seCfg != nil && seCfg.Enabled
	if !annEnabled && !evalEnabled {
		// Config.Validate guarantees seEnabled implies evalEnabled, so no
		// superego can exist on this path either.
		return built, nil
	}

	// The structured facet-call log lives next to the other KORTEX_LOG
	// artifacts, shared by every facet (each log line names its facet). An
	// open failure degrades to no facet log (stderr warning), never to no
	// facet.
	var logSink io.Writer
	if logPath != "" {
		built.logFile, err = os.OpenFile(logPath+".facets", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
		if err != nil {
			fmt.Fprintf(os.Stderr, "kortex: facet log disabled: %v\n", err)
			built.logFile = nil
		} else {
			logSink = built.logFile
		}
	}

	if annEnabled {
		client, systemPrompt, timeout, err := buildFacetDeps("inputAnnotator", annCfg.TokenEnv,
			annCfg.SystemPromptPath, annCfg.EmotionalMemoryPath, annCfg.TimeoutSeconds, config.DefaultTimeoutSeconds)
		if err != nil {
			return built, err
		}
		built.annotator = facet.NewInputAnnotator(facet.AnnotatorParams{
			Evaluator:    facet.NewAnthropicEvaluator(client, annCfg.Model),
			SystemPrompt: systemPrompt,
			Timeout:      timeout,
			Model:        annCfg.Model,
			LogSink:      logSink,
		})
	}
	// active superego mode replaces the async shadow review with the blocking
	// governor; the output evaluator is then built without its shadow trigger.
	active := seEnabled && seCfg.Mode == config.SuperegoModeActive
	if evalEnabled {
		// The superego is built before its trigger (the output evaluator)
		// so the evaluator can be handed the wired trigger at construction.
		if seEnabled {
			model := resolveSuperegoModel(seCfg.Model)
			client, systemPrompt, timeout, err := buildFacetDeps("superego", seCfg.TokenEnv,
				seCfg.SystemPromptPath, "", seCfg.TimeoutSeconds, config.DefaultSuperegoTimeoutSeconds)
			if err != nil {
				return built, err
			}
			built.superego = facet.NewSuperego(facet.SuperegoParams{
				Conversant:      facet.NewAnthropicConversant(client, model),
				SystemPrompt:    systemPrompt,
				Timeout:         timeout,
				Model:           model,
				MaxHistoryTurns: seCfg.MaxHistoryTurns,
				LogSink:         logSink,
			})
		}
		client, systemPrompt, timeout, err := buildFacetDeps("outputEvaluator", evalCfg.TokenEnv,
			evalCfg.SystemPromptPath, evalCfg.EmotionalMemoryPath, evalCfg.TimeoutSeconds, config.DefaultTimeoutSeconds)
		if err != nil {
			return built, err
		}
		gateMinInvestment := config.DefaultGateMinInvestment
		if evalCfg.GateMinInvestment != nil {
			gateMinInvestment = *evalCfg.GateMinInvestment
		}
		// In active mode the governor drives the superego synchronously, so the
		// evaluator gets no shadow trigger; in shadow mode it keeps firing the
		// async review on a gate hit.
		shadowTrigger := built.superego
		if active {
			shadowTrigger = nil
		}
		built.outputEvaluator = facet.NewOutputEvaluator(facet.OutputEvaluatorParams{
			Evaluator:         facet.NewAnthropicEvaluator(client, evalCfg.Model),
			SystemPrompt:      systemPrompt,
			Timeout:           timeout,
			Model:             evalCfg.Model,
			GateMinInvestment: gateMinInvestment,
			Superego:          shadowTrigger,
			LogSink:           logSink,
		})
	}
	if active {
		built.governor = facet.NewActiveSuperego(facet.ActiveSuperegoParams{
			Evaluator: built.outputEvaluator,
			Superego:  built.superego,
			Model:     resolveSuperegoModel(seCfg.Model),
			LogSink:   logSink,
		})
	}
	return built, nil
}

// resolveSuperegoModel lets a config omit model and still get the facet
// table's Opus default, so that choice holds without every config restating it.
func resolveSuperegoModel(configured string) string {
	if configured == "" {
		return config.DefaultSuperegoModel
	}
	return configured
}

// buildFacetDeps assembles the model-call dependencies every facet shares:
// the API client (token resolved from the env here, in the composition
// root), the loaded system prompt, and the call timeout. The caller wraps
// the client in its facet's own port (evaluator/conversant).
func buildFacetDeps(facetName string, tokenEnv string, promptPath string, memoryPath string, timeoutSeconds int, defaultTimeoutSeconds int) (*anthropic.Client, string, time.Duration, error) {
	token := os.Getenv(tokenEnv)
	if token == "" {
		return nil, "", 0, fmt.Errorf("facet %s: token env %s is empty or unset", facetName, tokenEnv)
	}
	systemPrompt, err := facet.LoadSystemPrompt(promptPath, memoryPath)
	if err != nil {
		return nil, "", 0, fmt.Errorf("facet %s: %w", facetName, err)
	}
	if timeoutSeconds == 0 {
		timeoutSeconds = defaultTimeoutSeconds
	}
	// The HTTP client gets no client-level timeout: the facet bounds every
	// call with a context deadline derived from the same config field.
	return anthropic.NewClient(token, "", 0), systemPrompt, time.Duration(timeoutSeconds) * time.Second, nil
}

// buildMCPClient constructs the MCP client for the OpenClaw loopback server.
// getenv is injected (main passes os.Getenv; tests pass a fake lookup) so
// internal/mcp never reads the process environment itself. It is fail-open:
// a nil client and nil error when --mcp-config is absent from argv, since no
// consumer wires the client into the production path yet (kortex still execs
// the real claude-cli).
func buildMCPClient(argv []string, getenv func(string) string) (*mcp.Client, error) {
	path := mcpConfigPathFromArgv(argv)
	if path == "" {
		return nil, nil
	}
	client, err := mcp.NewClientFromConfigFile(path, getenv, nil)
	if err != nil {
		return nil, fmt.Errorf("mcp client: %w", err)
	}
	return client, nil
}

// mcpConfigPathFromArgv scans argv for the "--mcp-config" flag OpenClaw
// passes the subprocess-CLI (kortex, impersonating claude-cli) alongside
// "--strict-mcp-config" (§F3c LIVE HANDSHAKE DUMP). Returns "" when absent.
func mcpConfigPathFromArgv(argv []string) string {
	return firstArgvValue(argv, "--mcp-config")
}

// firstArgvValue returns the value of flag in argv, accepting both the
// space-separated form ("--flag", "value", as OpenClaw passes --mcp-config and
// --model) and the "--flag=value" form (cheap to also accept, avoids a silent
// miss if claude-cli's parser allows it). Returns "" when the flag is absent or
// is the last element with no following value.
func firstArgvValue(argv []string, flag string) string {
	for i, arg := range argv {
		if value, ok := strings.CutPrefix(arg, flag+"="); ok {
			return value
		}
		if arg == flag && i+1 < len(argv) {
			return argv[i+1]
		}
	}
	return ""
}

// defaultMaxTokens caps each native turn's response length. It is fixed in v0
// (no env override). The canary core model (fable) thinks adaptively, and
// max_tokens is the shared budget for thinking AND the visible text: at 8192 a
// deep thinking pass could consume the whole budget and leave zero text, which
// OpenClaw rejects as an empty turn (observed live 2026-10-04). 16000 gives the
// text room to land after thinking while staying well within the model's output
// ceiling; the 120s per-call deadline (defaultLoopCallTimeout) is the real wall
// on a long generation, not this cap. Effort (see runNative) bounds how much of
// this budget thinking itself takes.
const defaultMaxTokens = 16000

// defaultRedraftTimeout bounds one redraft call to the core model inside the
// active superego loop. Fixed in v0: sized to the 60s superego magnitude
// (config.DefaultSuperegoTimeoutSeconds), since a redraft writes a full prose
// response rather than a short facet reply.
const defaultRedraftTimeout = 60 * time.Second

// defaultLoopCallTimeout bounds each individual core-model call inside the
// native tool-loop. This is the PRIMARY turn-generation call (it may
// legitimately run longer than the quick redraft), so the bound is generous —
// large enough to let a normal long generation finish, finite enough to catch
// a truly hung call instead of blocking the turn forever. Unlike the redraft
// (which fails open to the raw draft), a timed-out loop call fails closed.
const defaultLoopCallTimeout = 120 * time.Second

// resolveNativeToken resolves the OAuth token for the native path from the
// composition root. KORTEX_TOKEN_FILE, when set, names a file kortex reads and
// trims itself, and takes precedence over CODECOMPANION_OAUTH_TOKEN: it lets
// the launcher hand kortex a path instead of injecting the secret into the
// child's environment. A set-but-unreadable KORTEX_TOKEN_FILE is a hard error
// (a misconfigured launcher must fail loudly, not silently fall back to a stale
// env token). Absent KORTEX_TOKEN_FILE falls back to CODECOMPANION_OAUTH_TOKEN.
func resolveNativeToken(getenv func(string) string) (string, error) {
	if path := getenv("KORTEX_TOKEN_FILE"); path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			return "", fmt.Errorf("reading KORTEX_TOKEN_FILE %s: %w", path, err)
		}
		return strings.TrimSpace(string(data)), nil
	}
	return getenv("CODECOMPANION_OAUTH_TOKEN"), nil
}

// runNative runs the F3d producer path: kortex speaks the stream-json protocol
// itself instead of execing the real claude-cli. It wires the composition-root
// dependencies (Anthropic client from CODECOMPANION_OAUTH_TOKEN, the builtin +
// MCP tool dispatcher, the per-session history store) and hands them to
// session.Run. Returns the process exit code.
//
// v0 limitations (documented for the flip): no hook callbacks fire on this path
// (OpenClaw's declared user-scope hooks simply don't run), and interrupts and
// permission prompts are not implemented — kortex drives its own tools.
func runNative(argv []string, getenv func(string) string, stdin io.Reader, stdout io.Writer, stderr io.Writer) int {
	token, err := resolveNativeToken(getenv)
	if err != nil {
		fmt.Fprintf(stderr, "kortex: %v\n", err)
		return 1
	}
	if token == "" {
		fmt.Fprintln(stderr, "kortex: native mode needs KORTEX_TOKEN_FILE or CODECOMPANION_OAUTH_TOKEN; not set")
		return 1
	}
	// OpenClaw always passes --model, but guard it up front: an empty model
	// would otherwise fail every turn's Messages call, turning one clear
	// misconfiguration into a stream of per-turn error results.
	model := firstArgvValue(argv, "--model")
	if model == "" {
		fmt.Fprintln(stderr, "kortex: native mode needs --model in argv; not found")
		return 1
	}
	// OpenClaw passes --effort in argv; forward it to every core-model call as
	// output_config.effort. Bounding effort keeps a thinking model (fable) from
	// spending the whole max_tokens budget on thinking and returning no text.
	// Empty is fine: the API default stands.
	effort := firstArgvValue(argv, "--effort")

	// SIGINT/SIGTERM cancel the in-flight turn so the process exits promptly
	// instead of blocking on a long Messages call.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// The same composition-root facet wiring the passthrough path uses: a
	// present KORTEX_CONFIG builds the input annotator, output evaluator and
	// (gate-triggered) superego; absent/empty builds none. A broken config
	// fails fast here, before any turn runs — mirroring the passthrough path,
	// where silently degrading an explicitly-set flag would hide the error.
	facets, err := buildFacets(getenv("KORTEX_CONFIG"), getenv("KORTEX_LOG"))
	if err != nil {
		fmt.Fprintf(stderr, "kortex: %v\n", err)
		return 1
	}
	// os.Exit (in main) skips defers; the native path returns its code, so a
	// defer is safe and drains the facets before the process leaves: wait the
	// evaluators first, then the superegos they dispatched (same ordering as
	// the passthrough path), then close the shared facet log.
	defer func() {
		if facets.outputEvaluator != nil {
			facets.outputEvaluator.WaitInFlight()
		}
		if facets.superego != nil {
			facets.superego.WaitInFlight()
		}
		if facets.logFile != nil {
			facets.logFile.Close()
		}
	}()
	// A typed nil (e.g. a nil *facet.InputAnnotator) assigned straight into an
	// interface field is itself non-nil, so the facet's nil-check would miss
	// and a nil receiver would panic; guard each the way the passthrough path
	// does before handing it to the session.
	var annotator session.Annotator
	if facets.annotator != nil {
		annotator = facets.annotator
	}
	// The async TurnEvaluator and the active Governor are mutually exclusive: in
	// active mode the governor evaluates synchronously, so wiring the shadow
	// evaluator too would re-evaluate every draft for nothing. Governor present
	// ⇒ leave TurnEvaluator nil.
	var turnEvaluator session.TurnEvaluator
	if facets.outputEvaluator != nil && facets.governor == nil {
		turnEvaluator = facets.outputEvaluator
	}
	var governor session.OutputGovernor
	if facets.governor != nil {
		governor = facets.governor
	}
	// The ephemeral tone digest activates whenever the input annotator is on:
	// without annotations there is nothing to digest (it returns "" and injects
	// nothing), so the annotator's presence is the natural gate — no separate
	// config flag. minLevel 0 includes every emotion level (the task's tie-breaker
	// on the design's ambiguous >=3 cut, which authoritatively governs the diary,
	// not this digest).
	var toneDigester session.ToneDigester
	if facets.annotator != nil {
		toneDigester = facet.NewToneDigest(0)
	}

	// A client-level timeout of 0 leaves each turn bounded only by ctx: a
	// legitimate long generation (a deep tool loop) must not be cut off, and
	// OpenClaw owns the outer round deadline.
	client := anthropic.NewClient(token, "", 0)

	// Wire the prompt-cache watchdog on the CORE client only (the superego and
	// evaluator facets hold their own clients): it makes silent cache-cost
	// degradation LOUD by emitting a throttled structured WARN to stderr when a
	// core turn's stable system prefix fails to cache. It observes only — a fault
	// in it never breaks the turn. 0 cooldown selects the package default.
	client.SetCacheWatchdog(anthropic.NewCacheWatchdog(stderr, 0))

	// When KORTEX_LOG is set, mirror the passthrough path's wire log onto the
	// native path: the real claude-cli's traffic is captured by trafficLogger,
	// but the native client talks to the API directly, so its request/response
	// bodies are logged to a sibling "<KORTEX_LOG>.native" file. This is the only
	// way to see the core model's raw stop_reason and content blocks for
	// diagnosis. The OAuth token never reaches this file: it travels in the
	// Authorization header, which the client logs nothing of (see logWire). A
	// failed open degrades gracefully — the turn still runs, we just warn once.
	if logPath := getenv("KORTEX_LOG"); logPath != "" {
		wireFile, err := os.OpenFile(logPath+".native", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
		if err != nil {
			fmt.Fprintf(stderr, "kortex: native wire log disabled: %v\n", err)
		} else {
			defer wireFile.Close()
			client.SetWireLog(wireFile)
		}
	}

	dispatcher := buildNativeDispatcher(ctx, argv, getenv, stderr)

	sessionID := nativeSessionID(argv, newUUIDv4)
	err = session.Run(ctx, session.Config{
		Stdin:           stdin,
		Stdout:          stdout,
		Client:          client,
		Dispatcher:      dispatcher,
		Store:           session.NewStore(nativeStateDir(getenv)),
		SessionID:       sessionID,
		Model:           model,
		Effort:          effort,
		MaxTokens:       defaultMaxTokens,
		RedraftTimeout:  defaultRedraftTimeout,
		LoopCallTimeout: defaultLoopCallTimeout,
		NewUUID:         newUUIDv4,
		Diag:            stderr,
		Annotator:       annotator,
		TurnEvaluator:   turnEvaluator,
		Governor:        governor,
		ToneDigester:    toneDigester,
	})
	if err != nil {
		fmt.Fprintf(stderr, "kortex: native session: %v\n", err)
		return 1
	}
	return 0
}

// buildNativeDispatcher wires the tool dispatcher for the native path: the
// builtins always, plus the gateway's MCP tools when --mcp-config is present and
// the handshake succeeds. A missing config or a failed handshake degrades to
// builtins-only (warn, fail-open) rather than aborting the session — the canary
// stays useful for shell/git/fs work even if the loopback server is unreachable.
func buildNativeDispatcher(ctx context.Context, argv []string, getenv func(string) string, stderr io.Writer) *tools.Dispatcher {
	client, err := buildMCPClient(argv, getenv)
	if err != nil {
		fmt.Fprintf(stderr, "kortex: mcp tools disabled: %v\n", err)
		return tools.NewDispatcher()
	}
	if client == nil {
		return tools.NewDispatcher()
	}
	handshake, err := client.Handshake(ctx)
	if err != nil {
		fmt.Fprintf(stderr, "kortex: mcp tools disabled: handshake failed: %v\n", err)
		return tools.NewDispatcher()
	}
	return tools.NewDispatcherWithMCP(client, handshake.Tools)
}

// nativeSessionID resolves the session id this process owns from argv
// (--session-id for a fresh session, --resume for a continuation), falling back
// to a freshly minted v4 id when OpenClaw passed neither.
func nativeSessionID(argv []string, newUUID func() string) string {
	if id := firstArgvValue(argv, "--session-id"); id != "" {
		return id
	}
	if id := firstArgvValue(argv, "--resume"); id != "" {
		return id
	}
	return newUUID()
}

// nativeStateDir resolves where per-session history is persisted:
// KORTEX_STATE_DIR when set, else ${XDG_STATE_HOME:-$HOME/.local/state}/kortex/sessions.
func nativeStateDir(getenv func(string) string) string {
	if dir := getenv("KORTEX_STATE_DIR"); dir != "" {
		return dir
	}
	base := getenv("XDG_STATE_HOME")
	if base == "" {
		base = filepath.Join(getenv("HOME"), ".local", "state")
	}
	return filepath.Join(base, "kortex", "sessions")
}

// streamEmitterEnabled reads the KORTEX_STREAM_EMITTER gate. Only an explicit
// affirmative turns the F3d producer on, so an unset or unrelated value keeps
// the default pass-through path.
func streamEmitterEnabled(gate string) bool {
	switch strings.ToLower(strings.TrimSpace(gate)) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// newUUIDv4 mints a random RFC 4122 v4 UUID for the stream-json identity the
// protocol marshallers refuse to invent (the session id and each message
// uuid). crypto/rand never fails on the platforms kortex runs on, so a read
// error is unrecoverable rather than a case to degrade: panicking surfaces a
// broken entropy source instead of emitting a predictable or malformed id.
func newUUIDv4() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(fmt.Sprintf("kortex: reading random for uuid: %v", err))
	}
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

func authFDsFromEnv() []int {
	var fds []int
	for _, name := range authFDEnvVars {
		value := os.Getenv(name)
		if value == "" {
			continue
		}
		fd, err := strconv.Atoi(value)
		if err != nil {
			fmt.Fprintf(os.Stderr, "kortex: auth fd not forwarded: %s=%q is not a number\n", name, value)
			continue
		}
		fds = append(fds, fd)
	}
	return fds
}
