package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/config"
	"github.com/vingarcia/kortex/internal/facet"
	"github.com/vingarcia/kortex/internal/mcp"
	"github.com/vingarcia/kortex/internal/proxy"
)

// Env vars OpenClaw uses to announce the FD carrying the auth secret
// (spec F0 §2). Only one is set per spawn; both are honored if present.
var authFDEnvVars = []string{
	"CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
	"CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
}

func main() {
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
	logFile         *os.File
	mcpClient       *mcp.Client
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
	if evalEnabled {
		// The superego is built before its trigger (the output evaluator)
		// so the evaluator can be handed the wired trigger at construction.
		if seEnabled {
			model := seCfg.Model
			if model == "" {
				model = config.DefaultSuperegoModel
			}
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
		built.outputEvaluator = facet.NewOutputEvaluator(facet.OutputEvaluatorParams{
			Evaluator:         facet.NewAnthropicEvaluator(client, evalCfg.Model),
			SystemPrompt:      systemPrompt,
			Timeout:           timeout,
			Model:             evalCfg.Model,
			GateMinInvestment: gateMinInvestment,
			Superego:          built.superego,
			LogSink:           logSink,
		})
	}
	return built, nil
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
// "--strict-mcp-config" (§F3c LIVE HANDSHAKE DUMP). It accepts both the
// space-separated form ("--mcp-config", "<path>", observed) and the
// "--mcp-config=<path>" form (not observed, but cheap to also accept and
// avoids a silent miss if claude-cli's flag parser allows it). Returns ""
// when the flag is absent.
func mcpConfigPathFromArgv(argv []string) string {
	const flag = "--mcp-config"
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
