package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/config"
	"github.com/vingarcia/kortex/internal/facet"
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
	if facets.logFile != nil {
		facets.logFile.Close()
	}
	os.Exit(code)
}

// builtFacets is what the composition root assembled from the config: the
// constructed facets (each nil when absent or disabled) plus the facet-log
// file they share (nil when logging is off; the caller owns closing it).
type builtFacets struct {
	annotator       *facet.InputAnnotator
	outputEvaluator *facet.OutputEvaluator
	logFile         *os.File
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
	annEnabled := annCfg != nil && annCfg.Enabled
	evalEnabled := evalCfg != nil && evalCfg.Enabled
	if !annEnabled && !evalEnabled {
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
		evaluator, systemPrompt, timeout, err := buildEvaluatorDeps("inputAnnotator", annCfg.TokenEnv,
			annCfg.SystemPromptPath, annCfg.EmotionalMemoryPath, annCfg.Model, annCfg.TimeoutSeconds)
		if err != nil {
			return built, err
		}
		built.annotator = facet.NewInputAnnotator(facet.AnnotatorParams{
			Evaluator:    evaluator,
			SystemPrompt: systemPrompt,
			Timeout:      timeout,
			Model:        annCfg.Model,
			LogSink:      logSink,
		})
	}
	if evalEnabled {
		evaluator, systemPrompt, timeout, err := buildEvaluatorDeps("outputEvaluator", evalCfg.TokenEnv,
			evalCfg.SystemPromptPath, evalCfg.EmotionalMemoryPath, evalCfg.Model, evalCfg.TimeoutSeconds)
		if err != nil {
			return built, err
		}
		gateMinInvestment := config.DefaultGateMinInvestment
		if evalCfg.GateMinInvestment != nil {
			gateMinInvestment = *evalCfg.GateMinInvestment
		}
		built.outputEvaluator = facet.NewOutputEvaluator(facet.OutputEvaluatorParams{
			Evaluator:         evaluator,
			SystemPrompt:      systemPrompt,
			Timeout:           timeout,
			Model:             evalCfg.Model,
			GateMinInvestment: gateMinInvestment,
			LogSink:           logSink,
		})
	}
	return built, nil
}

// buildEvaluatorDeps assembles the model-call dependencies one facet needs:
// the anthropic-backed evaluator (token resolved from the env here, in the
// composition root), the loaded system prompt, and the call timeout.
func buildEvaluatorDeps(facetName string, tokenEnv string, promptPath string, memoryPath string, model string, timeoutSeconds int) (facet.Evaluator, string, time.Duration, error) {
	token := os.Getenv(tokenEnv)
	if token == "" {
		return nil, "", 0, fmt.Errorf("facet %s: token env %s is empty or unset", facetName, tokenEnv)
	}
	systemPrompt, err := facet.LoadSystemPrompt(promptPath, memoryPath)
	if err != nil {
		return nil, "", 0, fmt.Errorf("facet %s: %w", facetName, err)
	}
	if timeoutSeconds == 0 {
		timeoutSeconds = config.DefaultTimeoutSeconds
	}
	// The HTTP client gets no client-level timeout: the facet bounds every
	// call with a context deadline derived from the same config field.
	client := anthropic.NewClient(token, "", 0)
	return facet.NewAnthropicEvaluator(client, model), systemPrompt, time.Duration(timeoutSeconds) * time.Second, nil
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
