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
	annotator, facetLogFile, err := buildAnnotator(os.Getenv("KORTEX_CONFIG"), logPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "kortex: %v\n", err)
		os.Exit(1)
	}
	var interceptor proxy.Interceptor
	if annotator != nil {
		interceptor = annotator
	}

	// os.Exit skips defers, so the facet log is closed explicitly after Run.
	code := proxy.Run(proxy.Config{
		ClaudeBin:    claudeBin,
		Argv:         os.Args[1:],
		AuthFDs:      authFDsFromEnv(),
		LogPath:      logPath,
		EventLogPath: eventLogPath,
		Interceptor:  interceptor,
		Stdin:        os.Stdin,
		Stdout:       os.Stdout,
		Stderr:       os.Stderr,
	})
	if facetLogFile != nil {
		facetLogFile.Close()
	}
	os.Exit(code)
}

// buildAnnotator loads the config at cfgPath and constructs the input
// annotator facet, plus the facet-log file it writes to (nil when logging
// is off; the caller owns closing it). (nil, nil, nil) when cfgPath is
// empty or the facet is absent or disabled. All env reading stays here in
// the composition root: the facet receives the token, never the env name.
func buildAnnotator(cfgPath string, logPath string) (*facet.InputAnnotator, *os.File, error) {
	if cfgPath == "" {
		return nil, nil, nil
	}
	cfg, err := config.Load(cfgPath)
	if err != nil {
		return nil, nil, err
	}
	annCfg := cfg.Facets.InputAnnotator
	if annCfg == nil || !annCfg.Enabled {
		return nil, nil, nil
	}

	token := os.Getenv(annCfg.TokenEnv)
	if token == "" {
		return nil, nil, fmt.Errorf("facet inputAnnotator: token env %s is empty or unset", annCfg.TokenEnv)
	}
	systemPrompt, err := facet.LoadSystemPrompt(annCfg.SystemPromptPath, annCfg.EmotionalMemoryPath)
	if err != nil {
		return nil, nil, fmt.Errorf("facet inputAnnotator: %w", err)
	}
	timeoutSeconds := annCfg.TimeoutSeconds
	if timeoutSeconds == 0 {
		timeoutSeconds = config.DefaultTimeoutSeconds
	}
	timeout := time.Duration(timeoutSeconds) * time.Second

	// The structured facet-call log lives next to the other KORTEX_LOG
	// artifacts. An open failure degrades to no facet log (stderr warning),
	// never to no facet.
	var facetLogFile *os.File
	var logSink io.Writer
	if logPath != "" {
		facetLogFile, err = os.OpenFile(logPath+".facets", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
		if err != nil {
			fmt.Fprintf(os.Stderr, "kortex: facet log disabled: %v\n", err)
		} else {
			logSink = facetLogFile
		}
	}

	// The HTTP client gets no client-level timeout: the facet bounds every
	// call with a context deadline derived from the same config field.
	client := anthropic.NewClient(token, "", 0)
	return facet.NewInputAnnotator(facet.AnnotatorParams{
		Evaluator:    facet.NewAnthropicEvaluator(client, annCfg.Model),
		SystemPrompt: systemPrompt,
		Timeout:      timeout,
		Model:        annCfg.Model,
		LogSink:      logSink,
	}), facetLogFile, nil
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
