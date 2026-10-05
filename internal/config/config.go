// Package config loads the kortex facet configuration file. The file is
// pointed at by the KORTEX_CONFIG env var (read in main, like every env);
// when the env is absent no config exists and kortex stays a pure
// passthrough — the file's presence IS the feature flag for facets.
//
// Format: JSON. Chosen over TOML/YAML because the stdlib decodes it with
// zero new dependencies, the file is small and machine-authored, and
// DisallowUnknownFields gives typo detection (a misspelled key in a feature
// flag must fail loudly, not silently disable a facet).
package config

import (
	"encoding/json"
	"fmt"
	"os"
)

// DefaultTimeoutSeconds bounds a facet's model call when the file does not
// set timeoutSeconds. Facets fail open past it, so it only caps added
// latency, never correctness. The facet call runs inline on the stdin pump
// (later stdin lines wait behind it), so the default leans low; a config
// that prefers annotation coverage over latency can raise it.
const DefaultTimeoutSeconds = 10

// DefaultSuperegoTimeoutSeconds bounds the superego call when the file does
// not set timeoutSeconds. Much higher than DefaultTimeoutSeconds on purpose:
// the superego reads the whole conversation and writes a prose critique, and
// it runs async off the stream (shadow mode), so the timeout caps cost, not
// latency.
const DefaultSuperegoTimeoutSeconds = 60

// DefaultSuperegoModel is the superego model when the config leaves model
// empty. The final model choice at activation time belongs to the operator;
// this default only keeps the config minimal in the meantime.
const DefaultSuperegoModel = "claude-opus-4-8"

// DefaultGateMinInvestment is the gate threshold when the config does not
// set gateMinInvestment. Source: the primary path of sylphie's
// core/emotion-output-gate-prompt.md — "investment ≥ 4; ou investment = 3
// com valence mista ou negativa" — i.e. today's gate fires at 4, and at 4-1
// when the valence is negative or mixed.
const DefaultGateMinInvestment = 4

// DefaultRedraftToolBudget is the per-redraft tool-use round budget when the
// config leaves redraftToolBudget unset (0). A superego-triggered redraft in
// active mode may run the tool loop to ACT on the critique (e.g. actually run
// the verification the superego flagged as missing) instead of merely rewording;
// this budget caps how many tool-use round trips that single redraft may make
// before it must finalize with a text answer. It is deliberately small: a
// redraft should resolve the specific finding, not open an unbounded tool loop.
const DefaultRedraftToolBudget = 3

// SuperegoModeShadow (the default) runs the superego asynchronously after the
// turn is delivered: observe-only, log-only. SuperegoModeActive runs it
// synchronously before delivery, inside the blocking core↔superego loop.
const (
	SuperegoModeShadow = "shadow"
	SuperegoModeActive = "active"
)

// Config is the root of the kortex config file.
type Config struct {
	Facets Facets `json:"facets"`
}

// Facets declares which facets exist and how each is wired. A facet absent
// from the file (nil) is off.
type Facets struct {
	InputAnnotator  *InputAnnotator  `json:"inputAnnotator"`
	OutputEvaluator *OutputEvaluator `json:"outputEvaluator"`
	Superego        *Superego        `json:"superego"`
}

// InputAnnotator configures the emotional annotator that runs over real
// user messages on their way to the core model.
type InputAnnotator struct {
	Enabled bool `json:"enabled"`
	// Model is the evaluator model id (e.g. claude-haiku-4-5-20251001).
	Model string `json:"model"`
	// TokenEnv names the env var holding the OAuth token. The binary reads
	// the env at startup; the token value itself never lives in this file.
	TokenEnv string `json:"tokenEnv"`
	// SystemPromptPath is the evaluator prompt file. Only the body after the
	// first line that is exactly "---" is sent to the model (the header is a
	// maintenance note), matching the sylphie prompt-file convention.
	SystemPromptPath string `json:"systemPromptPath"`
	// EmotionalMemoryPath, when set, is a file injected into the evaluator's
	// system prompt as trusted context before the untrusted message.
	EmotionalMemoryPath string `json:"emotionalMemoryPath"`
	// TimeoutSeconds bounds each evaluator call; 0 means
	// DefaultTimeoutSeconds.
	TimeoutSeconds int `json:"timeoutSeconds"`
}

// OutputEvaluator configures the F2c emotional evaluator that runs over
// each completed assistant turn, in observability mode: the stream is never
// touched; tags + gate aggregate + the hypothetical gate decision go to the
// facet log and to in-memory turn metadata. The InputAnnotator and this
// struct stay separate types on purpose: they configure different facets
// that diverge (gateMinInvestment exists only here) even though most fields
// coincide today.
type OutputEvaluator struct {
	Enabled bool `json:"enabled"`
	// Model is the evaluator model id (e.g. claude-haiku-4-5-20251001).
	Model string `json:"model"`
	// TokenEnv names the env var holding the OAuth token. The binary reads
	// the env at startup; the token value itself never lives in this file.
	TokenEnv string `json:"tokenEnv"`
	// SystemPromptPath is the evaluator prompt file. Only the body after the
	// first line that is exactly "---" is sent to the model (the header is a
	// maintenance note), matching the sylphie prompt-file convention.
	SystemPromptPath string `json:"systemPromptPath"`
	// EmotionalMemoryPath, when set, is a file injected into the evaluator's
	// system prompt as trusted context before the untrusted text.
	EmotionalMemoryPath string `json:"emotionalMemoryPath"`
	// TimeoutSeconds bounds each evaluator call; 0 means
	// DefaultTimeoutSeconds.
	TimeoutSeconds int `json:"timeoutSeconds"`
	// GateMinInvestment is the aggregate investment at which the
	// hypothetical gate fires (it also fires at GateMinInvestment-1 when the
	// valence is negativa or mista, mirroring the gate prompt). A pointer so
	// an explicit 0 ("fire on everything") stays distinguishable from
	// absent; nil means DefaultGateMinInvestment.
	GateMinInvestment *int `json:"gateMinInvestment"`
}

// Superego configures the F2d facet in SHADOW MODE: when the output
// evaluator's hypothetical gate fires on a completed turn, the superego
// reviews that turn's text against the whole conversation; the critique goes
// only to the facet log and to in-memory per-turn metadata (the structure
// the F2e block→revise loop will consume). Nothing ever touches the stream.
type Superego struct {
	Enabled bool `json:"enabled"`
	// Mode selects how the superego acts on a gate hit. "shadow" (the default
	// when empty) is observe-only: the review runs asynchronously after the
	// turn's result is already delivered and only annotates the facet log.
	// "active" is the blocking loop: the review runs synchronously before the
	// turn is delivered and can revise the draft (core redrafts from the
	// critique) or hold it for the human. Any active-mode failure falls open to
	// delivering the original draft, so the mode never holds a reply hostage.
	Mode string `json:"mode"`
	// Model is the superego model id; empty means DefaultSuperegoModel. The
	// definitive model is an activation-time choice.
	Model string `json:"model"`
	// TokenEnv names the env var holding the OAuth token. The binary reads
	// the env at startup; the token value itself never lives in this file.
	TokenEnv string `json:"tokenEnv"`
	// SystemPromptPath is the superego prompt file. Only the body after the
	// first line that is exactly "---" is sent to the model (the header is a
	// maintenance note), matching the sylphie prompt-file convention.
	SystemPromptPath string `json:"systemPromptPath"`
	// TimeoutSeconds bounds each superego call; 0 means
	// DefaultSuperegoTimeoutSeconds.
	TimeoutSeconds int `json:"timeoutSeconds"`
	// MaxHistoryTurns caps how many of the most recent conversation turns
	// (the reviewed turn included) enter the superego's context; 0 means the
	// whole history. A cap trades review depth for token cost when a session
	// grows long.
	MaxHistoryTurns int `json:"maxHistoryTurns"`
	// RedraftToolBudget caps how many tool-use round trips a single active-mode
	// redraft may make before it must answer with text. It only matters in
	// mode "active": when the superego flags a finding that needs ACTION (for
	// example "you did not actually verify X"), the redraft drives the same
	// tool loop the turn uses so the core can run the check and answer with
	// evidence, instead of rewording the problem away. 0 means
	// DefaultRedraftToolBudget. Keep it small: the redraft should resolve the
	// specific finding, never open an unbounded tool loop.
	RedraftToolBudget int `json:"redraftToolBudget"`
}

// Load reads and validates the config file at path.
func Load(path string) (Config, error) {
	file, err := os.Open(path)
	if err != nil {
		return Config{}, fmt.Errorf("opening config: %w", err)
	}
	defer file.Close()

	var cfg Config
	dec := json.NewDecoder(file)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cfg); err != nil {
		return Config{}, fmt.Errorf("decoding config %s: %w", path, err)
	}
	if err := cfg.Validate(); err != nil {
		return Config{}, fmt.Errorf("invalid config %s: %w", path, err)
	}
	return cfg, nil
}

// Validate checks the declared facets are complete enough to construct.
func (c Config) Validate() error {
	if ann := c.Facets.InputAnnotator; ann != nil {
		if err := ann.Validate(); err != nil {
			return fmt.Errorf("facets.inputAnnotator: %w", err)
		}
	}
	if ev := c.Facets.OutputEvaluator; ev != nil {
		if err := ev.Validate(); err != nil {
			return fmt.Errorf("facets.outputEvaluator: %w", err)
		}
	}
	if se := c.Facets.Superego; se != nil {
		if err := se.Validate(); err != nil {
			return fmt.Errorf("facets.superego: %w", err)
		}
		// The superego only ever runs when the output evaluator's gate
		// fires, so enabling it without the evaluator silently disables it —
		// the kind of misconfiguration that must fail loudly at startup.
		if se.Enabled && (c.Facets.OutputEvaluator == nil || !c.Facets.OutputEvaluator.Enabled) {
			return fmt.Errorf("facets.superego: enabled but facets.outputEvaluator is absent or disabled (the superego is triggered by its gate)")
		}
	}
	return nil
}

// Validate checks the fields the facet cannot run without. A disabled facet
// is still validated: a config someone will later flip to enabled should be
// complete from day one.
func (a InputAnnotator) Validate() error {
	if a.Model == "" {
		return fmt.Errorf("model is required")
	}
	if a.TokenEnv == "" {
		return fmt.Errorf("tokenEnv is required")
	}
	if a.SystemPromptPath == "" {
		return fmt.Errorf("systemPromptPath is required")
	}
	if a.TimeoutSeconds < 0 {
		return fmt.Errorf("timeoutSeconds must be >= 0, got %d", a.TimeoutSeconds)
	}
	return nil
}

// Validate checks the fields the facet cannot run without. A disabled facet
// is still validated: a config someone will later flip to enabled should be
// complete from day one.
func (e OutputEvaluator) Validate() error {
	if e.Model == "" {
		return fmt.Errorf("model is required")
	}
	if e.TokenEnv == "" {
		return fmt.Errorf("tokenEnv is required")
	}
	if e.SystemPromptPath == "" {
		return fmt.Errorf("systemPromptPath is required")
	}
	if e.TimeoutSeconds < 0 {
		return fmt.Errorf("timeoutSeconds must be >= 0, got %d", e.TimeoutSeconds)
	}
	if e.GateMinInvestment != nil && (*e.GateMinInvestment < 0 || *e.GateMinInvestment > 5) {
		return fmt.Errorf("gateMinInvestment must be between 0 and 5, got %d", *e.GateMinInvestment)
	}
	return nil
}

// Validate checks the fields the facet cannot run without. A disabled facet
// is still validated: a config someone will later flip to enabled should be
// complete from day one. Model stays optional here (DefaultSuperegoModel).
func (s Superego) Validate() error {
	if s.TokenEnv == "" {
		return fmt.Errorf("tokenEnv is required")
	}
	if s.SystemPromptPath == "" {
		return fmt.Errorf("systemPromptPath is required")
	}
	if s.TimeoutSeconds < 0 {
		return fmt.Errorf("timeoutSeconds must be >= 0, got %d", s.TimeoutSeconds)
	}
	if s.MaxHistoryTurns < 0 {
		return fmt.Errorf("maxHistoryTurns must be >= 0, got %d", s.MaxHistoryTurns)
	}
	if s.RedraftToolBudget < 0 {
		return fmt.Errorf("redraftToolBudget must be >= 0, got %d", s.RedraftToolBudget)
	}
	if s.Mode != "" && s.Mode != SuperegoModeShadow && s.Mode != SuperegoModeActive {
		return fmt.Errorf("mode must be %q or %q, got %q", SuperegoModeShadow, SuperegoModeActive, s.Mode)
	}
	return nil
}
