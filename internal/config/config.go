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

// Config is the root of the kortex config file.
type Config struct {
	Facets Facets `json:"facets"`
}

// Facets declares which facets exist and how each is wired. A facet absent
// from the file (nil) is off.
type Facets struct {
	InputAnnotator *InputAnnotator `json:"inputAnnotator"`
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
