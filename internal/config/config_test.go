package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoad(t *testing.T) {
	tests := []struct {
		desc    string
		content string
		wantErr string
		check   func(t *testing.T, cfg Config)
	}{
		{
			desc: "full annotator config",
			content: `{
				"facets": {
					"inputAnnotator": {
						"enabled": true,
						"model": "claude-haiku-4-5-20251001",
						"tokenEnv": "CODECOMPANION_OAUTH_TOKEN",
						"systemPromptPath": "/prompts/input.md",
						"emotionalMemoryPath": "/memory/emotional-memories.md",
						"timeoutSeconds": 30
					}
				}
			}`,
			check: func(t *testing.T, cfg Config) {
				ann := cfg.Facets.InputAnnotator
				if ann == nil {
					t.Fatal("expected inputAnnotator to be present")
				}
				if !ann.Enabled || ann.Model != "claude-haiku-4-5-20251001" || ann.TokenEnv != "CODECOMPANION_OAUTH_TOKEN" ||
					ann.SystemPromptPath != "/prompts/input.md" || ann.EmotionalMemoryPath != "/memory/emotional-memories.md" ||
					ann.TimeoutSeconds != 30 {
					t.Errorf("unexpected config: %+v", ann)
				}
			},
		},
		{
			desc:    "empty facets means no annotator",
			content: `{"facets": {}}`,
			check: func(t *testing.T, cfg Config) {
				if cfg.Facets.InputAnnotator != nil {
					t.Errorf("expected nil annotator, got %+v", cfg.Facets.InputAnnotator)
				}
			},
		},
		{
			desc: "optional fields default to zero",
			content: `{"facets": {"inputAnnotator": {
				"enabled": false,
				"model": "m", "tokenEnv": "T", "systemPromptPath": "/p.md"
			}}}`,
			check: func(t *testing.T, cfg Config) {
				ann := cfg.Facets.InputAnnotator
				if ann.Enabled || ann.EmotionalMemoryPath != "" || ann.TimeoutSeconds != 0 {
					t.Errorf("unexpected defaults: %+v", ann)
				}
			},
		},
		{
			desc:    "unknown field is a loud error, not a silent ignore",
			content: `{"facets": {"inputAnnotator": {"enabld": true, "model": "m", "tokenEnv": "T", "systemPromptPath": "/p"}}}`,
			wantErr: "unknown field",
		},
		{
			desc:    "invalid json",
			content: `{not json`,
			wantErr: "decoding config",
		},
		{
			desc:    "missing model",
			content: `{"facets": {"inputAnnotator": {"enabled": true, "tokenEnv": "T", "systemPromptPath": "/p"}}}`,
			wantErr: "model is required",
		},
		{
			desc:    "missing tokenEnv",
			content: `{"facets": {"inputAnnotator": {"enabled": true, "model": "m", "systemPromptPath": "/p"}}}`,
			wantErr: "tokenEnv is required",
		},
		{
			desc:    "missing systemPromptPath",
			content: `{"facets": {"inputAnnotator": {"enabled": true, "model": "m", "tokenEnv": "T"}}}`,
			wantErr: "systemPromptPath is required",
		},
		{
			desc:    "disabled facet is still validated",
			content: `{"facets": {"inputAnnotator": {"enabled": false, "tokenEnv": "T", "systemPromptPath": "/p"}}}`,
			wantErr: "model is required",
		},
		{
			desc: "full outputEvaluator config",
			content: `{
				"facets": {
					"outputEvaluator": {
						"enabled": true,
						"model": "claude-haiku-4-5-20251001",
						"tokenEnv": "CODECOMPANION_OAUTH_TOKEN",
						"systemPromptPath": "/prompts/output.md",
						"timeoutSeconds": 30,
						"gateMinInvestment": 3
					}
				}
			}`,
			check: func(t *testing.T, cfg Config) {
				ev := cfg.Facets.OutputEvaluator
				if ev == nil {
					t.Fatal("expected outputEvaluator to be present")
				}
				if !ev.Enabled || ev.Model != "claude-haiku-4-5-20251001" || ev.TokenEnv != "CODECOMPANION_OAUTH_TOKEN" ||
					ev.SystemPromptPath != "/prompts/output.md" || ev.TimeoutSeconds != 30 ||
					ev.GateMinInvestment == nil || *ev.GateMinInvestment != 3 {
					t.Errorf("unexpected config: %+v", ev)
				}
			},
		},
		{
			desc: "outputEvaluator without gateMinInvestment leaves it nil (default applies later)",
			content: `{"facets": {"outputEvaluator": {
				"enabled": true, "model": "m", "tokenEnv": "T", "systemPromptPath": "/p.md"
			}}}`,
			check: func(t *testing.T, cfg Config) {
				if cfg.Facets.OutputEvaluator.GateMinInvestment != nil {
					t.Errorf("expected nil gateMinInvestment, got %d", *cfg.Facets.OutputEvaluator.GateMinInvestment)
				}
			},
		},
		{
			desc: "outputEvaluator explicit gateMinInvestment 0 is kept, not defaulted",
			content: `{"facets": {"outputEvaluator": {
				"enabled": true, "model": "m", "tokenEnv": "T", "systemPromptPath": "/p.md", "gateMinInvestment": 0
			}}}`,
			check: func(t *testing.T, cfg Config) {
				min := cfg.Facets.OutputEvaluator.GateMinInvestment
				if min == nil || *min != 0 {
					t.Errorf("expected explicit 0, got %v", min)
				}
			},
		},
		{
			desc:    "outputEvaluator gateMinInvestment out of range",
			content: `{"facets": {"outputEvaluator": {"enabled": true, "model": "m", "tokenEnv": "T", "systemPromptPath": "/p", "gateMinInvestment": 6}}}`,
			wantErr: "gateMinInvestment must be between 0 and 5",
		},
		{
			desc:    "outputEvaluator missing model",
			content: `{"facets": {"outputEvaluator": {"enabled": true, "tokenEnv": "T", "systemPromptPath": "/p"}}}`,
			wantErr: "facets.outputEvaluator: model is required",
		},
		{
			desc:    "negative timeout",
			content: `{"facets": {"inputAnnotator": {"enabled": true, "model": "m", "tokenEnv": "T", "systemPromptPath": "/p", "timeoutSeconds": -1}}}`,
			wantErr: "timeoutSeconds must be >= 0",
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "kortex.json")
			if err := os.WriteFile(path, []byte(test.content), 0o600); err != nil {
				t.Fatal(err)
			}
			cfg, err := Load(path)
			if test.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantErr) {
					t.Fatalf("want error containing %q, got %v", test.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			test.check(t, cfg)
		})
	}
}

func TestLoad_missingFile(t *testing.T) {
	_, err := Load(filepath.Join(t.TempDir(), "does-not-exist.json"))
	if err == nil || !strings.Contains(err.Error(), "opening config") {
		t.Fatalf("want opening error, got %v", err)
	}
}
