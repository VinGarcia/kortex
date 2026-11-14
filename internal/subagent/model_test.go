package subagent

import "testing"

func TestResolveModel(t *testing.T) {
	for name, tc := range map[string]struct {
		requested string
		wantModel string
		wantErr   bool
	}{
		"empty defaults to sonnet": {
			requested: "",
			wantModel: "anthropic/claude-sonnet-5",
		},
		"short name haiku": {
			requested: "haiku",
			wantModel: "anthropic/claude-haiku-4-5-20251001",
		},
		"short name sonnet": {
			requested: "sonnet",
			wantModel: "anthropic/claude-sonnet-5",
		},
		"short name opus": {
			requested: "opus",
			wantModel: "anthropic/claude-opus-5-5",
		},
		"fully-qualified id matches by substring": {
			requested: "anthropic/claude-haiku-4-5-20251001",
			wantModel: "anthropic/claude-haiku-4-5-20251001",
		},
		"case-insensitive match": {
			requested: "OPUS",
			wantModel: "anthropic/claude-opus-5-5",
		},
		"fable is rejected": {
			requested: "fable",
			wantErr:   true,
		},
		"fully-qualified fable id is rejected": {
			requested: "anthropic/claude-fable-5-1",
			wantErr:   true,
		},
		"unknown model is rejected": {
			requested: "gpt-4",
			wantErr:   true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := resolveModel(tc.requested)
			if (err != nil) != tc.wantErr {
				t.Fatalf("resolveModel(%q) error = %v, wantErr %v", tc.requested, err, tc.wantErr)
			}
			if !tc.wantErr && got != tc.wantModel {
				t.Errorf("resolveModel(%q) = %q, want %q", tc.requested, got, tc.wantModel)
			}
		})
	}
}
