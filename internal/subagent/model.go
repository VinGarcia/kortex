package subagent

import (
	"fmt"
	"strings"
)

// defaultModelFamily is the family resolveModel uses when spawn_subagent's
// "model" input is omitted.
const defaultModelFamily = "sonnet"

// modelIDs maps a subagent-allowed family name to the full Anthropic model id
// the gateway's sessions_spawn expects.
var modelIDs = map[string]string{
	"haiku":  "anthropic/claude-haiku-4-5-20251001",
	"sonnet": "anthropic/claude-sonnet-5",
	"opus":   "anthropic/claude-opus-5-5",
}

// subagentModelFamilies is the allowlist, in the priority order
// resolveModel matches them.
var subagentModelFamilies = []string{"haiku", "sonnet", "opus"}

// resolveModel maps requested (spawn_subagent's "model" input, possibly
// empty) to the gateway model id to send, or an error safe to show the model.
// Matching is by family SUBSTRING, not exact equality, so a caller may pass
// either the short alias ("haiku") or an already-qualified id that contains
// it ("anthropic/claude-haiku-4-5-20251001") and get the same result. fable
// is rejected outright regardless of substring match: spawning a subagent on
// fable is Vini's standing rule (2026-10-20) — fable is reserved for direct
// conversation, and a fan-out of subagents on it burns the token budget fast
// — so it is enforced here rather than left to the caller to remember.
func resolveModel(requested string) (model string, err error) {
	family := requested
	if family == "" {
		family = defaultModelFamily
	}
	lower := strings.ToLower(family)
	if strings.Contains(lower, "fable") {
		return "", fmt.Errorf("model %q is not allowed for subagents: fable is reserved for direct conversation with the user, never as a spawned executor", requested)
	}
	for _, name := range subagentModelFamilies {
		if strings.Contains(lower, name) {
			return modelIDs[name], nil
		}
	}
	return "", fmt.Errorf("model %q is not in the subagent allowlist (%s); omit it to default to %s", requested, strings.Join(subagentModelFamilies, ", "), defaultModelFamily)
}
