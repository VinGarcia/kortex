package anthropic

import "strings"

// modelTraits is the single place where this client knows that one Claude
// family differs from another on the wire. Every per-model rule lives here as a
// field, and CreateMessage/the cache code only ever consult traitsFor(model) —
// never the model string itself — so adding a model (or a new quirk) is one row,
// not a new branch scattered through the request path.
type modelTraits struct {
	// MinCacheableTokens is the smallest prefix the Messages API will actually
	// cache. A cache_control breakpoint below it is ignored silently, so marking
	// under this floor is pointless.
	MinCacheableTokens int
	// SupportsEffort reports whether the model accepts output_config.effort.
	// Sending it to a model that does not fails the whole call with
	// `400 invalid_request_error: This model does not support the effort
	// parameter` (observed live 2026-11-13 on claude-haiku-4-5-20251001).
	SupportsEffort bool
}

// defaultTraits describes every current non-haiku family this client targets
// (opus-4-8/5-5, sonnet, fable-5/5-1): 1024-token cache floor and effort on.
var defaultTraits = modelTraits{MinCacheableTokens: 1024, SupportsEffort: true}

// familyTraits overrides defaultTraits for a family, matched by substring of the
// API model id. Haiku 4.5 has a 4096-token cache floor and no effort knob.
var familyTraits = map[string]modelTraits{
	"haiku": {MinCacheableTokens: 4096, SupportsEffort: false},
}

// traitsFor resolves the traits of an API model id. Unknown families get
// defaultTraits, which is the behavior every call had before this table existed.
func traitsFor(model string) modelTraits {
	for family, traits := range familyTraits {
		if strings.Contains(model, family) {
			return traits
		}
	}
	return defaultTraits
}

// minCacheableTokens is kept as the name the cache code calls; it is a view on
// traitsFor so there is still exactly one source for the floor.
func minCacheableTokens(model string) int {
	return traitsFor(model).MinCacheableTokens
}
