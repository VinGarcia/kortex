package anthropic

import (
	"fmt"
	"io"
	"sync"
	"time"
)

// defaultCacheAlertCooldown is the minimum gap between two identical cache
// alerts (same model, same case). It stops a persistently-degraded session from
// emitting one WARN per turn forever while still re-surfacing a problem that
// outlives the window. NewCacheWatchdog uses it when the caller passes a
// non-positive cooldown.
const defaultCacheAlertCooldown = 5 * time.Minute

// cacheAlertKind names which degradation fired, and doubles as the throttle
// dimension together with the model: case (a) and case (b) on the same model
// are tracked and rate-limited independently.
type cacheAlertKind string

const (
	// cacheAlertNoCaching is F4 case (a): the request DID go out with a cache
	// breakpoint, but the response reported neither a cache write nor a cache
	// read (cache_creation_input_tokens + cache_read_input_tokens == 0) — the
	// breakpoint was present yet no caching happened.
	cacheAlertNoCaching cacheAlertKind = "breakpoint_set_but_no_caching"
	// cacheAlertBornBlind is F4 case (b): the request went out with NO cache
	// breakpoint at all even though the system prefix was large enough to be
	// cacheable — the "born blind" regression the pre-037d2951 native client
	// shipped, where every turn re-billed the whole context.
	cacheAlertBornBlind cacheAlertKind = "no_breakpoint_despite_eligible_prefix"
)

// cacheAlertKey is the throttle key: a given (model, kind) pair fires at most
// once per cooldown window.
type cacheAlertKey struct {
	model string
	kind  cacheAlertKind
}

// CacheWatchdog makes silent prompt-cache cost degradation LOUD. On the CORE
// message path, every turn is supposed to read its stable system prefix back
// from the prompt cache; when that silently stops (the API ignores a breakpoint
// below the per-model floor, or the client never set one at all), nothing errors
// — the session just re-bills the whole context every turn. The watchdog
// inspects each successful response and emits a throttled WARN when either
// failure mode is detected.
//
// The alert is a single `kortex:`-prefixed line written to the injected writer
// (typically the native session's stderr), matching the diagnostic idiom used
// across main/proxy/session so a degradation is greppable alongside every other
// kortex diagnostic — the key=value payload travels on that one line.
//
// It is an observer only. It never blocks, mutates, or errors the
// request/response flow (the Client calls it under a recover), and it never
// touches what gets cached — it reads the already-built request and the decoded
// usage, nothing more.
type CacheWatchdog struct {
	out      io.Writer
	cooldown time.Duration
	// now is the clock, injectable so throttle tests are deterministic.
	// NewCacheWatchdog always sets it to time.Now.
	now func() time.Time

	mu        sync.Mutex
	lastFired map[cacheAlertKey]time.Time
}

// NewCacheWatchdog builds a watchdog that writes `kortex:`-prefixed WARN lines
// to w (typically the native session's stderr/diagnostic stream). cooldown
// bounds how often an identical (model, case) alert repeats; a non-positive
// value uses defaultCacheAlertCooldown. A healthy session writes nothing — only
// degradation surfaces.
func NewCacheWatchdog(w io.Writer, cooldown time.Duration) *CacheWatchdog {
	if cooldown <= 0 {
		cooldown = defaultCacheAlertCooldown
	}
	return &CacheWatchdog{
		out:       w,
		cooldown:  cooldown,
		now:       time.Now,
		lastFired: map[cacheAlertKey]time.Time{},
	}
}

// observe classifies one successful response and fires the matching alert, if
// any. system is the exact wire block array (so the last block's cache_control
// is the ground truth of whether a breakpoint went out); usage is the decoded
// response usage; model selects the per-model cacheable floor via the shared
// minCacheableTokens source of truth (never a second copy).
//
// The two healthy / benign states are deliberately silent:
//   - breakpoint set AND caching happened (cache_creation or cache_read > 0):
//     the intended steady state — no alert.
//   - no breakpoint AND the prefix was below the floor (or empty): the fail-safe
//     markStableSystemPrefix path, exactly what the API would do anyway — no
//     alert, so the below-floor/empty case never cries wolf.
func (w *CacheWatchdog) observe(model string, system []systemBlock, usage Usage) {
	marked := systemHasBreakpoint(system)
	cachedTokens := usage.CacheCreationInputTokens + usage.CacheReadInputTokens
	eligible := len(system) > 0 && estimatedSystemTokens(system) >= minCacheableTokens(model)

	switch {
	case marked && cachedTokens == 0:
		// Case (a): the breakpoint went out but the API neither wrote nor read the
		// cache. A cache write (cache_creation_input_tokens > 0) is expected even on
		// a session's first turn, so both counters at zero despite a breakpoint is a
		// genuine failure, not a cold cache. (A near-floor edge exists: the ~4-byte/
		// token estimate can clear the model floor while the API's true token count
		// falls below it, so the API ignores the breakpoint; that surfaces here too,
		// and reads correctly as a real "this prefix is too small to cache" signal.)
		w.fire(model, cacheAlertNoCaching, system, usage)
	case !marked && eligible:
		// Case (b): the prefix was large enough to cache but no breakpoint went out
		// at all — the born-blind regression. NOTE: this branch is unreachable from
		// the only caller today. CreateMessage always runs markStableSystemPrefix,
		// which marks a breakpoint iff the SAME estimatedSystemTokens >=
		// minCacheableTokens predicate (on the same slice) that `eligible` tests —
		// so on the current path `eligible` already implies `marked`. It is kept
		// deliberately as a REGRESSION TRIPWIRE: a future core-request path that
		// builds a request without calling markStableSystemPrefix would be caught
		// here instead of silently re-billing the whole context every turn.
		w.fire(model, cacheAlertBornBlind, system, usage)
	}
}

// fire emits one WARN line unless an identical (model, kind) alert fired within
// the cooldown window. The throttle check and the timestamp update happen under
// the lock so concurrent core calls cannot both slip through; the write itself
// is outside the lock.
func (w *CacheWatchdog) fire(model string, kind cacheAlertKind, system []systemBlock, usage Usage) {
	key := cacheAlertKey{model: model, kind: kind}
	now := w.now()

	w.mu.Lock()
	if last, seen := w.lastFired[key]; seen && now.Sub(last) < w.cooldown {
		w.mu.Unlock()
		return
	}
	w.lastFired[key] = now
	w.mu.Unlock()

	fmt.Fprintf(w.out, "kortex: WARN prompt-cache degraded on core message path: "+
		"component=anthropic.cache_watchdog model=%s case=%s cache_breakpoint_set=%t "+
		"estimated_system_tokens=%d min_cacheable_tokens=%d cache_read_input_tokens=%d "+
		"cache_creation_input_tokens=%d input_tokens=%d\n",
		model, kind, systemHasBreakpoint(system), estimatedSystemTokens(system),
		minCacheableTokens(model), usage.CacheReadInputTokens,
		usage.CacheCreationInputTokens, usage.InputTokens)
}

// reportDefect emits one `kortex:`-prefixed line when the watchdog itself
// panicked. A fail-LOUD monitor must not fail silently on its own bug: the
// Client's recover calls this so the defect is visible, while still keeping the
// fail-safe half intact (the panic never propagates and the request succeeds).
func (w *CacheWatchdog) reportDefect(r any) {
	fmt.Fprintf(w.out, "kortex: WARN cache-watchdog defect (observation skipped, request unaffected): %v\n", r)
}
