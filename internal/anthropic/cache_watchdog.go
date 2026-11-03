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
// dimension together with the model: case (a), case (b), and the cross-turn
// history-breakpoint-dead case on the same model are tracked and
// rate-limited independently.
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
	// cacheAlertHistoryBreakpointDead is the cross-turn regression this file's
	// Vini backlog #6079 adds: the message-HISTORY breakpoint (see
	// markMessagesCachePrefix, cache_messages.go) has silently died. Unlike the
	// two cases above, this is not visible on one turn in isolation — a single
	// response's usage numbers cannot tell "history breakpoint dead" apart from
	// "history just isn't big enough to cache yet". It only shows up by
	// comparing usage ACROSS several turns of the same session (see
	// observeHistoryTrend): cache_read_input_tokens stops growing while
	// cache_creation_input_tokens stays large every turn, i.e. the whole
	// history is being re-written from scratch instead of the system prefix
	// (and, when healthy, the cached history) being read back. This is exactly
	// the autópsia'd 1.4M-token session's failure mode: only the system
	// breakpoint was ever alive, so every turn re-billed the entire growing
	// history at full price.
	cacheAlertHistoryBreakpointDead cacheAlertKind = "history_breakpoint_dead_cross_turn"
)

// historyBreakpointDeadStreak is how many CONSECUTIVE turns must show the
// sick cross-turn signature (see observeHistoryTrend) before
// cacheAlertHistoryBreakpointDead fires. This is the judgment call most
// likely to need revisiting:
//   - 1 turn is never enough by construction — a session's first-ever
//     observation has no prior cache_read to compare against, and a single
//     later turn can look sick for entirely benign reasons: a prompt-cache
//     entry has a short TTL, so a turn that arrives after an idle gap longer
//     than the TTL legitimately has to rewrite history it would otherwise
//     have read back, producing exactly one stagnant-read/large-write turn
//     with no underlying regression.
//   - 3 is chosen so that one such benign rewrite (or one early turn whose
//     history is still below the cacheable floor) can never by itself trip
//     the alert; three turns in a row with zero read growth, each separately
//     clearing the "large write" bar, is past the point any single benign
//     explanation stays plausible. There is no measured false-positive rate
//     behind this number — it is a deliberately conservative starting point,
//     not a tuned threshold.
const historyBreakpointDeadStreak = 3

// historyTurnState remembers one model's previous-turn usage numbers so
// observeHistoryTrend can compare turn N against turn N-1 instead of judging
// a single response in isolation. Guarded by CacheWatchdog.mu, the same lock
// that already guards lastFired.
//
// This lives only in process memory, like lastFired above it: it is a
// diagnostic streak counter, not state the product needs correct across a
// restart (see package u-no-durable-state-in-memory's "pods are ephemeral"
// reasoning for WHY that rule exists) — a kortex native process is already
// one-process-per-session (see main.go's nativeSessionID doc comment), so a
// restart starts a new session anyway, and losing an in-flight streak on
// restart just costs one extra turn before the detector re-arms, which is the
// same "benign, re-observed next call" shape as every other piece of state in
// this file.
type historyTurnState struct {
	// havePrev is false until this model's first turn is observed; the first
	// turn is never judged sick (see observeHistoryTrend).
	havePrev      bool
	prevCacheRead int
	// sickStreak counts consecutive turns (after the first) that matched the
	// sick signature; it resets to 0 the moment a turn breaks the pattern.
	sickStreak int
}

// cacheAlertKey is the throttle key: a given (model, kind) pair fires at most
// once per cooldown window.
type cacheAlertKey struct {
	model string
	kind  cacheAlertKind
}

// CacheWatchdog makes silent prompt-cache cost degradation LOUD. On the CORE
// message path, every turn is supposed to read its stable system prefix AND
// its growing message history back from the prompt cache (two breakpoints —
// see markStableSystemPrefix and markMessagesCachePrefix); when either
// silently stops (the API ignores a breakpoint below the per-model floor, the
// client never set one at all, or a cache entry simply never gets hit again),
// nothing errors — the session just re-bills that part of the context every
// turn. The watchdog inspects each successful response and emits a throttled
// WARN when a failure mode is detected: observe's two cases judge the system
// prefix from a single response, while observeHistoryTrend judges the
// history breakpoint from the TREND across several responses, since that one
// is only visible cross-turn (see its own doc comment).
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
	// historyState is the cross-turn bookkeeping for observeHistoryTrend,
	// keyed by model exactly like cacheAlertKey's model dimension — the
	// watchdog is wired once per native process, and a native process is one
	// session (see main.go's nativeSessionID doc comment), so "per model" and
	// "per core conversation" coincide here; there is no separate per-session
	// key to carry.
	historyState map[string]historyTurnState
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
		out:          w,
		cooldown:     cooldown,
		now:          time.Now,
		lastFired:    map[cacheAlertKey]time.Time{},
		historyState: map[string]historyTurnState{},
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
//
// Both cases above judge ONE response in isolation. observeHistoryTrend, called
// at the end, is the cross-turn counterpart that the message-HISTORY breakpoint
// needs instead (see its own doc comment for why a single turn cannot tell
// "history breakpoint dead" apart from "history not cacheable yet").
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

	w.observeHistoryTrend(model, system, usage)
}

// observeHistoryTrend detects a dead message-HISTORY breakpoint by comparing
// this turn's usage against the SAME model's previous turn (see
// historyTurnState) — something observe's two cases above cannot do, because
// they only ever see one response at a time. A single turn's
// cache_read/cache_creation pair cannot distinguish "the history breakpoint
// died" from "the history just isn't cacheable yet"; only the TREND across
// several turns can.
//
// The sick signature, and why each piece of the predicate is there:
//   - Stagnant read: usage.CacheReadInputTokens <= the previous turn's. A
//     healthy session's read count is expected to be non-decreasing as more
//     history becomes eligible to be read back (system prefix, then also the
//     growing cached history once the fix in 7065f8d/4be22b9 is doing its
//     job); strict non-growth — not a tolerance band — means a session that
//     is merely growing slowly is never flagged, only one that never grows
//     across several turns in a row.
//   - Large write: usage.CacheCreationInputTokens >= minCacheableTokens(model).
//     Reusing the existing per-model cacheable floor as the bar for "large"
//     (rather than inventing a new magnitude) is a judgment call: it is the
//     same floor markMessagesCachePrefix itself uses to decide a breakpoint is
//     even worth placing, so a write at or above it is, by this client's own
//     yardstick, a non-trivial rewrite — consistent with "the whole history
//     got re-billed", not a small incremental tail. The known false-positive
//     risk this accepts: a turn whose NEW tail content alone clears the floor
//     (e.g. one huge tool result) looks sick by this test alone; the streak
//     requirement below is what keeps that from mattering.
//   - historyBreakpointDeadStreak consecutive turns matching BOTH of the above
//     — see that constant's doc comment for why 3, and for the single-turn
//     benign explanations (TTL-expiry rewrite, early below-floor turns) this
//     guards against.
//
// Like observe, this never blocks/mutates/errors the request flow — it only
// ever writes a throttled WARN line via the shared fire/lastFired path, so
// cacheAlertHistoryBreakpointDead rate-limits independently of the two
// system-prefix cases on the same model.
func (w *CacheWatchdog) observeHistoryTrend(model string, system []systemBlock, usage Usage) {
	w.mu.Lock()
	state := w.historyState[model]
	sick := state.havePrev &&
		usage.CacheReadInputTokens <= state.prevCacheRead &&
		usage.CacheCreationInputTokens >= minCacheableTokens(model)
	if sick {
		state.sickStreak++
	} else {
		state.sickStreak = 0
	}
	state.havePrev = true
	state.prevCacheRead = usage.CacheReadInputTokens
	w.historyState[model] = state
	shouldFire := state.sickStreak >= historyBreakpointDeadStreak
	w.mu.Unlock()

	if shouldFire {
		w.fire(model, cacheAlertHistoryBreakpointDead, system, usage)
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
