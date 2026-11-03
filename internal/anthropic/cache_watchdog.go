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
	// cacheAlertHistoryBreakpointDead is the cross-turn regression Vini backlog
	// #6079 adds: the message-HISTORY breakpoint (markMessagesCachePrefix,
	// cache_messages.go) has silently died — the autópsia'd 1.4M-token
	// session's failure mode, where only the system breakpoint was ever alive
	// and the whole growing history re-billed at full price every turn. Unlike
	// the two cases above, no single response can tell "breakpoint dead" apart
	// from "history not cacheable yet"; only the TREND across turns can (see
	// observeHistoryTrend).
	cacheAlertHistoryBreakpointDead cacheAlertKind = "history_breakpoint_dead_cross_turn"
)

// historyBreakpointDeadStreak is how many CONSECUTIVE sick turns (see
// observeHistoryTrend) must accumulate before cacheAlertHistoryBreakpointDead
// fires. Judgment call, most likely to need revisiting: 1 turn is never
// enough — the first-ever observation has no baseline, and even a single
// later turn can look sick for a benign reason (a cache entry's TTL expiring
// after an idle gap forces one legitimate rewrite). 3 is a deliberately
// conservative starting point — not a tuned, measured number — chosen so one
// benign rewrite can never by itself trip the alert.
const historyBreakpointDeadStreak = 3

// historyTurnState remembers one model's previous-turn usage numbers so
// observeHistoryTrend can compare turn N against turn N-1 instead of judging
// a single response in isolation. Guarded by CacheWatchdog.mu, the same lock
// that already guards lastFired. Like lastFired, it lives only in process
// memory: a kortex native process is already one-process-per-session (see
// main.go's nativeSessionID doc comment), so losing an in-flight streak on
// restart just costs one extra turn before the detector re-arms — the same
// benign, re-observed-next-call shape as every other piece of state here.
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
// this turn's usage against the SAME model's previous turn (historyTurnState)
// — the TREND observe's two single-response cases cannot see. Sick signature,
// required every turn of the streak:
//   - Stagnant read: usage.CacheReadInputTokens <= the previous turn's. Strict
//     non-growth, not a tolerance band, so a session merely growing slowly is
//     never flagged — only one that never grows at all.
//   - Large write: usage.CacheCreationInputTokens >= minCacheableTokens(model).
//     Reusing the existing per-model floor as the "large" bar (judgment call,
//     not a new magnitude) is markMessagesCachePrefix's own yardstick for "a
//     non-trivial rewrite happened". Known false-positive risk: a turn whose
//     new tail alone clears the floor (one huge tool result) looks sick on
//     this check alone — historyBreakpointDeadStreak is what absorbs that.
//
// Like observe, this never blocks/mutates/errors the request flow — it only
// ever writes a throttled WARN line via the shared fire/lastFired path.
func (w *CacheWatchdog) observeHistoryTrend(model string, system []systemBlock, usage Usage) {
	w.mu.Lock()
	state := w.historyState[model]
	// previousCacheRead is the baseline this turn is judged against; captured
	// before state.prevCacheRead is overwritten below so the WARN line (if any)
	// can report what "stagnant" was measured relative to, not just this turn's
	// own number.
	previousCacheRead := state.prevCacheRead
	sick := state.havePrev &&
		usage.CacheReadInputTokens <= previousCacheRead &&
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
	streak := state.sickStreak
	w.mu.Unlock()

	if shouldFire {
		// prev_cache_read_input_tokens and sick_streak are the two numbers this
		// alert's own verdict rests on but that the shared fire() payload has no
		// field for (it reports only THIS turn's counters) — without them a
		// reader sees case=history_breakpoint_dead_cross_turn fire with no way to
		// confirm the read count actually stalled across turns.
		w.fire(model, cacheAlertHistoryBreakpointDead, system, usage,
			fmt.Sprintf("prev_cache_read_input_tokens=%d", previousCacheRead),
			fmt.Sprintf("sick_streak=%d", streak))
	}
}

// fire emits one WARN line unless an identical (model, kind) alert fired within
// the cooldown window. The throttle check and the timestamp update happen under
// the lock so concurrent core calls cannot both slip through; the write itself
// is outside the lock. extra appends additional "key=value" fields after the
// shared payload — e.g. observeHistoryTrend uses it to carry the cross-turn
// comparison data (previous cache_read, streak length) that the shared fields
// alone cannot express; observe's two single-turn cases pass none.
func (w *CacheWatchdog) fire(model string, kind cacheAlertKind, system []systemBlock, usage Usage, extra ...string) {
	key := cacheAlertKey{model: model, kind: kind}
	now := w.now()

	w.mu.Lock()
	if last, seen := w.lastFired[key]; seen && now.Sub(last) < w.cooldown {
		w.mu.Unlock()
		return
	}
	w.lastFired[key] = now
	w.mu.Unlock()

	line := fmt.Sprintf("kortex: WARN prompt-cache degraded on core message path: "+
		"component=anthropic.cache_watchdog model=%s case=%s cache_breakpoint_set=%t "+
		"estimated_system_tokens=%d min_cacheable_tokens=%d cache_read_input_tokens=%d "+
		"cache_creation_input_tokens=%d input_tokens=%d",
		model, kind, systemHasBreakpoint(system), estimatedSystemTokens(system),
		minCacheableTokens(model), usage.CacheReadInputTokens,
		usage.CacheCreationInputTokens, usage.InputTokens)
	for _, field := range extra {
		line += " " + field
	}
	fmt.Fprintln(w.out, line)
}

// reportDefect emits one `kortex:`-prefixed line when the watchdog itself
// panicked. A fail-LOUD monitor must not fail silently on its own bug: the
// Client's recover calls this so the defect is visible, while still keeping the
// fail-safe half intact (the panic never propagates and the request succeeds).
func (w *CacheWatchdog) reportDefect(r any) {
	fmt.Fprintf(w.out, "kortex: WARN cache-watchdog defect (observation skipped, request unaffected): %v\n", r)
}
