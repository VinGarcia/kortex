package anthropic

// MarkMessagesCachePrefix returns messages with one ephemeral cache_control
// breakpoint added to the last eligible content block of the LAST message, so
// a follow-up call whose prefix up to that point is byte-identical reads the
// entire conversation history back from cache (~0.1x price) instead of
// re-billing it in full every turn. markStableSystemPrefix already caches
// tools+system; this closes the gap on the growing message history.
//
// One moving breakpoint is enough for incremental caching across a whole
// tool-loop session: each round trip appends at most two messages before the
// next call, so the new breakpoint always lands inside the Messages API's
// 20-position cache lookback window and finds the previous call's entry.
// Combined with the system breakpoint, this uses 2 of the API's 4-breakpoint
// ceiling.
//
// It never mutates messages or anything inside it — every level it touches is
// a fresh copy, so the caller's original slice is untouched. It is fail-safe
// like markStableSystemPrefix: when there is nothing eligible to mark, it
// returns messages unchanged and the request proceeds uncached.
//
// Call this ONLY with canonical conversation history — never with a slice
// that already has a volatile tail appended (e.g. a per-turn tone digest), or
// every call writes a fresh, never-read cache entry for the whole
// conversation instead of reusing the last one. A caller that appends such a
// tail must call this on the real history first, then append the tail — see
// tools.withEphemeralContext, which enforces that order in one place.
//
// Wrapping a string Message.Content into a []messageContentBlock (below)
// changes its JSON shape for this call only; callers never persist the
// wrapped form, so the same message goes out as a plain string again on the
// next turn. This assumes the API normalizes both shapes to the same
// cache-prefix hash; if it doesn't, the worst case is benign — the lookback
// window still finds the previous turn's cache entry. Confirm against the
// usage counters on the first live session.
//
// Observability gap (known, accepted): unlike markStableSystemPrefix, this
// breakpoint is not covered by CacheWatchdog. The failure signature is
// already known: the system prefix has a fixed size, so cache_read staying
// pinned near that size while cache_creation keeps paying for the full
// history on every request is the sick pattern this breakpoint exists to
// avoid. What's deferred is cost, not knowledge — baselining
// cache_creation/cache_read across turns to apply that signal automatically
// is new monitoring surface, not a small addition to the existing
// per-request classifier — left as a follow-up rather than folded into this
// change.
func MarkMessagesCachePrefix(messages []Message) []Message {
	if len(messages) == 0 {
		return messages
	}
	marked, ok := markLastEligibleBlock(messages[len(messages)-1])
	if !ok {
		return messages
	}
	out := make([]Message, len(messages))
	copy(out, messages)
	out[len(out)-1] = marked
	return out
}

// markLastEligibleBlock returns msg with a cache_control breakpoint placed on
// its last eligible content block, and whether one was found. msg.Content
// must be one of the shapes this client ever assigns to a history message
// (see Message's doc): a plain string or []ToolResultBlock. Both are
// constructed right here as a block type the Messages API always accepts a
// cache_control marker on ("text", "tool_result") — the eligible set per the
// API's docs also includes "image", "tool_use", and "document", but nothing
// here ever needs to classify one of those, since neither live case produces
// them. Any other shape — including nil, empty content, or an echoed raw
// assistant turn — is reported ineligible rather than guessed at: a
// cache_control marker on an ineligible block is a 400, not a no-op, so
// skipping an unrecognized shape is the only safe default for a type this
// function has not itself vetted.
func markLastEligibleBlock(msg Message) (Message, bool) {
	switch content := msg.Content.(type) {
	case string:
		if content == "" {
			return Message{}, false
		}
		msg.Content = []messageContentBlock{{
			Type:         "text",
			Text:         content,
			CacheControl: &cacheControl{Type: "ephemeral"},
		}}
		return msg, true

	case []ToolResultBlock:
		if len(content) == 0 {
			return Message{}, false
		}
		cloned := make([]ToolResultBlock, len(content))
		copy(cloned, content)
		cloned[len(cloned)-1].CacheControl = &cacheControl{Type: "ephemeral"}
		msg.Content = cloned
		return msg, true

	default:
		return Message{}, false
	}
}

// messageContentBlock is the array-of-blocks form a plain string
// Message.Content must become to carry a cache_control marker — the API has
// no way to attach cache_control to a bare string content field.
type messageContentBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text"`
	CacheControl *cacheControl `json:"cache_control,omitempty"`
}
