package anthropic

import (
	"bytes"
	"encoding/json"
)

// markMessagesCachePrefix returns messages with one ephemeral cache_control
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
// It is fail-safe like markStableSystemPrefix: below the model's cacheable
// floor (see minCacheableTokens — applied here exactly as it is to the system
// prefix, over the estimated size of the WHOLE messages array, not just the
// marked block) or with nothing eligible to mark, it returns messages
// unchanged and the request proceeds uncached.
//
// It never mutates messages or anything inside it — every level it touches is
// a fresh copy, so the caller's original slice is untouched.
//
// Called from CreateMessage only, on req.Messages — never on a request's
// EphemeralTail, which is concatenated AFTER this runs. A caller that must
// append a volatile per-turn tail (e.g. a tone digest, a critique) puts the
// stable history in Messages and the tail in MessageRequest.EphemeralTail so
// the breakpoint always lands on real history, never on content that is
// never read back.
//
// Wrapping a string Message.Content into a []messageContentBlock (below)
// changes its JSON shape for this call only; callers never persist the
// wrapped form, so the same message goes out as a plain string again on the
// next turn. This assumes the API normalizes both shapes to the same
// cache-prefix hash; if it doesn't, the worst case is benign — the lookback
// window still finds the previous turn's cache entry. Unverified; confirm
// against usage counters on the first live session.
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
func markMessagesCachePrefix(model string, messages []Message) []Message {
	if len(messages) == 0 {
		return messages
	}
	if estimatedMessagesTokens(messages) < minCacheableTokens(model) {
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

// estimatedMessagesTokens approximates the total size of a messages array in
// tokens using the same ~4-bytes-per-token heuristic as estimatedSystemTokens,
// so the same per-model floor (minCacheableTokens) gates both the system
// prefix and the message-history breakpoint identically. It marshals the
// whole array because message content varies in shape (plain string,
// tool_result blocks, raw echoed assistant blocks) with no single field to
// sum, unlike the system block array's fixed text field; a failed marshal —
// never expected, since every shape here already round-tripped through this
// same client — degrades to 0, so the floor check just does not mark rather
// than panicking.
func estimatedMessagesTokens(messages []Message) int {
	raw, err := json.Marshal(messages)
	if err != nil {
		return 0
	}
	return len(raw) / 4
}

// markLastEligibleBlock returns msg with a cache_control breakpoint placed on
// its last eligible content block, and whether one was found. msg.Content
// must be one of the shapes this client ever assigns to a history message
// (see Message's doc): a plain string, []ToolResultBlock, or []json.RawMessage
// (an echoed assistant turn, see MessageResponse.RawContent — session.
// textOnlyRedraft's canonical history ends in exactly this shape, since it is
// the core's own final draft turn echoed back). All three are handled as a
// block type the Messages API accepts a cache_control marker on ("text",
// "tool_result", and most raw block types — but NOT every one: Anthropic's
// docs state thinking blocks cannot carry cache_control directly, so a raw
// block decodes its "type" first and skips "thinking"/"redacted_thinking",
// the one shape among the API's broader eligible set (which also includes
// "tool_use", "image", and "document") this function must not guess at) —
// any other shape, including nil or empty content, is reported ineligible
// rather than guessed at: a cache_control marker on an ineligible block is a
// 400, not a no-op, so skipping an unrecognized shape is the only safe
// default for a type this function has not itself vetted.
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

	case []json.RawMessage:
		if len(content) == 0 {
			return Message{}, false
		}
		marked, ok := addCacheControlToRawBlock(content[len(content)-1])
		if !ok {
			return Message{}, false
		}
		cloned := make([]json.RawMessage, len(content))
		copy(cloned, content)
		cloned[len(cloned)-1] = marked
		msg.Content = cloned
		return msg, true

	default:
		return Message{}, false
	}
}

// thinkingBlockTypes are the raw block "type" values the Messages API
// rejects a cache_control marker on (confirmed against Anthropic's prompt-
// caching docs: thinking blocks cannot be cached directly) — the one gap in
// "mark whatever type the raw block already carries" that addCacheControlToRawBlock
// must check for explicitly rather than guess at.
var thinkingBlockTypes = map[string]bool{"thinking": true, "redacted_thinking": true}

// addCacheControlToRawBlock returns raw with a top-level "cache_control":
// {"type":"ephemeral"} field added, and whether that succeeded. The cache
// key Anthropic hashes is the exact bytes of the rendered block, so this
// must not round-trip raw through json.Unmarshal into a map and back
// through json.Marshal: Go's map marshaling sorts keys alphabetically and
// HTML-escapes '<', '>', '&', either of which reorders or rewrites bytes
// the previous turn already hashed, silently breaking the cache hit on
// every following turn. Instead it splices the new field in literally, byte
// for byte, immediately before the block's closing '}', so every original
// byte — field order, spacing, escaping — survives untouched.
//
// It still decodes raw into a generic field map, but only to read, never to
// re-encode: that read gets the block's "type" so it can be checked against
// thinkingBlockTypes, the one shape the Messages API rejects a
// cache_control marker on. A decode failure (including a non-object block,
// or the bare JSON literal "null" decoding into a nil map with no error)
// reports false immediately, before any byte is touched.
//
// Once the type check clears, it re-validates the raw bytes directly: the
// trimmed block must start with '{' and end with '}', or splicing before a
// closing brace that doesn't belong to the object would corrupt the block.
// Only the trimmed bytes are spliced — never raw itself — so any
// leading/trailing whitespace around the object is dropped rather than
// ending up on the wrong side of the spliced field.
func addCacheControlToRawBlock(raw json.RawMessage) (json.RawMessage, bool) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil || fields == nil {
		return nil, false
	}
	var blockType string
	if err := json.Unmarshal(fields["type"], &blockType); err == nil && thinkingBlockTypes[blockType] {
		return nil, false
	}

	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '{' || trimmed[len(trimmed)-1] != '}' {
		return nil, false
	}

	const spliced = `,"cache_control":{"type":"ephemeral"}`
	out := make(json.RawMessage, 0, len(trimmed)+len(spliced))
	out = append(out, trimmed[:len(trimmed)-1]...)
	out = append(out, spliced...)
	out = append(out, '}')
	return out, true
}

// messageContentBlock is the array-of-blocks form a plain string
// Message.Content must become to carry a cache_control marker — the API has
// no way to attach cache_control to a bare string content field.
type messageContentBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text"`
	CacheControl *cacheControl `json:"cache_control,omitempty"`
}
