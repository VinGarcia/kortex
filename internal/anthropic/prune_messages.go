package anthropic

import "encoding/json"

// PruneThinkingFromHistory returns messages with every thinking and
// redacted_thinking raw content block removed from each message, leaving
// every other message and block byte-identical. It never mutates messages or
// anything inside it — like markMessagesCachePrefix, every level it touches
// is a fresh copy, so the caller's original slice (and the slices nested
// inside it) are untouched.
//
// Why this exists: kortex re-sends the FULL message history on every
// Messages API call (there is no server-side session state). run.go's
// replaceFinalAssistantText already flattens each turn's TERMINAL assistant
// message to plain text before persisting, so a completed turn's own final
// draft carries no thinking. But an intermediate tool_use turn from a PAST,
// already-completed exchange is persisted via RawContent and echoed
// verbatim (see tools/loop.go's RunLoop) — including its thinking/
// redacted_thinking blocks, which carry a heavy base64 signature and little
// to no readable text. Those blocks served their purpose (grounding the
// tool_use that answered them) inside the request where they were produced
// and never need to be sent again afterward. Left alone, every one of them
// is re-billed on every later call for the rest of the session.
//
// Why it is safe to drop them: Anthropic requires thinking blocks preserved
// only WITHIN an active tool-use cycle — a thinking+tool_use assistant turn
// answered by a tool_result in the SAME request (confirmed live 2026-10-04,
// see run.go's governTurn comment). The messages this function is meant to
// run over are, by construction, never that: the caller (session.runTurn)
// applies it ONLY to the prior-turns History it is about to hand to
// tools.RunLoop as the base of a NEW request — never to the live,
// in-progress round-by-round slice RunLoop itself builds while a turn's own
// tool loop is still running (that slice is a local clone RunLoop appends to
// in memory; it is never routed through this function, or through the
// session Store, until the turn ends). By the time any message reaches this
// function, every tool_use/tool_result pair in it has already been through
// its one request-response round trip; there is no following request left
// in which its thinking block could still be "within" a cycle.
//
// Why it must be deterministic: the moving prompt-cache breakpoint
// (markMessagesCachePrefix) depends on today's call's prefix matching a
// previous call's prefix byte-for-byte. Pruning changes the history's shape,
// which costs exactly one cache miss the first time a given completed turn's
// intermediate messages are pruned — expected and accepted. But if the same
// input ever produced different pruned output across calls, the prefix would
// keep drifting every turn and the cache would never re-stabilize,
// eliminating the saving this function exists to add. This is a pure
// function with no randomness, no clock, and no map iteration in its output
// order (content blocks are walked and rebuilt in their original slice
// order), so the same input always produces the same output; see
// TestPruneThinkingFromHistory_isDeterministic.
func PruneThinkingFromHistory(messages []Message) []Message {
	if len(messages) == 0 {
		return messages
	}
	// Scan first, allocate only if something actually changes: most sessions
	// run core models that do not emit thinking blocks at all, and this keeps
	// that common case a true no-op rather than a needless full clone on
	// every turn.
	anyPruned := false
	pruned := make([]Message, len(messages))
	for i, msg := range messages {
		out, changed := pruneThinkingBlocks(msg)
		pruned[i] = out
		anyPruned = anyPruned || changed
	}
	if !anyPruned {
		return messages
	}
	return pruned
}

// pruneThinkingBlocks returns msg with its thinking/redacted_thinking raw
// blocks removed, and whether anything was removed. Only a message whose
// Content is []json.RawMessage — the shape MessageResponse.RawContent
// produces, i.e. an echoed assistant turn — can carry a thinking block at
// all (see markLastEligibleBlock's doc comment for the full enumeration of
// shapes this client ever assigns to Message.Content); every other shape
// (a plain string, []ToolResultBlock) is returned unchanged immediately.
//
// If removing thinking blocks would leave the content array EMPTY, msg is
// returned unchanged instead: the Messages API rejects an empty content
// array, so emitting one here would trade a cost saving for a hard failure.
// This should never actually happen (an intermediate assistant turn with no
// surviving block, e.g. no tool_use, would not have been looped on in the
// first place), but this function does not assume that invariant holds
// upstream — it fails safe exactly like addCacheControlToRawBlock and
// markLastEligibleBlock do for their own edge cases.
func pruneThinkingBlocks(msg Message) (Message, bool) {
	content, ok := msg.Content.([]json.RawMessage)
	if !ok || len(content) == 0 {
		return msg, false
	}
	kept := make([]json.RawMessage, 0, len(content))
	for _, block := range content {
		if isThinkingBlock(block) {
			continue
		}
		kept = append(kept, block)
	}
	if len(kept) == len(content) || len(kept) == 0 {
		return msg, false
	}
	msg.Content = kept
	return msg, true
}

// isThinkingBlock reports whether raw's "type" field is "thinking" or
// "redacted_thinking", reusing thinkingBlockTypes (cache_messages.go) rather
// than redefining the classification elsewhere: that map is already the
// decision of which raw block types the Messages API treats specially, and
// this function needs exactly the same decision, not a second definition of
// it that could drift out of sync. A decode failure (malformed JSON, a
// non-object block, or "type" missing/non-string) reports false: an
// unrecognized block is left in place rather than guessed at as thinking,
// the same fail-safe-toward-keeping bias addCacheControlToRawBlock applies
// in the opposite direction (fail-safe toward not marking).
func isThinkingBlock(raw json.RawMessage) bool {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil || fields == nil {
		return false
	}
	var blockType string
	if err := json.Unmarshal(fields["type"], &blockType); err != nil {
		return false
	}
	return thinkingBlockTypes[blockType]
}
