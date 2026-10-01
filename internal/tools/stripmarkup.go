package tools

import "regexp"

// invokeBlockPattern matches one Anthropic tool-call "<invoke name="...">
// ... </invoke>" block. The (?s) flag lets "." cross newlines, since a real
// invoke block always spans several lines (one per <parameter>). The match
// is non-greedy ("*?") so a string with two separate invoke blocks yields
// two matches instead of one match spanning both. Nested
// "<parameter name="...">...</parameter>" tags fall inside the matched span
// and are removed along with their enclosing invoke block — they are never
// matched on their own.
//
// A block only matches when both the opening and closing tag are present.
// An unterminated "<invoke ...>" with no later "</invoke>" has nothing to
// match against, so it is left in the text untouched rather than guessed
// at — see StripToolMarkup's conservative-on-malformed-input contract.
//
// Known limitation: "[^>]*" stops at the first literal '>' after "<invoke",
// so an attribute value containing a literal '>' would truncate the match
// early. The real tool-call format never puts '>' in a tool name, so this
// is not expected to occur in practice.
var invokeBlockPattern = regexp.MustCompile(`(?s)<invoke\b[^>]*>.*?</invoke>`)

// StripToolMarkup removes raw Anthropic tool-call markup — <invoke
// name="...">...</invoke> blocks, including any <parameter> tags nested
// inside them — from model output text. Driver code must call this before
// handing result text to protocol.EmitResult: EmitResult's contract (see
// internal/protocol/emit.go) requires the text already be free of this
// markup, because OpenClaw rejects a result line that still contains it.
//
// Text outside a matched block is returned byte-for-byte unchanged,
// including whitespace — StripToolMarkup only ever deletes matched spans,
// it never reformats or trims the surrounding prose. When no <invoke>...
// </invoke> block is found at all (no markup, or an unclosed/malformed
// one), the input is returned as-is.
func StripToolMarkup(text string) string {
	return invokeBlockPattern.ReplaceAllString(text, "")
}
