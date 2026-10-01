package tools

import "regexp"

// invokeBlockPattern matches one Anthropic tool-call
// "<invoke name="...">...</invoke>" block, non-greedy so two separate blocks
// stay two matches. A block only matches with both tags present, so an
// unterminated "<invoke ...>" is left untouched — conservative on malformed
// input rather than guessing at a span.
//
// Known limitations (a real single tool-call block hits neither, so both are
// accepted rather than fixed):
//   - "[^>]*" stops at the first '>', so a literal '>' inside the opening
//     tag's attributes truncates the match early.
//   - ".*?</invoke>" closes on the first literal "</invoke>", so a <parameter>
//     value that contains that text would leave a stray "<invoke ...>" prefix
//     unstripped.
var invokeBlockPattern = regexp.MustCompile(`(?s)<invoke\b[^>]*>.*?</invoke>`)

// StripToolMarkup removes raw Anthropic tool-call markup (<invoke>...</invoke>
// blocks, nested <parameter> tags included) from model output text. Driver
// code must call this before handing result text to protocol.EmitResult, whose
// contract (see internal/protocol/emit.go) requires text already free of this
// markup because OpenClaw rejects a result line that still contains it.
func StripToolMarkup(text string) string {
	return invokeBlockPattern.ReplaceAllString(text, "")
}
