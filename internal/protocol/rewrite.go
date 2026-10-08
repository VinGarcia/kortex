package protocol

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
)

// ErrUnsupportedContent marks a message whose content shape cannot be
// rewritten safely: no text, undecodable structure, or more than one text
// block (splitting one rewrite across blocks would desync any index-based
// consumer of the text).
var ErrUnsupportedContent = errors.New("unsupported message content shape")

// RewriteUserText rewrites the text of a user-message wire line in place,
// preserving the original content shape — plain string vs content-block
// array — and every envelope field it does not understand. The wire-format
// knowledge (envelope walking, message/content key path, block shapes)
// lives here on purpose: protocol is the only package that decodes lines.
//
// rewrite receives the message text and returns the replacement; ok=false
// declines the rewrite, in which case RewriteUserText returns (nil, nil).
// Content the function cannot rewrite safely returns ErrUnsupportedContent.
// A successful rewrite returns a complete '\n'-terminated line.
func RewriteUserText(line []byte, rewrite func(text string) (newText string, ok bool)) ([]byte, error) {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(line, &envelope); err != nil {
		return nil, fmt.Errorf("%w: decoding envelope: %v", ErrUnsupportedContent, err)
	}
	var message map[string]json.RawMessage
	if err := json.Unmarshal(envelope["message"], &message); err != nil {
		return nil, fmt.Errorf("%w: decoding message: %v", ErrUnsupportedContent, err)
	}
	content, ok := message["content"]
	if !ok {
		return nil, fmt.Errorf("%w: message has no content", ErrUnsupportedContent)
	}

	text, rebuild, err := contentText(content)
	if err != nil {
		return nil, err
	}
	newText, ok := rewrite(text)
	if !ok {
		return nil, nil
	}

	newContent, err := rebuild(newText)
	if err != nil {
		return nil, fmt.Errorf("rebuilding content: %w", err)
	}
	message["content"] = newContent
	newMessage, err := json.Marshal(message)
	if err != nil {
		return nil, fmt.Errorf("re-encoding message: %w", err)
	}
	envelope["message"] = newMessage
	newLine, err := json.Marshal(envelope)
	if err != nil {
		return nil, fmt.Errorf("re-encoding envelope: %w", err)
	}
	return append(newLine, '\n'), nil
}

// inboundContextMarker is the provenance sentinel OpenClaw appends to the header
// line of every inbound-context block it injects ahead of the real user message
// (its INBOUND_CONTEXT_MARKER; the gateway keys its own strippers on this same
// value — openclaw dist/strip-inbound-meta). Context blocks are joined with a
// blank line and the genuinely new message is appended last, so the text after
// the final marker-bearing block is the new message. OpenClaw collapses each
// echoed turn's body to a single line (sanitizeTranscriptBody: \s+ -> " "), so a
// context block never carries an internal blank line — that is what makes the
// blank-line split below unambiguous; only the real new message spans paragraphs.
const inboundContextMarker = "⟦openclaw:ctx⟧"

// NewMessageSegment returns the genuinely new user message within text: the part
// after the last OpenClaw inbound-context block. When no context echo is present
// the whole text is returned unchanged. OpenClaw prepends a "Conversation
// context" echo of prior turns ahead of each real user message; a caller that
// must inspect only the new turn (e.g. a facet's idempotency guard) scopes to
// this segment, or echoed content from prior turns is mistaken for the new one.
// Wire-format knowledge of the marker lives here because protocol is the only
// package that owns the claude-cli/OpenClaw wire format. DecodeInbound returns
// this same segment alongside the echo and typed info when a caller needs more
// than the new message (see inbound.go).
func NewMessageSegment(text string) string {
	_, newMessage, _ := splitInbound(text)
	return newMessage
}

// contentText extracts the rewritable text from a message content value and
// returns a rebuild function that re-encodes new text in the original
// shape. Non-text blocks (e.g. images) are preserved verbatim.
func contentText(content json.RawMessage) (text string, rebuild func(string) (json.RawMessage, error), err error) {
	trimmed := bytes.TrimSpace(content)
	if len(trimmed) == 0 {
		return "", nil, fmt.Errorf("%w: empty content value", ErrUnsupportedContent)
	}
	// Plain-string content: rewrite and re-encode as a string.
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return "", nil, fmt.Errorf("%w: decoding string content: %v", ErrUnsupportedContent, err)
		}
		return s, func(newText string) (json.RawMessage, error) {
			return json.Marshal(newText)
		}, nil
	}
	// Block-array content: rewritable only when exactly one text block exists.
	var blocks []json.RawMessage
	if err := json.Unmarshal(trimmed, &blocks); err != nil {
		return "", nil, fmt.Errorf("%w: decoding content blocks: %v", ErrUnsupportedContent, err)
	}
	textIndex := -1
	var textValue string
	for i, raw := range blocks {
		var block struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if err := json.Unmarshal(raw, &block); err != nil {
			return "", nil, fmt.Errorf("%w: decoding content block: %v", ErrUnsupportedContent, err)
		}
		if block.Type != "text" {
			continue
		}
		if textIndex != -1 {
			return "", nil, fmt.Errorf("%w: more than one text block", ErrUnsupportedContent)
		}
		textIndex = i
		textValue = block.Text
	}
	if textIndex == -1 {
		return "", nil, fmt.Errorf("%w: no text block", ErrUnsupportedContent)
	}
	return textValue, func(newText string) (json.RawMessage, error) {
		var block map[string]json.RawMessage
		if err := json.Unmarshal(blocks[textIndex], &block); err != nil {
			return nil, err
		}
		encoded, err := json.Marshal(newText)
		if err != nil {
			return nil, err
		}
		block["text"] = encoded
		newBlock, err := json.Marshal(block)
		if err != nil {
			return nil, err
		}
		blocks[textIndex] = newBlock
		return json.Marshal(blocks)
	}, nil
}
