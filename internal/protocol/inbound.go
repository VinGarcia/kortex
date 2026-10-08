package protocol

import (
	"encoding/json"
	"strings"
)

// Inbound is the decoded shape of one OpenClaw inbound user message. OpenClaw
// prepends a run of "inbound context" blocks (a typed "Conversation info" block
// plus an echo of prior turns) ahead of the genuinely new user message; this
// type separates those parts so a caller can forward only the new message to the
// backend and drop the echo it already holds in canonical history (DIRETIVA
// #4803). Decoding lives in package protocol because it is the only package that
// owns the claude-cli/OpenClaw wire format — the marker, the block layout, and
// the JSON shape of the Conversation info block all stay here.
type Inbound struct {
	// Info is the typed decode of the "Conversation info" block. Valid only when
	// HasInfo is true (the block was present and parsed).
	Info InboundInfo
	// HasInfo reports whether a parseable Conversation info block was found.
	HasInfo bool
	// Echo is the raw run of OpenClaw-injected context blocks that preceded the
	// new message, verbatim and still joined as received (blocks separated by a
	// blank line). Empty when the message carried no inbound context. It is the
	// echo of prior turns a caller discards on a resumed session but preserves as
	// marked bootstrap history on the first turn of a fresh one (#4803 part 4).
	Echo string
	// NewMessage is the genuinely new user message: the text after the last
	// context block, or the whole text when no inbound context is present.
	NewMessage string
}

// InboundInfo is the typed decode of OpenClaw's "Conversation info" JSON block.
// Fields map to the block's keys; any key OpenClaw omits for a given message
// (e.g. reply_to_id on a message that replies to nothing) decodes to its zero
// value. Unknown keys are ignored, so a future OpenClaw field never breaks the
// decode.
type InboundInfo struct {
	ChatID           string        `json:"chat_id"`
	MessageID        string        `json:"message_id"`
	ReplyToID        string        `json:"reply_to_id"`
	Sender           InboundSender `json:"sender"`
	Timestamp        string        `json:"timestamp"`
	InboundEventKind string        `json:"inbound_event_kind"`
}

// InboundSender is the "sender" object nested in the Conversation info block.
type InboundSender struct {
	Name     string `json:"name"`
	Username string `json:"username"`
}

// conversationInfoHeader is the exact header line OpenClaw emits for the
// Conversation info block (its label with the provenance marker appended; see
// markInboundContextLabel in openclaw dist/strip-inbound-meta). The block that
// carries it holds the typed metadata DecodeInbound parses.
const conversationInfoHeader = "Conversation info: " + inboundContextMarker

// DecodeInbound splits one inbound user message into its echo, typed info, and
// genuinely new message. When no OpenClaw inbound context is present it returns
// the whole text as NewMessage with Echo empty and HasInfo false, so a caller
// can treat a plain message and an echo-laden one through the same path.
func DecodeInbound(text string) Inbound {
	echo, newMessage, hasEcho := splitInbound(text)
	inbound := Inbound{Echo: echo, NewMessage: newMessage}
	if !hasEcho {
		return inbound
	}
	// The Conversation info block lives among the echo blocks; find it by its
	// marker-bearing header and decode its fenced JSON body.
	for _, block := range strings.Split(echo, "\n\n") {
		if !strings.Contains(block, conversationInfoHeader) {
			continue
		}
		if info, ok := parseConversationInfo(block); ok {
			inbound.Info = info
			inbound.HasInfo = true
		}
		break
	}
	return inbound
}

// splitInbound divides text at the boundary after the last OpenClaw context
// block: echo is every context block (joined as received), newMessage is the
// text after them, and hasEcho reports whether any context block was present.
// OpenClaw joins the context blocks and the new message with a blank line and
// collapses each echoed turn's body to a single line (sanitizeTranscriptBody),
// so a context block never carries an internal blank line — that is what makes
// this blank-line split unambiguous; only the real new message spans paragraphs.
func splitInbound(text string) (echo string, newMessage string, hasEcho bool) {
	if !strings.Contains(text, inboundContextMarker) {
		return "", text, false
	}
	blocks := strings.Split(text, "\n\n")
	lastContext := -1
	for i, block := range blocks {
		if strings.Contains(block, inboundContextMarker) {
			lastContext = i
		}
	}
	echo = strings.Join(blocks[:lastContext+1], "\n\n")
	newMessage = strings.Join(blocks[lastContext+1:], "\n\n")
	return echo, newMessage, true
}

// parseConversationInfo extracts and decodes the JSON body of a Conversation
// info block. The block is formatted as a header line, a "```json" fence, the
// single-line JSON payload, and a closing "```" fence (openclaw dist
// formatContextJsonBlock), so the body is the text between the fences. ok=false
// when the fences are missing or the JSON does not decode, so a malformed block
// degrades to "no typed info" rather than failing the whole decode.
func parseConversationInfo(block string) (InboundInfo, bool) {
	const fenceOpen = "```json\n"
	open := strings.Index(block, fenceOpen)
	if open < 0 {
		return InboundInfo{}, false
	}
	body := block[open+len(fenceOpen):]
	close := strings.LastIndex(body, "```")
	if close < 0 {
		return InboundInfo{}, false
	}
	var info InboundInfo
	if err := json.Unmarshal([]byte(strings.TrimSpace(body[:close])), &info); err != nil {
		return InboundInfo{}, false
	}
	return info, true
}
