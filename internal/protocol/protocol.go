// Package protocol holds types and parsing helpers for the claude-cli
// stream-json (NDJSON) wire protocol spoken between OpenClaw and the backend
// (spec: F0). Parsing is strictly observational and lenient: a line that is
// not valid JSON, or a known type whose payload does not decode, never
// produces a failure that could break the byte-identical passthrough — the
// caller always gets an Event it can log and move past.
package protocol

import (
	"bytes"
	"encoding/json"
)

// Direction of a wire line relative to the real claude process.
type Direction string

const (
	// ToBackend is OpenClaw -> claude (the child's stdin).
	ToBackend Direction = "in"
	// FromBackend is claude -> OpenClaw (the child's stdout).
	FromBackend Direction = "out"
)

// Top-level "type" values the F0 spec documents on the wire.
const (
	TypeUser                 = "user"
	TypeAssistant            = "assistant"
	TypeSystem               = "system"
	TypeStreamEvent          = "stream_event"
	TypeResult               = "result"
	TypeControlRequest       = "control_request"
	TypeControlResponse      = "control_response"
	TypeControlCancelRequest = "control_cancel_request"
	TypeKeepAlive            = "keep_alive"
	TypeCommandLifecycle     = "command_lifecycle"

	// TypeUnknown marks a line that is not a JSON object with a string
	// "type" — legal on the wire (readers skip it) and passed through as-is.
	TypeUnknown = "unknown"
)

// Event is one parsed NDJSON line. Only the fields relevant to the line's
// type are populated; everything else stays zero. Raw aliases the input line
// (trimmed), so it is only valid until the caller's buffer is reused.
type Event struct {
	Type string
	// Subtype disambiguates within a type: system init/task_started/...,
	// result success/error*, and the request subtype of a control_request
	// (initialize, can_use_tool, hook_callback, ...).
	Subtype   string
	SessionID string
	UUID      string
	// RequestID correlates control_request/control_response/cancel pairs.
	RequestID string
	// IsReplay marks user messages echoed back on stdout by
	// --replay-user-messages.
	IsReplay bool
	// Message is the Anthropic message envelope of user/assistant events.
	Message *Message
	// Result is populated for the terminal type:"result" event.
	Result *Result
	// StreamEventType is the inner SSE event type of a stream_event
	// (message_start, content_block_delta, ...).
	StreamEventType string
	Raw             json.RawMessage
	// ParseErr is non-nil when the JSON object decoded partially: the type
	// is known but some payload field had an unexpected shape. The event is
	// still usable; the error exists for debug logging only.
	ParseErr error
}

// Message is the Anthropic message envelope carried by user/assistant events.
type Message struct {
	ID      string  `json:"id,omitempty"`
	Role    string  `json:"role"`
	Content Content `json:"content"`
	Usage   *Usage  `json:"usage,omitempty"`
}

// Content normalizes the wire's two shapes for message content — a plain
// string or an array of content blocks — into a block list (a string becomes
// a single text block).
type Content []ContentBlock

func (c *Content) UnmarshalJSON(data []byte) error {
	data = bytes.TrimSpace(data)
	if len(data) == 0 || bytes.Equal(data, []byte("null")) {
		return nil
	}
	if data[0] == '"' {
		var s string
		if err := json.Unmarshal(data, &s); err != nil {
			return err
		}
		*c = Content{{Type: "text", Text: s}}
		return nil
	}
	var blocks []ContentBlock
	if err := json.Unmarshal(data, &blocks); err != nil {
		return err
	}
	*c = blocks
	return nil
}

// ContentBlock is one Anthropic content block. Fields are a union across the
// block types seen on this wire (text, thinking, tool_use, tool_result);
// only the ones matching Type are set.
type ContentBlock struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Thinking string `json:"thinking,omitempty"`
	// tool_use fields.
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
	// tool_result fields. Content stays raw because the wire allows either
	// a string or nested blocks there.
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

// Usage carries the four token counters OpenClaw reads (F0 §4.4).
type Usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

// Result carries the fields of the terminal type:"result" event that
// OpenClaw reads (F0 §4.2).
type Result struct {
	IsError        bool
	Text           string
	NumTurns       int
	StopReason     string
	TerminalReason string
}

// wireEvent mirrors the superset of top-level fields across all documented
// event types; Parse projects it into the public Event.
type wireEvent struct {
	Type      string          `json:"type"`
	Subtype   string          `json:"subtype"`
	UUID      string          `json:"uuid"`
	RequestID string          `json:"request_id"`
	IsReplay  bool            `json:"isReplay"`
	Message   json.RawMessage `json:"message"`
	Event     *innerType      `json:"event"`
	Request   *controlRequest `json:"request"`
	Response  *controlPayload `json:"response"`

	// The four aliases OpenClaw accepts for the session id (F0 §4.3).
	SessionID         string `json:"session_id"`
	SessionIDCamel    string `json:"sessionId"`
	ConversationID    string `json:"conversation_id"`
	ConversationCamel string `json:"conversationId"`

	// type:"result" fields.
	IsError        bool    `json:"is_error"`
	Result         string  `json:"result"`
	NumTurns       int     `json:"num_turns"`
	StopReason     *string `json:"stop_reason"`
	TerminalReason string  `json:"terminal_reason"`
}

type innerType struct {
	Type string `json:"type"`
}

type controlRequest struct {
	Subtype string `json:"subtype"`
}

type controlPayload struct {
	Subtype   string `json:"subtype"`
	RequestID string `json:"request_id"`
}

// Parse decodes one wire line into an Event. It never fails: a line that is
// not a JSON object with a string type comes back as TypeUnknown, and a
// known type with a partially decodable payload comes back with the decoded
// subset plus ParseErr set.
func Parse(line []byte) Event {
	trimmed := bytes.TrimSpace(line)
	var w wireEvent
	err := json.Unmarshal(trimmed, &w)
	if w.Type == "" {
		return Event{Type: TypeUnknown, Raw: trimmed}
	}

	// The message envelope decodes in a second pass so that a message with
	// an unexpected shape cannot abort the envelope decode and lose the
	// session id / uuid that came after it on the line.
	var msg *Message
	if len(w.Message) > 0 && !bytes.Equal(w.Message, []byte("null")) {
		var m Message
		if merr := json.Unmarshal(w.Message, &m); merr != nil {
			if err == nil {
				err = merr
			}
		} else {
			msg = &m
		}
	}

	ev := Event{
		Type:      w.Type,
		Subtype:   w.Subtype,
		SessionID: firstNonEmpty(w.SessionID, w.SessionIDCamel, w.ConversationID, w.ConversationCamel),
		UUID:      w.UUID,
		RequestID: w.RequestID,
		IsReplay:  w.IsReplay,
		Message:   msg,
		Raw:       trimmed,
		ParseErr:  err,
	}
	switch w.Type {
	case TypeResult:
		result := Result{
			IsError:        w.IsError,
			Text:           w.Result,
			NumTurns:       w.NumTurns,
			TerminalReason: w.TerminalReason,
		}
		if w.StopReason != nil {
			result.StopReason = *w.StopReason
		}
		ev.Result = &result
	case TypeStreamEvent:
		if w.Event != nil {
			ev.StreamEventType = w.Event.Type
		}
	case TypeControlRequest:
		if w.Request != nil {
			ev.Subtype = w.Request.Subtype
		}
	case TypeControlResponse:
		if w.Response != nil {
			ev.Subtype = w.Response.Subtype
			ev.RequestID = w.Response.RequestID
		}
	}
	return ev
}

// TextContent joins the text blocks of a message with newlines; "" when the
// message carries no plain text.
func (m *Message) TextContent() string {
	if m == nil {
		return ""
	}
	var buf bytes.Buffer
	for _, block := range m.Content {
		if block.Type != "text" || block.Text == "" {
			continue
		}
		if buf.Len() > 0 {
			buf.WriteByte('\n')
		}
		buf.WriteString(block.Text)
	}
	return buf.String()
}

type envelope struct {
	Type string `json:"type"`
}

// MessageType returns the top-level "type" field of a stream-json line, or
// "" when the line is not a JSON object with a string type. It never fails:
// non-JSON lines are legal on the wire (the reader on the other side ignores
// them) and must never break the stream.
func MessageType(line []byte) string {
	var e envelope
	if err := json.Unmarshal(bytes.TrimSpace(line), &e); err != nil {
		return ""
	}
	return e.Type
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}
