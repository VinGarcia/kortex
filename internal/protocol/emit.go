package protocol

import "encoding/json"

// This file is the write side of the wire protocol: pure functions that
// synthesize the exact NDJSON line for each event kind. It mirrors the
// read-side types (protocol.go) so that Parse(Emit(x)) recovers x's fields.
// The caller owns event identity it cannot derive here — session id, the v4
// message uuid, and the tool_use/tool_result block payloads are passed in.

// messageEnvelope is the shared shell of the assistant and user events. Both
// carry the same top-level fields and differ only in message.role and blocks.
type messageEnvelope struct {
	Type    string   `json:"type"`
	Message *Message `json:"message"`
	UUID    string   `json:"uuid"`
	// Top-level conversation messages are never tool-nested, so this is always
	// present as null — the key is emitted, not omitted.
	ParentToolUseID *string `json:"parent_tool_use_id"`
	SessionID       string  `json:"session_id"`
}

type systemInitEnvelope struct {
	Type         string   `json:"type"`
	Subtype      string   `json:"subtype"`
	Capabilities []string `json:"capabilities"`
	SessionID    string   `json:"session_id"`
}

type resultEnvelope struct {
	Type           string  `json:"type"`
	Subtype        string  `json:"subtype"`
	IsError        bool    `json:"is_error"`
	Result         string  `json:"result"`
	NumTurns       int     `json:"num_turns"`
	StopReason     *string `json:"stop_reason"`
	TerminalReason string  `json:"terminal_reason,omitempty"`
	Usage          Usage   `json:"usage"`
	SessionID      string  `json:"session_id"`
}

type controlResponseEnvelope struct {
	Type     string         `json:"type"`
	Response controlPayload `json:"response"`
}

// EmitSystemInit builds the system/init line that opens a round. capabilities
// stays empty: msg_lifecycle_v1 may only be advertised if we also emit
// command_lifecycle{state:"started"} (F0 §4.1), which this backend does not.
func EmitSystemInit(sessionID string) ([]byte, error) {
	return marshalLine(systemInitEnvelope{
		Type:         TypeSystem,
		Subtype:      "init",
		Capabilities: []string{},
		SessionID:    sessionID,
	})
}

// EmitAssistant builds an assistant message line carrying the given content
// blocks (text and/or tool_use) and the round's usage counters.
func EmitAssistant(sessionID string, eventUUID string, blocks []ContentBlock, usage Usage) ([]byte, error) {
	return marshalLine(messageEnvelope{
		Type:      TypeAssistant,
		Message:   &Message{Role: "assistant", Content: blocks, Usage: &usage},
		UUID:      eventUUID,
		SessionID: sessionID,
	})
}

// EmitUser builds the user line that echoes tool_result blocks back to
// OpenClaw after the backend has run the tools.
func EmitUser(sessionID string, eventUUID string, blocks []ContentBlock) ([]byte, error) {
	return marshalLine(messageEnvelope{
		Type:      TypeUser,
		Message:   &Message{Role: "user", Content: blocks},
		UUID:      eventUUID,
		SessionID: sessionID,
	})
}

// EmitResult builds the terminal result line that closes a round. subtype
// mirrors is_error: OpenClaw fails the round on any subtype prefixed "error"
// (F0 §4.2). result.Text must already be free of raw tool-call markup
// (<invoke>/<parameter>) — OpenClaw rejects such a result as malformed; the
// caller assembling the final text owns that stripping, not this marshaller.
func EmitResult(sessionID string, result Result, usage Usage) ([]byte, error) {
	subtype := "success"
	if result.IsError {
		subtype = "error"
	}
	var stopReason *string
	if result.StopReason != "" {
		stopReason = &result.StopReason
	}
	return marshalLine(resultEnvelope{
		Type:           TypeResult,
		Subtype:        subtype,
		IsError:        result.IsError,
		Result:         result.Text,
		NumTurns:       result.NumTurns,
		StopReason:     stopReason,
		TerminalReason: result.TerminalReason,
		Usage:          usage,
		SessionID:      sessionID,
	})
}

// EmitControlResponse builds the success acknowledgement for a control_request
// (e.g. initialize), echoing the request's id back (F0 §5.1).
func EmitControlResponse(requestID string) ([]byte, error) {
	return marshalLine(controlResponseEnvelope{
		Type:     TypeControlResponse,
		Response: controlPayload{Subtype: "success", RequestID: requestID},
	})
}

// marshalLine renders v as a single '\n'-terminated NDJSON line, matching the
// house marshal pattern in rewrite.go.
func marshalLine(v any) ([]byte, error) {
	line, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return append(line, '\n'), nil
}
