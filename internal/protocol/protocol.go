// Package protocol holds types and parsing helpers for the claude-cli
// stream-json (NDJSON) wire protocol spoken between OpenClaw and the backend.
//
// In F1 (pass-through) only lenient message-type sniffing is needed; the
// full message types land here in F2 when kortex starts interpreting the
// stream instead of proxying it.
package protocol

import (
	"bytes"
	"encoding/json"
)

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
