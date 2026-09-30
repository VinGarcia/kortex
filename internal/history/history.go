// Package history holds the canonical conversation history of the current
// session, reconstructed out of the protocol events flowing through the
// proxy. It is purely observational state: the F2b facets will read it, and
// nothing here ever writes back to the stream.
//
// The recorder exposes intent-named entry points (a user prompt, an
// assistant message, a tool result, a completed turn); classifying wire
// lines into those intents — which pipe carried them, replay echoes — is
// the caller's job, so this package knows nothing about transport details.
package history

import (
	"github.com/vingarcia/kortex/internal/protocol"
)

// ToolCall is one assistant tool invocation, summarized: the raw input JSON
// is kept verbatim as an opaque blob, and the result as the plain text the
// tool_result block carried.
type ToolCall struct {
	ID     string `json:"id"`
	Name   string `json:"name"`
	Input  string `json:"input,omitempty"`
	Result string `json:"result,omitempty"`
	// IsError mirrors the tool_result block's is_error flag.
	IsError bool `json:"isError,omitempty"`
}

// Turn is one round of the conversation: a user input and everything the
// assistant produced until the terminal result event.
type Turn struct {
	UserText      string     `json:"userText"`
	AssistantText string     `json:"assistantText,omitempty"`
	ToolCalls     []ToolCall `json:"toolCalls,omitempty"`
	// Completed flips when the round's terminal result event arrives.
	Completed  bool   `json:"completed"`
	IsError    bool   `json:"isError,omitempty"`
	StopReason string `json:"stopReason,omitempty"`
}

// Recorder accumulates the canonical history of one session. It is not safe
// for concurrent use; the caller serializes calls (the proxy observer holds
// a mutex).
type Recorder struct {
	sessionID string
	turns     []Turn
}

func NewRecorder() *Recorder {
	return &Recorder{}
}

// SetSessionID pins the session id on first sight; later calls are no-ops
// (the id never changes within one process).
func (r *Recorder) SetSessionID(id string) {
	if r.sessionID == "" && id != "" {
		r.sessionID = id
	}
}

// RecordUserPrompt opens a new turn for a fresh user input.
func (r *Recorder) RecordUserPrompt(text string) {
	if text == "" {
		return
	}
	if current := r.openTurn(); current != nil && current.UserText == "" {
		current.UserText = text
		return
	}
	r.turns = append(r.turns, Turn{UserText: text})
}

// RecordAssistantMessage folds an assistant message's text and tool_use
// blocks into the open turn.
func (r *Recorder) RecordAssistantMessage(message *protocol.Message) {
	if message == nil {
		return
	}
	current := r.openTurn()
	if current == nil {
		// Assistant output with no open turn (e.g. kortex attached
		// mid-session): open one so the content is not dropped.
		r.turns = append(r.turns, Turn{})
		current = &r.turns[len(r.turns)-1]
	}
	for _, block := range message.Content {
		switch block.Type {
		case "text":
			if block.Text == "" {
				continue
			}
			if current.AssistantText != "" {
				current.AssistantText += "\n"
			}
			current.AssistantText += block.Text
		case "tool_use":
			if r.hasToolCall(current, block.ID) {
				continue
			}
			current.ToolCalls = append(current.ToolCalls, ToolCall{
				ID:    block.ID,
				Name:  block.Name,
				Input: string(block.Input),
			})
		}
	}
}

// AttachToolResult records a tool's outcome on the matching tool call.
func (r *Recorder) AttachToolResult(toolUseID string, resultText string, isError bool) {
	if toolUseID == "" {
		return
	}
	for i := len(r.turns) - 1; i >= 0; i-- {
		calls := r.turns[i].ToolCalls
		for j := range calls {
			if calls[j].ID != toolUseID {
				continue
			}
			// First writer wins: the same result can arrive again as a
			// replay echo.
			if calls[j].Result == "" {
				calls[j].Result = resultText
				calls[j].IsError = isError
			}
			return
		}
	}
}

// CompleteTurn closes the open turn with the terminal result's outcome.
func (r *Recorder) CompleteTurn(result *protocol.Result) {
	current := r.openTurn()
	if current == nil {
		return
	}
	current.Completed = true
	if result != nil {
		current.IsError = result.IsError
		current.StopReason = result.StopReason
	}
}

// openTurn returns the turn still accumulating output, or nil when the last
// turn already completed (or none exists).
func (r *Recorder) openTurn() *Turn {
	if len(r.turns) == 0 {
		return nil
	}
	last := &r.turns[len(r.turns)-1]
	if last.Completed {
		return nil
	}
	return last
}

func (r *Recorder) hasToolCall(turn *Turn, id string) bool {
	if id == "" {
		return false
	}
	for _, call := range turn.ToolCalls {
		if call.ID == id {
			return true
		}
	}
	return false
}

// Snapshot is an immutable copy of the reconstructed history.
type Snapshot struct {
	SessionID string `json:"sessionId,omitempty"`
	Turns     []Turn `json:"turns"`
}

func (r *Recorder) Snapshot() Snapshot {
	turns := make([]Turn, len(r.turns))
	copy(turns, r.turns)
	for i := range turns {
		turns[i].ToolCalls = append([]ToolCall(nil), turns[i].ToolCalls...)
	}
	return Snapshot{SessionID: r.sessionID, Turns: turns}
}
