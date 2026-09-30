// Package history reconstructs the canonical conversation history of the
// current session out of the parsed protocol events flowing through the
// proxy. It is purely observational state: the F2b facets will read it, and
// nothing here ever writes back to the stream.
package history

import (
	"encoding/json"

	"github.com/vingarcia/kortex/internal/protocol"
)

// ToolCall is one assistant tool invocation, summarized: the raw input JSON
// is kept verbatim but the result is whatever the tool_result block carried
// (string or nested blocks, as raw JSON).
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

// Recorder accumulates the canonical history of one session from observed
// wire events. It is not safe for concurrent use; the caller serializes
// Observe calls (the proxy observer holds a mutex).
type Recorder struct {
	sessionID string
	turns     []Turn
}

func NewRecorder() *Recorder {
	return &Recorder{}
}

// Observe folds one parsed wire event into the history. Events that carry
// no durable conversation state (stream deltas, control traffic, system
// notices, unknown lines) are ignored.
func (r *Recorder) Observe(direction protocol.Direction, event protocol.Event) {
	if r.sessionID == "" && event.SessionID != "" {
		r.sessionID = event.SessionID
	}
	switch event.Type {
	case protocol.TypeUser:
		r.observeUser(direction, event)
	case protocol.TypeAssistant:
		r.observeAssistant(event)
	case protocol.TypeResult:
		r.observeResult(event)
	}
}

func (r *Recorder) observeUser(direction protocol.Direction, event protocol.Event) {
	if event.Message == nil {
		return
	}
	// Tool results travel as user messages; they belong to the open turn's
	// tool calls no matter which direction carried them.
	for _, block := range event.Message.Content {
		if block.Type == "tool_result" {
			r.attachToolResult(block)
		}
	}
	// Only a fresh prompt on stdin opens a turn: stdout user events are
	// either --replay-user-messages echoes or tool-result carriers.
	if direction != protocol.ToBackend || event.IsReplay {
		return
	}
	text := event.Message.TextContent()
	if text == "" {
		return
	}
	if current := r.openTurn(); current != nil && current.UserText == "" {
		current.UserText = text
		return
	}
	r.turns = append(r.turns, Turn{UserText: text})
}

func (r *Recorder) observeAssistant(event protocol.Event) {
	if event.Message == nil {
		return
	}
	current := r.openTurn()
	if current == nil {
		// Assistant output with no open turn (e.g. kortex attached
		// mid-session): open one so the content is not dropped.
		r.turns = append(r.turns, Turn{})
		current = &r.turns[len(r.turns)-1]
	}
	for _, block := range event.Message.Content {
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

func (r *Recorder) observeResult(event protocol.Event) {
	current := r.openTurn()
	if current == nil {
		return
	}
	current.Completed = true
	if event.Result != nil {
		current.IsError = event.Result.IsError
		current.StopReason = event.Result.StopReason
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

func (r *Recorder) attachToolResult(block protocol.ContentBlock) {
	if block.ToolUseID == "" {
		return
	}
	for i := len(r.turns) - 1; i >= 0; i-- {
		calls := r.turns[i].ToolCalls
		for j := range calls {
			if calls[j].ID != block.ToolUseID {
				continue
			}
			// First writer wins: the same result can arrive again as a
			// --replay-user-messages echo.
			if calls[j].Result == "" {
				calls[j].Result = toolResultText(block.Content)
				calls[j].IsError = block.IsError
			}
			return
		}
	}
}

// toolResultText renders a tool_result payload (string or nested blocks on
// the wire) as plain text for the summary.
func toolResultText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var blocks protocol.Content
	if err := json.Unmarshal(raw, &blocks); err == nil {
		msg := protocol.Message{Content: blocks}
		if text := msg.TextContent(); text != "" {
			return text
		}
	}
	return string(raw)
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
