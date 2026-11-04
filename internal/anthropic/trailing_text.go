package anthropic

import (
	"encoding/json"
	"fmt"
	"slices"
)

// WithTrailingUserText returns a copy of messages whose last message, which must be
// a user turn, also carries text as a final text block. The client does not merge
// consecutive user messages, so the text rides inside the last user turn rather than
// as a new one. messages is never mutated: the last message is replaced in the copy
// only. An unrecognized content shape falls back to a separate trailing user message,
// which the Messages API merges itself.
func WithTrailingUserText(messages []Message, text string) ([]Message, error) {
	out := slices.Clone(messages)
	if len(out) == 0 {
		return append(out, Message{Role: "user", Content: text}), nil
	}
	last := len(out) - 1
	var blocks []json.RawMessage
	switch content := out[last].Content.(type) {
	case string:
		first, err := textBlockJSON(content)
		if err != nil {
			return nil, err
		}
		blocks = []json.RawMessage{first}
	case []ToolResultBlock:
		for _, block := range content {
			raw, err := json.Marshal(block)
			if err != nil {
				return nil, fmt.Errorf("anthropic: encode tool_result block: %w", err)
			}
			blocks = append(blocks, raw)
		}
	case []json.RawMessage:
		blocks = slices.Clone(content)
	default:
		return append(out, Message{Role: "user", Content: text}), nil
	}
	trailing, err := textBlockJSON(text)
	if err != nil {
		return nil, err
	}
	out[last] = Message{Role: out[last].Role, Content: append(blocks, trailing)}
	return out, nil
}

func textBlockJSON(text string) (json.RawMessage, error) {
	raw, err := json.Marshal(map[string]string{"type": "text", "text": text})
	if err != nil {
		return nil, fmt.Errorf("anthropic: encode text block: %w", err)
	}
	return raw, nil
}
