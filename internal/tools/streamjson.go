package tools

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/protocol"
)

// StreamEmitter writes a RunLoop's turns to an OpenClaw stream-json (NDJSON)
// output, turning the pure protocol.Emit* marshallers (F3d slice-1) into the
// live RunLoop output path. A nil *StreamEmitter is the no-op the pre-F3d
// RunLoop had: every method returns early, so RunLoop can call them
// unconditionally and stay byte-for-byte the same when no sink is wired.
//
// Identity the marshallers refuse to invent — the session id and each event's
// uuid (see internal/protocol/emit.go) — is injected here by the caller:
// newUUID mints one v4 message uuid per assistant/user event so tests can feed
// a deterministic sequence and main can feed a real generator.
type StreamEmitter struct {
	w         io.Writer
	sessionID string
	newUUID   func() string
}

// NewStreamEmitter builds an emitter writing to w. sessionID and newUUID are
// the caller-owned identity the protocol marshallers cannot derive.
func NewStreamEmitter(w io.Writer, sessionID string, newUUID func() string) *StreamEmitter {
	return &StreamEmitter{w: w, sessionID: sessionID, newUUID: newUUID}
}

// systemInit emits the system/init line that opens the round.
func (e *StreamEmitter) systemInit() error {
	if e == nil {
		return nil
	}
	line, err := protocol.EmitSystemInit(e.sessionID)
	if err != nil {
		return fmt.Errorf("tools: emitting system/init: %w", err)
	}
	return e.write(line)
}

// assistant emits one assistant turn: the response's text and tool_use blocks
// with the turn's usage counters. A thinking block has no live field on this
// client's response (anthropic.ContentBlock models only text/tool_use), so it
// is dropped rather than emitted empty.
func (e *StreamEmitter) assistant(resp anthropic.MessageResponse) error {
	if e == nil {
		return nil
	}
	line, err := protocol.EmitAssistant(e.sessionID, e.newUUID(), assistantBlocks(resp.Content), protocol.Usage(resp.Usage))
	if err != nil {
		return fmt.Errorf("tools: emitting assistant turn: %w", err)
	}
	return e.write(line)
}

// toolResults emits the user turn that echoes the tool_result blocks back
// after the dispatcher ran the requested tools.
func (e *StreamEmitter) toolResults(results []anthropic.ToolResultBlock) error {
	if e == nil {
		return nil
	}
	line, err := protocol.EmitUser(e.sessionID, e.newUUID(), toolResultBlocks(results))
	if err != nil {
		return fmt.Errorf("tools: emitting tool_result turn: %w", err)
	}
	return e.write(line)
}

// result emits the terminal result line. The text is run through
// StripToolMarkup (F3d slice-2a) to satisfy protocol.EmitResult's contract
// that result text carry no raw tool-call markup.
func (e *StreamEmitter) result(res LoopResult, usage anthropic.Usage) error {
	if e == nil {
		return nil
	}
	line, err := protocol.EmitResult(e.sessionID, protocol.Result{
		Text:       StripToolMarkup(res.FinalText),
		NumTurns:   res.Turns,
		StopReason: res.StopReason,
	}, protocol.Usage(usage))
	if err != nil {
		return fmt.Errorf("tools: emitting result: %w", err)
	}
	return e.write(line)
}

func (e *StreamEmitter) write(line []byte) error {
	if _, err := e.w.Write(line); err != nil {
		return fmt.Errorf("tools: writing stream-json line: %w", err)
	}
	return nil
}

// assistantBlocks maps an assistant response's content blocks to the protocol
// content blocks of an assistant event. Only text and tool_use cross the wire
// on an assistant turn; any other block type is skipped.
func assistantBlocks(content []anthropic.ContentBlock) []protocol.ContentBlock {
	blocks := make([]protocol.ContentBlock, 0, len(content))
	for _, b := range content {
		switch b.Type {
		case "text":
			blocks = append(blocks, protocol.ContentBlock{Type: "text", Text: b.Text})
		case "tool_use":
			blocks = append(blocks, protocol.ContentBlock{Type: "tool_use", ID: b.ID, Name: b.Name, Input: b.Input})
		}
	}
	return blocks
}

// toolResultBlocks maps dispatched tool_result blocks to protocol content
// blocks. The wire carries tool_result content as a JSON value, so the
// builtin's plain-string output is JSON-encoded here.
func toolResultBlocks(results []anthropic.ToolResultBlock) []protocol.ContentBlock {
	blocks := make([]protocol.ContentBlock, len(results))
	for i, r := range results {
		// Marshalling a string never fails, so the error is dropped.
		content, _ := json.Marshal(r.Content)
		blocks[i] = protocol.ContentBlock{
			Type:      "tool_result",
			ToolUseID: r.ToolUseID,
			Content:   content,
			IsError:   r.IsError,
		}
	}
	return blocks
}
