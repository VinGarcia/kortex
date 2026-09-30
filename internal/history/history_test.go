package history

import (
	"testing"

	"github.com/vingarcia/kortex/internal/protocol"
)

// wireLine ties a raw fixture (F0 shapes) to the direction it travels.
type wireLine struct {
	direction protocol.Direction
	line      string
}

func feed(t *testing.T, lines []wireLine) *Recorder {
	t.Helper()
	rec := NewRecorder()
	for _, l := range lines {
		rec.Observe(l.direction, protocol.Parse([]byte(l.line)))
	}
	return rec
}

func TestRecorder_multiTurnSession(t *testing.T) {
	rec := feed(t, []wireLine{
		// Handshake + init: no history impact.
		{protocol.ToBackend, `{"type":"control_request","request_id":"req-1","request":{"subtype":"initialize","appendSystemPrompt":"..."}}`},
		{protocol.FromBackend, `{"type":"control_response","response":{"subtype":"success","request_id":"req-1"}}`},
		{protocol.FromBackend, `{"type":"system","subtype":"init","session_id":"sess-1","capabilities":[]}`},

		// Turn 1: plain text exchange, with replay echo and stream deltas.
		{protocol.ToBackend, `{"type":"user","message":{"role":"user","content":"oi, tudo bem?"},"uuid":"u1","session_id":"sess-1"}`},
		{protocol.FromBackend, `{"type":"user","message":{"role":"user","content":"oi, tudo bem?"},"uuid":"u1","session_id":"sess-1","isReplay":true}`},
		{protocol.FromBackend, `{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Tudo"}},"session_id":"sess-1"}`},
		{protocol.FromBackend, `{"type":"assistant","message":{"id":"msg_01","role":"assistant","content":[{"type":"text","text":"Tudo ótimo!"}]},"uuid":"a1","session_id":"sess-1"}`},
		{protocol.FromBackend, `{"type":"result","subtype":"success","is_error":false,"result":"Tudo ótimo!","num_turns":1,"stop_reason":"end_turn","session_id":"sess-1"}`},

		// Turn 2: tool use with result coming back as a stdout user event.
		{protocol.ToBackend, `{"type":"user","message":{"role":"user","content":[{"type":"text","text":"lista os arquivos"}]},"uuid":"u2","session_id":"sess-1"}`},
		{protocol.FromBackend, `{"type":"assistant","message":{"id":"msg_02","role":"assistant","content":[{"type":"tool_use","id":"toolu_01","name":"Bash","input":{"command":"ls"}}]},"uuid":"a2","session_id":"sess-1"}`},
		{protocol.FromBackend, `{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01","content":"main.go\nREADME.md"}]},"uuid":"tr1","session_id":"sess-1"}`},
		{protocol.FromBackend, `{"type":"assistant","message":{"id":"msg_03","role":"assistant","content":[{"type":"text","text":"Tem main.go e README.md."}]},"uuid":"a3","session_id":"sess-1"}`},
		{protocol.FromBackend, `{"type":"result","subtype":"success","is_error":false,"result":"Tem main.go e README.md.","num_turns":1,"stop_reason":"end_turn","session_id":"sess-1"}`},

		// Turn 3: still open (no result yet); stray non-JSON on the wire.
		{protocol.FromBackend, `stray non-json line`},
		{protocol.ToBackend, `{"type":"user","message":{"role":"user","content":"valeu!"},"uuid":"u3","session_id":"sess-1"}`},
	})

	snap := rec.Snapshot()
	if snap.SessionID != "sess-1" {
		t.Errorf("SessionID = %q, want sess-1", snap.SessionID)
	}
	if len(snap.Turns) != 3 {
		t.Fatalf("got %d turns, want 3: %+v", len(snap.Turns), snap.Turns)
	}

	turn1 := snap.Turns[0]
	if turn1.UserText != "oi, tudo bem?" || turn1.AssistantText != "Tudo ótimo!" || !turn1.Completed {
		t.Errorf("turn 1 = %+v", turn1)
	}
	if turn1.StopReason != "end_turn" || len(turn1.ToolCalls) != 0 {
		t.Errorf("turn 1 = %+v", turn1)
	}

	turn2 := snap.Turns[1]
	if turn2.UserText != "lista os arquivos" || !turn2.Completed {
		t.Errorf("turn 2 = %+v", turn2)
	}
	if len(turn2.ToolCalls) != 1 {
		t.Fatalf("turn 2 tool calls = %+v", turn2.ToolCalls)
	}
	call := turn2.ToolCalls[0]
	if call.ID != "toolu_01" || call.Name != "Bash" || call.Input != `{"command":"ls"}` || call.Result != "main.go\nREADME.md" {
		t.Errorf("tool call = %+v", call)
	}
	if turn2.AssistantText != "Tem main.go e README.md." {
		t.Errorf("turn 2 assistant text = %q", turn2.AssistantText)
	}

	turn3 := snap.Turns[2]
	if turn3.UserText != "valeu!" || turn3.Completed {
		t.Errorf("turn 3 = %+v", turn3)
	}
}

func TestRecorder_edgeCases(t *testing.T) {
	tests := []struct {
		desc      string
		lines     []wireLine
		wantTurns int
		check     func(t *testing.T, snap Snapshot)
	}{
		{
			desc: "replayed user message does not open a turn",
			lines: []wireLine{
				{protocol.FromBackend, `{"type":"user","message":{"role":"user","content":"oi"},"isReplay":true,"session_id":"s"}`},
			},
			wantTurns: 0,
		},
		{
			desc: "unknown and malformed events leave history untouched",
			lines: []wireLine{
				{protocol.ToBackend, `garbage`},
				{protocol.FromBackend, `{"type":"totally_new_thing"}`},
				{protocol.FromBackend, `{"type":"assistant","message":{"role":"assistant","content":42}}`},
			},
			wantTurns: 0,
		},
		{
			desc: "assistant output without an open turn opens one (mid-session attach)",
			lines: []wireLine{
				{protocol.FromBackend, `{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"resposta órfã"}]},"session_id":"s"}`},
				{protocol.FromBackend, `{"type":"result","is_error":false,"stop_reason":"end_turn","session_id":"s"}`},
			},
			wantTurns: 1,
			check: func(t *testing.T, snap Snapshot) {
				if snap.Turns[0].AssistantText != "resposta órfã" || !snap.Turns[0].Completed {
					t.Errorf("turn = %+v", snap.Turns[0])
				}
			},
		},
		{
			desc: "duplicate tool_use id is recorded once, first result wins",
			lines: []wireLine{
				{protocol.ToBackend, `{"type":"user","message":{"role":"user","content":"roda"},"session_id":"s"}`},
				{protocol.FromBackend, `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{}}]}}`},
				{protocol.FromBackend, `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{}}]}}`},
				{protocol.FromBackend, `{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}}`},
				{protocol.FromBackend, `{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"dup"}]}}`},
			},
			wantTurns: 1,
			check: func(t *testing.T, snap Snapshot) {
				calls := snap.Turns[0].ToolCalls
				if len(calls) != 1 || calls[0].Result != "ok" {
					t.Errorf("tool calls = %+v", calls)
				}
			},
		},
		{
			desc: "error result marks the turn",
			lines: []wireLine{
				{protocol.ToBackend, `{"type":"user","message":{"role":"user","content":"quebra"},"session_id":"s"}`},
				{protocol.FromBackend, `{"type":"result","subtype":"error_during_execution","is_error":true,"result":"","num_turns":0,"stop_reason":null}`},
			},
			wantTurns: 1,
			check: func(t *testing.T, snap Snapshot) {
				if !snap.Turns[0].IsError || !snap.Turns[0].Completed {
					t.Errorf("turn = %+v", snap.Turns[0])
				}
			},
		},
		{
			desc: "tool_result with nested block content extracts the text",
			lines: []wireLine{
				{protocol.ToBackend, `{"type":"user","message":{"role":"user","content":"roda"},"session_id":"s"}`},
				{protocol.FromBackend, `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Read","input":{}}]}}`},
				{protocol.FromBackend, `{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"conteúdo"}]}]}}`},
			},
			wantTurns: 1,
			check: func(t *testing.T, snap Snapshot) {
				if snap.Turns[0].ToolCalls[0].Result != "conteúdo" {
					t.Errorf("tool result = %q", snap.Turns[0].ToolCalls[0].Result)
				}
			},
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			snap := feed(t, test.lines).Snapshot()
			if len(snap.Turns) != test.wantTurns {
				t.Fatalf("got %d turns, want %d: %+v", len(snap.Turns), test.wantTurns, snap.Turns)
			}
			if test.check != nil {
				test.check(t, snap)
			}
		})
	}
}
