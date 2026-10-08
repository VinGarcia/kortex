package history

import (
	"testing"

	"github.com/vingarcia/kortex/internal/protocol"
)

func TestRecorder_turnLifecycle(t *testing.T) {
	rec := NewRecorder()
	rec.SetSessionID("sess-1")
	rec.SetSessionID("sess-other") // first id wins

	// Turn 1: plain text exchange.
	rec.RecordUserPrompt("oi, tudo bem?")
	rec.RecordAssistantMessage(&protocol.Message{Content: protocol.Content{
		{Type: "text", Text: "Tudo ótimo!"},
	}})
	rec.CompleteTurn(&protocol.Result{StopReason: "end_turn"})

	// Turn 2: tool use, result attached, then final text.
	rec.RecordUserPrompt("lista os arquivos")
	rec.RecordAssistantMessage(&protocol.Message{Content: protocol.Content{
		{Type: "tool_use", ID: "toolu_01", Name: "Bash", Input: []byte(`{"command":"ls"}`)},
	}})
	rec.AttachToolResult("toolu_01", "main.go\nREADME.md", false)
	rec.RecordAssistantMessage(&protocol.Message{Content: protocol.Content{
		{Type: "text", Text: "Tem main.go e README.md."},
	}})
	rec.CompleteTurn(&protocol.Result{StopReason: "end_turn"})

	// Turn 3: still open.
	rec.RecordUserPrompt("valeu!")

	snap := rec.Snapshot()
	if snap.SessionID != "sess-1" {
		t.Errorf("SessionID = %q, want sess-1", snap.SessionID)
	}
	if len(snap.Turns) != 3 {
		t.Fatalf("got %d turns, want 3: %+v", len(snap.Turns), snap.Turns)
	}
	want1 := Turn{UserText: "oi, tudo bem?", AssistantText: "Tudo ótimo!", Completed: true, StopReason: "end_turn"}
	if got := snap.Turns[0]; got.UserText != want1.UserText || got.AssistantText != want1.AssistantText ||
		got.Completed != want1.Completed || got.StopReason != want1.StopReason || len(got.ToolCalls) != 0 {
		t.Errorf("turn 1 = %+v, want %+v", got, want1)
	}
	turn2 := snap.Turns[1]
	wantCall := ToolCall{ID: "toolu_01", Name: "Bash", Input: `{"command":"ls"}`, Result: "main.go\nREADME.md"}
	if len(turn2.ToolCalls) != 1 || turn2.ToolCalls[0] != wantCall {
		t.Errorf("turn 2 tool calls = %+v, want [%+v]", turn2.ToolCalls, wantCall)
	}
	if turn2.UserText != "lista os arquivos" || turn2.AssistantText != "Tem main.go e README.md." || !turn2.Completed {
		t.Errorf("turn 2 = %+v", turn2)
	}
	if turn3 := snap.Turns[2]; turn3.UserText != "valeu!" || turn3.Completed {
		t.Errorf("turn 3 = %+v", turn3)
	}
}

func TestRecorder_edgeCases(t *testing.T) {
	tests := []struct {
		desc      string
		feed      func(rec *Recorder)
		wantTurns []Turn
	}{
		{
			desc: "empty prompt does not open a turn",
			feed: func(rec *Recorder) {
				rec.RecordUserPrompt("")
			},
			wantTurns: []Turn{},
		},
		{
			desc: "nil assistant message and nil result are no-ops",
			feed: func(rec *Recorder) {
				rec.RecordAssistantMessage(nil)
				rec.CompleteTurn(nil)
			},
			wantTurns: []Turn{},
		},
		{
			desc: "assistant output without an open turn opens one (mid-session attach)",
			feed: func(rec *Recorder) {
				rec.RecordAssistantMessage(&protocol.Message{Content: protocol.Content{
					{Type: "text", Text: "resposta órfã"},
				}})
				rec.CompleteTurn(&protocol.Result{StopReason: "end_turn"})
			},
			wantTurns: []Turn{{AssistantText: "resposta órfã", Completed: true, StopReason: "end_turn"}},
		},
		{
			desc: "duplicate tool_use id is recorded once, first result wins",
			feed: func(rec *Recorder) {
				rec.RecordUserPrompt("roda")
				toolUse := &protocol.Message{Content: protocol.Content{
					{Type: "tool_use", ID: "t1", Name: "Bash", Input: []byte(`{}`)},
				}}
				rec.RecordAssistantMessage(toolUse)
				rec.RecordAssistantMessage(toolUse)
				rec.AttachToolResult("t1", "ok", false)
				rec.AttachToolResult("t1", "dup", true)
			},
			wantTurns: []Turn{{UserText: "roda", ToolCalls: []ToolCall{
				{ID: "t1", Name: "Bash", Input: `{}`, Result: "ok"},
			}}},
		},
		{
			desc: "tool result for an unknown id is dropped",
			feed: func(rec *Recorder) {
				rec.RecordUserPrompt("roda")
				rec.AttachToolResult("missing", "ok", false)
			},
			wantTurns: []Turn{{UserText: "roda"}},
		},
		{
			desc: "error result marks the turn",
			feed: func(rec *Recorder) {
				rec.RecordUserPrompt("quebra")
				rec.CompleteTurn(&protocol.Result{IsError: true})
			},
			wantTurns: []Turn{{UserText: "quebra", Completed: true, IsError: true}},
		},
		{
			desc: "user prompt after an incomplete turn starts a fresh turn",
			feed: func(rec *Recorder) {
				rec.RecordUserPrompt("primeiro")
				rec.RecordAssistantMessage(&protocol.Message{Content: protocol.Content{
					{Type: "text", Text: "parcial"},
				}})
				rec.RecordUserPrompt("segundo")
			},
			wantTurns: []Turn{
				{UserText: "primeiro", AssistantText: "parcial"},
				{UserText: "segundo"},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			rec := NewRecorder()
			test.feed(rec)
			snap := rec.Snapshot()
			if len(snap.Turns) != len(test.wantTurns) {
				t.Fatalf("got %d turns, want %d: %+v", len(snap.Turns), len(test.wantTurns), snap.Turns)
			}
			for i, want := range test.wantTurns {
				got := snap.Turns[i]
				if got.UserText != want.UserText || got.AssistantText != want.AssistantText ||
					got.Completed != want.Completed || got.IsError != want.IsError ||
					got.StopReason != want.StopReason {
					t.Errorf("turn %d = %+v, want %+v", i, got, want)
				}
				if len(got.ToolCalls) != len(want.ToolCalls) {
					t.Fatalf("turn %d tool calls = %+v, want %+v", i, got.ToolCalls, want.ToolCalls)
				}
				for j := range want.ToolCalls {
					if got.ToolCalls[j] != want.ToolCalls[j] {
						t.Errorf("turn %d call %d = %+v, want %+v", i, j, got.ToolCalls[j], want.ToolCalls[j])
					}
				}
			}
		})
	}
}

func TestRecorder_lastCompletedTurn(t *testing.T) {
	rec := NewRecorder()

	if _, _, ok := rec.LastCompletedTurn(); ok {
		t.Error("empty recorder should have no completed turn")
	}

	rec.RecordUserPrompt("oi")
	rec.RecordAssistantMessage(&protocol.Message{Content: protocol.Content{
		{Type: "text", Text: "resposta"},
	}})
	if _, _, ok := rec.LastCompletedTurn(); ok {
		t.Error("open turn must not report as completed")
	}

	if !rec.CompleteTurn(&protocol.Result{StopReason: "end_turn"}) {
		t.Fatal("CompleteTurn should report closing the open turn")
	}
	index, turn, ok := rec.LastCompletedTurn()
	if !ok || index != 0 || turn.AssistantText != "resposta" || !turn.Completed {
		t.Fatalf("LastCompletedTurn = (%d, %+v, %v)", index, turn, ok)
	}

	// A stray duplicate result closes nothing: the hook must not re-fire.
	if rec.CompleteTurn(&protocol.Result{}) {
		t.Error("duplicate result must not report a newly closed turn")
	}
}

func TestAppendToolCalls(t *testing.T) {
	base := []ToolCall{{ID: "a"}, {ID: "b"}}
	extra := []ToolCall{{ID: "c"}, {ID: "d"}}

	got := AppendToolCalls(base, extra...)

	// Order: base first, then extra, in the given order.
	want := []string{"a", "b", "c", "d"}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d (%+v)", len(got), len(want), got)
	}
	for i, id := range want {
		if got[i].ID != id {
			t.Errorf("got[%d].ID = %q, want %q", i, got[i].ID, id)
		}
	}

	// Non-mutation and non-aliasing: the owner must never disturb the slice the
	// caller still holds. Appending past the result's length must not reach into
	// base's backing array, and mutating the result must not change base.
	got = append(got, ToolCall{ID: "e"})
	got[0].ID = "mutated"
	if len(base) != 2 || base[0].ID != "a" || base[1].ID != "b" {
		t.Errorf("base was mutated: %+v", base)
	}

	// Empty extra yields an independent copy of base, not an alias.
	copyOnly := AppendToolCalls(base)
	if len(copyOnly) != len(base) {
		t.Fatalf("copy len = %d, want %d", len(copyOnly), len(base))
	}
	copyOnly[0].ID = "mutated"
	if base[0].ID != "a" {
		t.Errorf("empty-extra copy aliased base: %+v", base)
	}
}
