package proxy

import (
	"encoding/json"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/vingarcia/kortex/internal/protocol"
)

// wireLine ties a raw fixture (F0 shapes) to the direction it travels.
type wireLine struct {
	direction protocol.Direction
	line      string
}

func observeAll(obs *observer, lines []wireLine) {
	for _, l := range lines {
		obs.Observe(l.direction, protocol.Parse([]byte(l.line)))
	}
}

func TestObserver_disabledWithoutLogPath(t *testing.T) {
	obs := newObserver("", io.Discard, nil)
	defer obs.Close()
	// Must observe (history state) without a file and without panicking.
	obs.Observe(protocol.ToBackend, protocol.Parse([]byte(`{"type":"user","message":{"role":"user","content":"oi"}}`)))
	obs.Observe(protocol.FromBackend, protocol.Parse([]byte("garbage")))
	if obs.file != nil {
		t.Error("observer opened a file with no log path")
	}
}

// TestObserver_reconstructsMultiTurnSession drives the observer with the
// full wire shape of a 3-turn session (F0 fixtures) and checks the
// reconstructed canonical history in the logged snapshot.
func TestObserver_reconstructsMultiTurnSession(t *testing.T) {
	eventLog := t.TempDir() + "/traffic.log.events"
	obs := newObserver(eventLog, io.Discard, nil)
	defer obs.Close()

	observeAll(obs, []wireLine{
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

	entries := readEntries(t, eventLog)
	var lastSnapshot *entry
	for i := range entries {
		if entries[i].Snapshot != nil {
			lastSnapshot = &entries[i]
		}
	}
	if lastSnapshot == nil {
		t.Fatal("no snapshot entry in the structured log")
	}
	snap := *lastSnapshot.Snapshot
	if snap.SessionID != "sess-1" {
		t.Errorf("SessionID = %q, want sess-1", snap.SessionID)
	}
	// The last snapshot is taken at the turn-2 result, so it holds 2 turns.
	if len(snap.Turns) != 2 {
		t.Fatalf("got %d turns, want 2: %+v", len(snap.Turns), snap.Turns)
	}
	turn1 := snap.Turns[0]
	if turn1.UserText != "oi, tudo bem?" || turn1.AssistantText != "Tudo ótimo!" ||
		!turn1.Completed || turn1.StopReason != "end_turn" || len(turn1.ToolCalls) != 0 {
		t.Errorf("turn 1 = %+v", turn1)
	}
	turn2 := snap.Turns[1]
	if turn2.UserText != "lista os arquivos" || turn2.AssistantText != "Tem main.go e README.md." || !turn2.Completed {
		t.Errorf("turn 2 = %+v", turn2)
	}
	if len(turn2.ToolCalls) != 1 {
		t.Fatalf("turn 2 tool calls = %+v", turn2.ToolCalls)
	}
	call := turn2.ToolCalls[0]
	if call.ID != "toolu_01" || call.Name != "Bash" || call.Input != `{"command":"ls"}` || call.Result != "main.go\nREADME.md" {
		t.Errorf("tool call = %+v", call)
	}
}

func TestObserver_writesStructuredLog(t *testing.T) {
	eventLog := t.TempDir() + "/traffic.log.events"
	obs := newObserver(eventLog, io.Discard, nil)
	defer obs.Close()

	observeAll(obs, []wireLine{
		{protocol.ToBackend, `{"type":"user","message":{"role":"user","content":"oi"},"session_id":"s1"}`},
		{protocol.FromBackend, `not json`},
		{protocol.FromBackend, `{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"olá!"}]},"session_id":"s1"}`},
		{protocol.FromBackend, `{"type":"result","is_error":false,"stop_reason":"end_turn","session_id":"s1"}`},
	})

	entries := readEntries(t, eventLog)
	if len(entries) != 4 {
		t.Fatalf("got %d entries, want 4: %+v", len(entries), entries)
	}
	wantTypes := []string{"user", "unknown", "assistant", "result"}
	wantDirs := []string{"in", "out", "out", "out"}
	for i, e := range entries {
		if e.Type != wantTypes[i] || e.Direction != wantDirs[i] {
			t.Errorf("entry %d = {dir:%s type:%s}, want {dir:%s type:%s}", i, e.Direction, e.Type, wantDirs[i], wantTypes[i])
		}
	}
	last := entries[len(entries)-1]
	if last.Snapshot == nil {
		t.Fatal("result entry has no history snapshot")
	}
	if len(last.Snapshot.Turns) != 1 {
		t.Fatalf("snapshot turns = %+v", last.Snapshot.Turns)
	}
	turn := last.Snapshot.Turns[0]
	if turn.UserText != "oi" || turn.AssistantText != "olá!" || !turn.Completed {
		t.Errorf("snapshot turn = %+v", turn)
	}
}

func TestObserver_truncatesSnapshotFields(t *testing.T) {
	long := strings.Repeat("á", 400) // multi-byte: exercises the rune-safe cut
	eventLog := t.TempDir() + "/traffic.log.events"
	obs := newObserver(eventLog, io.Discard, nil)
	defer obs.Close()

	userLine, err := json.Marshal(map[string]any{
		"type":    "user",
		"message": map[string]any{"role": "user", "content": long},
	})
	if err != nil {
		t.Fatal(err)
	}
	obs.Observe(protocol.ToBackend, protocol.Parse(userLine))
	obs.Observe(protocol.FromBackend, protocol.Parse([]byte(`{"type":"result","is_error":false}`)))

	entries := readEntries(t, eventLog)
	last := entries[len(entries)-1]
	if last.Snapshot == nil {
		t.Fatal("no snapshot on result entry")
	}
	got := last.Snapshot.Turns[0].UserText
	if len(got) >= len(long) {
		t.Errorf("snapshot user text not truncated: %d bytes", len(got))
	}
	if !strings.HasSuffix(got, "…") {
		t.Errorf("truncated text missing ellipsis: %q", got)
	}
}

func readEntries(t *testing.T, path string) []entry {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var entries []entry
	for _, raw := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var e entry
		if err := json.Unmarshal([]byte(raw), &e); err != nil {
			t.Fatalf("structured log line is not JSON: %q: %v", raw, err)
		}
		entries = append(entries, e)
	}
	return entries
}

// recordingTurnEvaluator captures each turn-completion notification.
type recordingTurnEvaluator struct {
	indexes []int
	texts   []string
}

func (r *recordingTurnEvaluator) EvaluateCompletedTurn(turnIndex int, assistantText string) {
	r.indexes = append(r.indexes, turnIndex)
	r.texts = append(r.texts, assistantText)
}

// TestObserver_notifiesTurnEvaluatorOncePerCompletedTurn checks the F2c
// hook: each terminal result fires exactly one notification with the turn's
// full final assistant text, and a stray duplicate result fires nothing.
func TestObserver_notifiesTurnEvaluatorOncePerCompletedTurn(t *testing.T) {
	evaluator := &recordingTurnEvaluator{}
	obs := newObserver("", io.Discard, evaluator)
	defer obs.Close()

	observeAll(obs, []wireLine{
		{protocol.ToBackend, `{"type":"user","message":{"role":"user","content":"oi"},"uuid":"u1","session_id":"s1"}`},
		{protocol.FromBackend, `{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"primeira parte"}]},"session_id":"s1"}`},
		{protocol.FromBackend, `{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"segunda parte"}]},"session_id":"s1"}`},
		{protocol.FromBackend, `{"type":"result","subtype":"success","is_error":false,"stop_reason":"end_turn","session_id":"s1"}`},
		// Stray duplicate result: no open turn, must not re-notify.
		{protocol.FromBackend, `{"type":"result","subtype":"success","is_error":false,"stop_reason":"end_turn","session_id":"s1"}`},
		{protocol.ToBackend, `{"type":"user","message":{"role":"user","content":"mais um"},"uuid":"u2","session_id":"s1"}`},
		{protocol.FromBackend, `{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"segunda resposta"}]},"session_id":"s1"}`},
		{protocol.FromBackend, `{"type":"result","subtype":"success","is_error":false,"stop_reason":"end_turn","session_id":"s1"}`},
	})

	if len(evaluator.indexes) != 2 {
		t.Fatalf("got %d notifications, want 2: %v", len(evaluator.indexes), evaluator.indexes)
	}
	if evaluator.indexes[0] != 0 || evaluator.texts[0] != "primeira parte\nsegunda parte" {
		t.Errorf("first notification = (%d, %q)", evaluator.indexes[0], evaluator.texts[0])
	}
	if evaluator.indexes[1] != 1 || evaluator.texts[1] != "segunda resposta" {
		t.Errorf("second notification = (%d, %q)", evaluator.indexes[1], evaluator.texts[1])
	}
}
