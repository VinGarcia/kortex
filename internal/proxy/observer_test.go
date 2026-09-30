package proxy

import (
	"encoding/json"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/vingarcia/kortex/internal/protocol"
)

func TestObserver_disabledWithoutLogPath(t *testing.T) {
	obs := newObserver("", io.Discard)
	defer obs.Close()
	// Must observe (history state) without a file and without panicking.
	obs.Observe(protocol.ToBackend, []byte(`{"type":"user","message":{"role":"user","content":"oi"}}`))
	obs.Observe(protocol.FromBackend, []byte("garbage"))
	if obs.file != nil {
		t.Error("observer opened a file with no log path")
	}
}

func TestObserver_writesStructuredLog(t *testing.T) {
	rawLog := t.TempDir() + "/traffic.log"
	obs := newObserver(rawLog, io.Discard)
	defer obs.Close()

	lines := []struct {
		direction protocol.Direction
		line      string
	}{
		{protocol.ToBackend, `{"type":"user","message":{"role":"user","content":"oi"},"session_id":"s1"}`},
		{protocol.FromBackend, `not json`},
		{protocol.FromBackend, `{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"olá!"}]},"session_id":"s1"}`},
		{protocol.FromBackend, `{"type":"result","is_error":false,"stop_reason":"end_turn","session_id":"s1"}`},
	}
	for _, l := range lines {
		obs.Observe(l.direction, []byte(l.line))
	}

	data, err := os.ReadFile(structuredLogPath(rawLog))
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
	if len(entries) != len(lines) {
		t.Fatalf("got %d entries, want %d:\n%s", len(entries), len(lines), data)
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
	rawLog := t.TempDir() + "/traffic.log"
	obs := newObserver(rawLog, io.Discard)
	defer obs.Close()

	userLine, err := json.Marshal(map[string]any{
		"type":    "user",
		"message": map[string]any{"role": "user", "content": long},
	})
	if err != nil {
		t.Fatal(err)
	}
	obs.Observe(protocol.ToBackend, userLine)
	obs.Observe(protocol.FromBackend, []byte(`{"type":"result","is_error":false}`))

	data, err := os.ReadFile(structuredLogPath(rawLog))
	if err != nil {
		t.Fatal(err)
	}
	logLines := strings.Split(strings.TrimSpace(string(data)), "\n")
	var last entry
	if err := json.Unmarshal([]byte(logLines[len(logLines)-1]), &last); err != nil {
		t.Fatal(err)
	}
	if last.Snapshot == nil {
		t.Fatal("no snapshot on result entry")
	}
	got := last.Snapshot.Turns[0].UserText
	if len(got) >= len(long) {
		t.Errorf("snapshot user text not truncated: %d bytes", len(got))
	}
	if !strings.HasSuffix(got, "…") {
		t.Errorf("truncated text missing ellipsis: %q", got[len(got)-12:])
	}
	if !json.Valid([]byte(logLines[len(logLines)-1])) {
		t.Error("snapshot entry is not valid JSON after truncation")
	}
}
