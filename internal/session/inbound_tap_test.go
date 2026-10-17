package session

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/tools"
)

// tapEntry mirrors the JSONL record tapInbound writes, so tests decode it back
// and assert on the fields rather than on the serialized bytes.
type tapEntry struct {
	TS  string `json:"ts"`
	Raw string `json:"raw"`
}

// TestRun_InboundTapCapturesRawPreDecode proves the #5368 invariant: when an
// InboundTap is wired, Run records the RAW channel text of every genuine inbound
// user message — exactly as OpenClaw sent it, BEFORE DecodeInbound strips the
// echo — one self-delimiting JSONL line per message. The echo a resumed turn
// discards is therefore still auditable in the tap, even though it never reaches
// the core (rec.userTexts) or any other log.
func TestRun_InboundTapCapturesRawPreDecode(t *testing.T) {
	server, rec := textResponder(t, "ok")
	client := anthropic.NewClient("secret", server.URL, time.Second)

	marker := "⟦openclaw:ctx⟧"
	infoBlock := func(json string) string {
		return "Conversation info: " + marker + "\n" + "```json" + "\n" + json + "\n" + "```"
	}
	history := "Recent chat history: " + marker + "\n[Sun 2026-10-04 20:33] Vinícius: contei algo ontem"

	// turn1 carries a multi-line echo (info block + history + new message); turn2
	// is a plain resumed turn whose echo would be discarded before any other log.
	turn1 := infoBlock(`{"message_id":"4800","reply_to_id":"4799","sender":{"name":"Vinícius"},"timestamp":"14:17","inbound_event_kind":"message"}`) +
		"\n\n" + history + "\n\n" + "roda o script"
	turn2 := infoBlock(`{"message_id":"4801","reply_to_id":"4800","sender":{"name":"Vinícius"},"timestamp":"14:20","inbound_event_kind":"message"}`) +
		"\n\n" + "mais uma coisa"

	in := strings.Join([]string{
		initializeLine("sys"),
		userLine("sess-1", turn1),
		userLine("sess-1", turn2),
	}, "\n") + "\n"

	var out strings.Builder
	var tap bytes.Buffer

	err := Run(context.Background(), Config{
		Stdin: strings.NewReader(in), Stdout: &out, Client: client,
		Dispatcher: tools.NewDispatcher(), Store: NewStore(t.TempDir()),
		SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		InboundTap: &tap,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// One self-delimiting JSONL line per inbound user message, nothing more.
	lines := strings.Split(strings.TrimRight(tap.String(), "\n"), "\n")
	if len(lines) != 2 {
		t.Fatalf("tap wrote %d lines, want 2 (one per inbound message):\n%s", len(lines), tap.String())
	}

	var entries []tapEntry
	for i, line := range lines {
		var e tapEntry
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Fatalf("tap line %d is not valid JSON: %v\nline: %q", i, err, line)
		}
		if _, err := time.Parse(time.RFC3339Nano, e.TS); err != nil {
			t.Errorf("tap line %d ts %q not RFC3339Nano: %v", i, e.TS, err)
		}
		entries = append(entries, e)
	}

	// The tap captures the raw text verbatim, pre-decode: turn1's full echo
	// (marker + prior-turn body) survives, exactly the bytes OpenClaw sent.
	if entries[0].Raw != turn1 {
		t.Errorf("tap[0].Raw =\n%q\nwant\n%q", entries[0].Raw, turn1)
	}
	if entries[1].Raw != turn2 {
		t.Errorf("tap[1].Raw =\n%q\nwant\n%q", entries[1].Raw, turn2)
	}

	// Pre-decode, not the forwarded text: the resumed turn's tap entry still holds
	// the echo marker and the prior-turn body that the core (rec.userTexts) never
	// saw, which is the whole point of the tap.
	if !strings.Contains(entries[1].Raw, marker) || !strings.Contains(entries[1].Raw, "message_id") {
		t.Errorf("tap[1].Raw must preserve the raw (pre-decode) channel text:\n%s", entries[1].Raw)
	}
	if len(rec.userTexts) == 2 && strings.Contains(rec.userTexts[1], marker) {
		t.Errorf("sanity: the forwarded (decoded) text should NOT contain the marker:\n%s", rec.userTexts[1])
	}
}

// TestRun_InboundTapNilNoChange proves the no-log path: a nil InboundTap leaves
// Run's behavior untouched — the turns still run and forward correctly, with no
// panic and no tap output to assert.
func TestRun_InboundTapNilNoChange(t *testing.T) {
	server, rec := textResponder(t, "ok")
	client := anthropic.NewClient("secret", server.URL, time.Second)

	in := strings.Join([]string{
		initializeLine("sys"),
		userLine("sess-1", "oi"),
	}, "\n") + "\n"

	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin: strings.NewReader(in), Stdout: &out, Client: client,
		Dispatcher: tools.NewDispatcher(), Store: NewStore(t.TempDir()),
		SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		InboundTap: nil,
	})
	if err != nil {
		t.Fatalf("Run with nil InboundTap: %v", err)
	}
	if len(rec.userTexts) != 1 || !strings.Contains(rec.userTexts[0], "oi") {
		t.Fatalf("nil tap must not change forwarding, got %q", rec.userTexts)
	}
}

// TestTapInbound proves the serialization contract directly: a nil writer is a
// no-op (no panic), and a multi-line raw message is written as exactly one JSONL
// record that round-trips back to the original text verbatim.
func TestTapInbound(t *testing.T) {
	// nil writer: no-op, no panic.
	tapInbound(nil, "anything")

	raw := "linha um\nlinha dois\n\nlinha quatro com \"aspas\" e emoji ⟦ctx⟧"
	var buf bytes.Buffer
	tapInbound(&buf, raw)

	// Exactly one record: a single trailing newline, no interior line split even
	// though raw itself is multi-line (JSON escaping keeps it on one line).
	if got := strings.Count(buf.String(), "\n"); got != 1 {
		t.Fatalf("want exactly one JSONL record (one newline), got %d:\n%q", got, buf.String())
	}
	var e tapEntry
	if err := json.Unmarshal([]byte(strings.TrimRight(buf.String(), "\n")), &e); err != nil {
		t.Fatalf("tap output not valid JSON: %v", err)
	}
	if e.Raw != raw {
		t.Errorf("round-trip Raw =\n%q\nwant\n%q", e.Raw, raw)
	}
	if e.TS == "" {
		t.Errorf("tap entry missing ts")
	}
}
