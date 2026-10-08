package session

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/tools"
)

// TestRun_InboundEchoBootstrapThenDiscard proves the DIRETIVA #4803 invariant at
// the Run seam: the FIRST turn of a fresh session (empty canonical history)
// preserves the channel echo as marked bootstrap history, and the SECOND turn
// (canonical history now non-empty) discards the echo, forwarding only the new
// message plus the meta line.
func TestRun_InboundEchoBootstrapThenDiscard(t *testing.T) {
	server, rec := textResponder(t, "ok")
	client := anthropic.NewClient("secret", server.URL, time.Second)

	marker := "⟦openclaw:ctx⟧"
	infoBlock := func(json string) string {
		return "Conversation info: " + marker + "\n" + "```json" + "\n" + json + "\n" + "```"
	}
	history := "Recent chat history: " + marker + "\n[Sun 2026-10-04 20:33] Vinícius: contei algo ontem"

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

	err := Run(context.Background(), Config{
		Stdin: strings.NewReader(in), Stdout: &out, Client: client,
		Dispatcher: tools.NewDispatcher(), Store: NewStore(t.TempDir()),
		SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(rec.userTexts) != 2 {
		t.Fatalf("got %d core calls, want 2: %q", len(rec.userTexts), rec.userTexts)
	}

	first, second := rec.userTexts[0], rec.userTexts[1]

	// First turn: bootstrap. The echo is preserved under the marker, nothing lost.
	if !strings.Contains(first, bootstrapEchoOpen) || !strings.Contains(first, "contei algo ontem") {
		t.Errorf("first turn must preserve echo as bootstrap history:\n%s", first)
	}
	if !strings.Contains(first, "roda o script") || !strings.Contains(first, "respondendo à msg #4799") {
		t.Errorf("first turn must still carry the new message + meta:\n%s", first)
	}

	// Second turn: discard. No bootstrap block, no echoed prior-turn text, just
	// the new message + its meta line.
	if strings.Contains(second, bootstrapEchoOpen) || strings.Contains(second, "contei algo ontem") || strings.Contains(second, marker) {
		t.Errorf("second turn must discard the echo entirely:\n%s", second)
	}
	wantSecond := "mais uma coisa\n\n[meta: respondendo à msg #4800; de Vinícius; 14:20]"
	if second != wantSecond {
		t.Errorf("second turn forwarded =\n%q\nwant\n%q", second, wantSecond)
	}
}
