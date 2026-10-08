package session

import (
	"bytes"
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
	// A long, multi-paragraph, markerless message: the shape most likely to trip a
	// size-only drift heuristic, so it proves warnOnEchoDrift stays silent on it.
	turn3 := "Pensei bastante sobre o problema e queria registrar o raciocínio completo.\n\n" +
		"Primeiro, a abordagem atual tem um gargalo claro na etapa de ingestão.\n\n" +
		"Segundo, dá pra paralelizar sem mudar o contrato externo.\n\n" +
		"Terceiro, o custo cai bastante se a gente cachear o passo intermediário."

	in := strings.Join([]string{
		initializeLine("sys"),
		userLine("sess-1", turn1),
		userLine("sess-1", turn2),
		userLine("sess-1", turn3),
	}, "\n") + "\n"
	var out strings.Builder
	var diag bytes.Buffer

	err := Run(context.Background(), Config{
		Stdin: strings.NewReader(in), Stdout: &out, Client: client,
		Dispatcher: tools.NewDispatcher(), Store: NewStore(t.TempDir()),
		SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		Diag: &diag,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(rec.userTexts) != 3 {
		t.Fatalf("got %d core calls, want 3: %q", len(rec.userTexts), rec.userTexts)
	}

	// No false positive: a healthy collapsed echo (turn 2) and a long markerless
	// message (turn 3) must never trip the drift warning.
	if diag.Len() != 0 {
		t.Errorf("drift warning must stay silent on healthy echo and long markerless message, got:\n%s", diag.String())
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

// TestRun_InboundEchoDriftRenamedMarker is the negative control: if OpenClaw
// RENAMES the inbound-context marker, DecodeInbound strips nothing, the whole echo
// re-inflates canonical history, and the only defense is the drift warning. The
// test documents the exact harm by asserting BOTH that the warning fired and that
// the echoed prior turn actually leaked back into the core's input.
func TestRun_InboundEchoDriftRenamedMarker(t *testing.T) {
	server, rec := textResponder(t, "ok")
	client := anthropic.NewClient("secret", server.URL, time.Second)

	renamed := "⟦openclaw:context⟧"
	infoBlock := func(json string) string {
		return "Conversation info: " + renamed + "\n" + "```json" + "\n" + json + "\n" + "```"
	}
	driftedEcho := infoBlock(`{"message_id":"4801","reply_to_id":"4800","sender":{"name":"Vinícius"},"timestamp":"14:20"}`) +
		"\n\n" + "Recent chat history: " + renamed + "\n[Sun 2026-10-04 20:33] Vinícius: contei algo ontem" +
		"\n\n" + "roda o script"

	in := strings.Join([]string{
		initializeLine("sys"),
		userLine("sess-1", "oi, primeira mensagem"),
		userLine("sess-1", driftedEcho),
	}, "\n") + "\n"
	var out strings.Builder
	var diag bytes.Buffer

	err := Run(context.Background(), Config{
		Stdin: strings.NewReader(in), Stdout: &out, Client: client,
		Dispatcher: tools.NewDispatcher(), Store: NewStore(t.TempDir()),
		SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		Diag: &diag,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(rec.userTexts) != 2 {
		t.Fatalf("got %d core calls, want 2: %q", len(rec.userTexts), rec.userTexts)
	}

	if !strings.Contains(diag.String(), "wire-drift") {
		t.Errorf("renamed marker must raise the drift warning, got:\n%s", diag.String())
	}
	// The harm the warning witnesses: the echoed prior turn re-inflated the core's
	// input because the renamed marker defeated the split.
	if !strings.Contains(rec.userTexts[1], "contei algo ontem") {
		t.Errorf("expected the echo to re-inflate into the core input, got:\n%s", rec.userTexts[1])
	}
	if !strings.Contains(diag.String(), "marker_present=false") {
		t.Errorf("a renamed (absent) marker must log marker_present=false, got:\n%s", diag.String())
	}
}

// TestRun_InboundEchoDriftUncollapsedBody is the second drift case: the marker is
// intact but OpenClaw stopped collapsing an echoed turn's body, so part of the
// echo crosses a blank line and leaks past the split into the new message. The
// warning fires on the leftover echo shape and logs marker_present=true, which
// distinguishes this case from a rename at a glance.
func TestRun_InboundEchoDriftUncollapsedBody(t *testing.T) {
	server, rec := textResponder(t, "ok")
	client := anthropic.NewClient("secret", server.URL, time.Second)

	marker := "⟦openclaw:ctx⟧"
	infoBlock := func(json string) string {
		return "Conversation info: " + marker + "\n" + "```json" + "\n" + json + "\n" + "```"
	}
	// The history block stays collapsed, but a second uncollapsed paragraph of
	// echoed turns crosses a blank line and so lands after the last marker block.
	driftedEcho := infoBlock(`{"message_id":"4801","reply_to_id":"4800","sender":{"name":"Vinícius"},"timestamp":"14:20"}`) +
		"\n\n" + "Recent chat history: " + marker + "\n[Sun 2026-10-04 20:30] Vinícius: linha um" +
		"\n\n" + "[Sun 2026-10-04 20:33] Vinícius: linha quatro que vazou\n[Sun 2026-10-04 20:34] Vinícius: linha cinco que vazou" +
		"\n\n" + "mensagem nova real"

	in := strings.Join([]string{
		initializeLine("sys"),
		userLine("sess-1", "oi, primeira mensagem"),
		userLine("sess-1", driftedEcho),
	}, "\n") + "\n"
	var out strings.Builder
	var diag bytes.Buffer

	err := Run(context.Background(), Config{
		Stdin: strings.NewReader(in), Stdout: &out, Client: client,
		Dispatcher: tools.NewDispatcher(), Store: NewStore(t.TempDir()),
		SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		Diag: &diag,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(rec.userTexts) != 2 {
		t.Fatalf("got %d core calls, want 2: %q", len(rec.userTexts), rec.userTexts)
	}

	if !strings.Contains(diag.String(), "wire-drift") {
		t.Errorf("an uncollapsed echoed body must raise the drift warning, got:\n%s", diag.String())
	}
	if !strings.Contains(diag.String(), "marker_present=true") {
		t.Errorf("an intact marker must log marker_present=true, got:\n%s", diag.String())
	}
}
