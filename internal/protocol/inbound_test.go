package protocol

import "testing"

// infoBlock renders a Conversation info block exactly as OpenClaw emits it
// (formatContextJsonBlock: header line, ```json fence, single-line JSON, closing
// fence). Using "```" in a double-quoted literal keeps the fence out of a raw
// string, which cannot contain backticks.
func infoBlock(json string) string {
	return conversationInfoHeader + "\n" + "```json" + "\n" + json + "\n" + "```"
}

func TestDecodeInbound(t *testing.T) {
	historyBlock := "Recent chat history: " + inboundContextMarker + "\n" +
		"[Sun 2026-10-04 20:33] Vinícius: contei uma novidade ontem"

	t.Run("full inbound: typed info, echo, and new message", func(t *testing.T) {
		info := `{"chat_id":"c1","message_id":"4800","reply_to_id":"4799",` +
			`"sender":{"name":"Vinícius","username":"vini"},` +
			`"timestamp":"Sun 2026-10-04 14:17","inbound_event_kind":"message"}`
		text := infoBlock(info) + "\n\n" + historyBlock + "\n\n" + "roda o script de novo"

		got := DecodeInbound(text)
		if !got.HasInfo {
			t.Fatalf("HasInfo = false, want true")
		}
		if got.Info.ChatID != "c1" || got.Info.MessageID != "4800" || got.Info.ReplyToID != "4799" {
			t.Errorf("ids = %+v, want chat c1 / msg 4800 / reply 4799", got.Info)
		}
		if got.Info.Sender.Name != "Vinícius" || got.Info.Sender.Username != "vini" {
			t.Errorf("sender = %+v, want Vinícius/vini", got.Info.Sender)
		}
		if got.Info.Timestamp != "Sun 2026-10-04 14:17" || got.Info.InboundEventKind != "message" {
			t.Errorf("timestamp/kind = %q/%q", got.Info.Timestamp, got.Info.InboundEventKind)
		}
		if got.NewMessage != "roda o script de novo" {
			t.Errorf("NewMessage = %q, want the new message", got.NewMessage)
		}
		// The echo must carry every context block verbatim (bootstrap depends on
		// nothing being lost) and must NOT carry the new message.
		wantEcho := infoBlock(info) + "\n\n" + historyBlock
		if got.Echo != wantEcho {
			t.Errorf("Echo = %q, want %q", got.Echo, wantEcho)
		}
	})

	t.Run("no echo: whole text is the new message", func(t *testing.T) {
		got := DecodeInbound("oi tudo\n\nroda o script")
		if got.HasInfo || got.Echo != "" {
			t.Errorf("want no info and empty echo, got HasInfo=%v Echo=%q", got.HasInfo, got.Echo)
		}
		if got.NewMessage != "oi tudo\n\nroda o script" {
			t.Errorf("NewMessage = %q, want whole text", got.NewMessage)
		}
	})

	t.Run("echo without a Conversation info block: HasInfo false, echo kept", func(t *testing.T) {
		text := historyBlock + "\n\n" + "mensagem nova"
		got := DecodeInbound(text)
		if got.HasInfo {
			t.Errorf("HasInfo = true, want false (no info block present)")
		}
		if got.Echo != historyBlock {
			t.Errorf("Echo = %q, want the history block", got.Echo)
		}
		if got.NewMessage != "mensagem nova" {
			t.Errorf("NewMessage = %q", got.NewMessage)
		}
	})

	t.Run("malformed info JSON degrades to no typed info, echo intact", func(t *testing.T) {
		text := infoBlock("{not valid json") + "\n\n" + "mensagem nova"
		got := DecodeInbound(text)
		if got.HasInfo {
			t.Errorf("HasInfo = true, want false on malformed JSON")
		}
		if got.NewMessage != "mensagem nova" {
			t.Errorf("NewMessage = %q", got.NewMessage)
		}
		if got.Echo == "" {
			t.Errorf("Echo must stay intact even when info does not parse")
		}
	})

	t.Run("info with omitted keys decodes to zero values", func(t *testing.T) {
		text := infoBlock(`{"message_id":"42","inbound_event_kind":"message"}`) + "\n\n" + "oi"
		got := DecodeInbound(text)
		if !got.HasInfo {
			t.Fatalf("HasInfo = false, want true")
		}
		if got.Info.MessageID != "42" || got.Info.ReplyToID != "" || got.Info.Sender.Name != "" {
			t.Errorf("info = %+v, want only message_id/kind set", got.Info)
		}
	})
}

func TestSuspectedEchoDrift(t *testing.T) {
	tests := []struct {
		name       string
		newMessage string
		want       bool
	}{
		{
			name:       "plain new message is not drift",
			newMessage: "roda o script de novo\n\nmais um parágrafo normal",
			want:       false,
		},
		{
			name:       "marker leaked into the new message (stopped collapsing)",
			newMessage: "Recent chat history: " + inboundContextMarker + "\n[Sun 2026-10-04 20:33] Vinícius: oi",
			want:       true,
		},
		{
			name:       "renamed marker but Conversation info label survives",
			newMessage: "Conversation info: ⟦openclaw:context⟧\n```json\n{\"chat_id\":\"c1\"}\n```",
			want:       true,
		},
		{
			name:       "renamed marker but Recent chat history label survives",
			newMessage: "Recent chat history: ⟦openclaw:context⟧\n[Sun 2026-10-04 20:33] Vinícius: oi",
			want:       true,
		},
		{
			name:       "json fence alone is a fingerprint",
			newMessage: "algo\n```json\n{\"x\":1}\n```",
			want:       true,
		},
		{
			name:       "mass heuristic: four lines, two timestamped",
			newMessage: "[Sun 2026-10-04 20:30] Vini: um\n[Sun 2026-10-04 20:31] Vini: dois\ntexto solto\nmais texto",
			want:       true,
		},
		{
			name:       "three timestamped lines are below the line floor",
			newMessage: "[Sun 2026-10-04 20:30] Vini: um\n[Sun 2026-10-04 20:31] Vini: dois\n[Sun 2026-10-04 20:32] Vini: tres",
			want:       false,
		},
		{
			name:       "four lines but only one timestamped is not drift",
			newMessage: "[Sun 2026-10-04 20:30] Vini: um\nlinha dois\nlinha tres\nlinha quatro",
			want:       false,
		},
		{
			name:       "bracketed prefixes without a digit do not count as timestamps",
			newMessage: "[nota] uma\n[aside] duas\n[todo] tres\n[fim] quatro",
			want:       false,
		},
		{
			name:       "long legitimate multi-paragraph message stays silent",
			newMessage: "Oi, tudo bem? Queria te contar uma coisa longa.\n\nPrimeiro ponto que é importante.\n\nSegundo ponto que também importa.\n\nTerceiro ponto para fechar o raciocínio.",
			want:       false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := Inbound{NewMessage: test.newMessage}.SuspectedEchoDrift()
			if got != test.want {
				t.Errorf("SuspectedEchoDrift() = %v, want %v for %q", got, test.want, test.newMessage)
			}
		})
	}
}

// TestNewMessageSegment_SharesSplit guards that NewMessageSegment and
// DecodeInbound agree on the new-message boundary (they share splitInbound), so
// the proxy annotator's guard and the native decode never diverge.
func TestNewMessageSegment_SharesSplit(t *testing.T) {
	text := "a " + inboundContextMarker + " one\n\nb " + inboundContextMarker + " two\n\nnova"
	if seg := NewMessageSegment(text); seg != DecodeInbound(text).NewMessage {
		t.Errorf("NewMessageSegment = %q, DecodeInbound.NewMessage = %q; must match", seg, DecodeInbound(text).NewMessage)
	}
}
