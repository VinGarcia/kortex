package session

import (
	"strings"
	"testing"

	"github.com/vingarcia/kortex/internal/protocol"
)

// TestComposeInboundUserText_BootstrapGuard is the DIRETIVA #4803 correctness
// invariant: parts 2 (discard) and 4 (bootstrap) are coupled through
// canonicalEmpty. Empty canonical context must PRESERVE the echo as marked
// bootstrap history (nothing lost); non-empty must DISCARD it, forwarding only
// the new message plus the meta line.
func TestComposeInboundUserText_BootstrapGuard(t *testing.T) {
	inbound := protocol.Inbound{
		HasInfo:    true,
		Info:       protocol.InboundInfo{ReplyToID: "4799", Sender: protocol.InboundSender{Name: "Vinícius"}, Timestamp: "Sun 2026-10-04 14:17"},
		Echo:       "Recent chat history: ⟦openclaw:ctx⟧\n[Sun 2026-10-04 20:33] Vinícius: contei uma novidade",
		NewMessage: "roda o script de novo",
	}

	t.Run("empty canonical context preserves echo as marked bootstrap history", func(t *testing.T) {
		got := composeInboundUserText(inbound, true, nil)

		if !strings.Contains(got, bootstrapEchoOpen) || !strings.Contains(got, bootstrapEchoClose) {
			t.Fatalf("bootstrap markers missing:\n%s", got)
		}
		// Nothing lost: the full echo text survives between the markers.
		if !strings.Contains(got, inbound.Echo) {
			t.Errorf("echo text not preserved in bootstrap output:\n%s", got)
		}
		// The new message and meta still ride along.
		if !strings.Contains(got, inbound.NewMessage) {
			t.Errorf("new message missing from bootstrap output:\n%s", got)
		}
		if !strings.Contains(got, "respondendo à msg #4799") {
			t.Errorf("meta line missing from bootstrap output:\n%s", got)
		}
		// Order: the previous-conversation block comes before the new message.
		if strings.Index(got, bootstrapEchoOpen) > strings.Index(got, inbound.NewMessage) {
			t.Errorf("bootstrap history must precede the new message:\n%s", got)
		}
	})

	t.Run("non-empty canonical context discards echo, forwards new message + meta", func(t *testing.T) {
		got := composeInboundUserText(inbound, false, nil)

		if strings.Contains(got, bootstrapEchoOpen) || strings.Contains(got, "contei uma novidade") {
			t.Errorf("echo must be discarded when canonical context is non-empty:\n%s", got)
		}
		want := "roda o script de novo\n\n[meta: respondendo à msg #4799; de Vinícius; Sun 2026-10-04 14:17]"
		if got != want {
			t.Errorf("discard output =\n%q\nwant\n%q", got, want)
		}
	})
}

// TestComposeInboundUserText_EvaluatorSeesOnlyNewMessage proves part 3: the
// annotator (the emotion evaluator's seam) is handed ONLY the genuinely new
// message, never the echo — on both the discard and the bootstrap path.
func TestComposeInboundUserText_EvaluatorSeesOnlyNewMessage(t *testing.T) {
	inbound := protocol.Inbound{
		HasInfo:    true,
		Info:       protocol.InboundInfo{ReplyToID: "4799", Sender: protocol.InboundSender{Name: "Vinícius"}, Timestamp: "14:17"},
		Echo:       "Recent chat history: ⟦openclaw:ctx⟧\n[Sun 2026-10-04 20:33] Vinícius: contei uma novidade",
		NewMessage: "roda o script de novo",
	}

	for _, canonicalEmpty := range []bool{true, false} {
		ann := &fakeAnnotator{rewrite: "ANNOTATED"}
		got := composeInboundUserText(inbound, canonicalEmpty, ann)

		if len(ann.seen) != 1 || ann.seen[0] != inbound.NewMessage {
			t.Fatalf("canonicalEmpty=%v: annotator saw %q, want only the new message %q",
				canonicalEmpty, ann.seen, inbound.NewMessage)
		}
		// The annotator's rewrite — not the raw new message — is what gets forwarded.
		if !strings.Contains(got, "ANNOTATED") || strings.Contains(got, inbound.NewMessage) {
			t.Errorf("canonicalEmpty=%v: annotated text not forwarded:\n%s", canonicalEmpty, got)
		}
	}
}

// TestComposeInboundUserText_NoInbound covers a plain message with no OpenClaw
// echo: no bootstrap block, no meta line, just the (annotated) message.
func TestComposeInboundUserText_NoInbound(t *testing.T) {
	inbound := protocol.Inbound{NewMessage: "oi tudo"}
	if got := composeInboundUserText(inbound, true, nil); got != "oi tudo" {
		t.Errorf("plain message output = %q, want %q", got, "oi tudo")
	}
}

func TestInboundMetaLine(t *testing.T) {
	tests := []struct {
		desc    string
		inbound protocol.Inbound
		want    string
	}{
		{
			desc:    "no info yields no meta",
			inbound: protocol.Inbound{HasInfo: false},
			want:    "",
		},
		{
			desc: "all fields present",
			inbound: protocol.Inbound{HasInfo: true, Info: protocol.InboundInfo{
				ReplyToID: "4799", Sender: protocol.InboundSender{Name: "Vinícius"}, Timestamp: "Sun 2026-10-04 14:17",
			}},
			want: "[meta: respondendo à msg #4799; de Vinícius; Sun 2026-10-04 14:17]",
		},
		{
			desc: "no reply-to drops that part",
			inbound: protocol.Inbound{HasInfo: true, Info: protocol.InboundInfo{
				Sender: protocol.InboundSender{Name: "Ana"}, Timestamp: "09:00",
			}},
			want: "[meta: de Ana; 09:00]",
		},
		{
			desc: "username used when name absent",
			inbound: protocol.Inbound{HasInfo: true, Info: protocol.InboundInfo{
				ReplyToID: "7", Sender: protocol.InboundSender{Username: "ana_bot"},
			}},
			want: "[meta: respondendo à msg #7; de ana_bot]",
		},
		{
			desc:    "info present but all fields empty yields no meta",
			inbound: protocol.Inbound{HasInfo: true},
			want:    "",
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			if got := inboundMetaLine(test.inbound); got != test.want {
				t.Errorf("inboundMetaLine = %q, want %q", got, test.want)
			}
		})
	}
}
