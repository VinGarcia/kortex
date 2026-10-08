package session

import (
	"io"
	"strings"

	"github.com/vingarcia/kortex/internal/protocol"
)

// The bootstrap-echo delimiters wrap the channel's echo of prior turns when it
// is preserved on the first turn of a fresh session (see composeInboundUserText).
// They mark the enclosed text as PREVIOUS-conversation context — analogous to a
// recent-memory block — so the core never mistakes it for canonical turns of the
// current session. The text is kortex's own convention (not the OpenClaw wire
// format), so it lives here, beside the other session-authored framing, rather
// than in package protocol.
const (
	bootstrapEchoOpen  = "<<<MENSAGENS_DA_CONVERSA_ANTERIOR (via echo do canal; não são deste turno)>>>"
	bootstrapEchoClose = "<<<FIM_MENSAGENS_DA_CONVERSA_ANTERIOR>>>"
)

// composeInboundUserText turns one decoded OpenClaw inbound message into the
// text that reaches the core model and the canonical history (DIRETIVA #4803).
// It always forwards the genuinely new message — annotated when an annotator is
// wired, so the emotion evaluator only ever sees the new message — followed by a
// compact meta line carrying the useful per-message metadata (reply-to, sender,
// timestamp).
//
// The channel's echo of prior turns is handled by canonicalEmpty, and the two
// cases are deliberately coupled:
//   - canonicalEmpty == true  (first turn of a fresh session): the echo is the
//     ONLY record of the prior conversation, so it is preserved as a clearly
//     marked bootstrap-history block ahead of the new message — never discarded.
//   - canonicalEmpty == false (any later turn): the canonical history already
//     holds those turns, so the echo is pure redundant cost (it rides the moving
//     tail and never caches, and it would re-inflate canonical history on every
//     turn) and is discarded.
//
// The discard is therefore gated strictly on canonicalEmpty being false; the
// empty case can only ever take the preserve path, so first-run history can
// never be silently lost.
func composeInboundUserText(inbound protocol.Inbound, canonicalEmpty bool, annotator Annotator) string {
	newMessage := inbound.NewMessage
	if annotator != nil {
		// Annotate only the new message: the echo (when present) is either
		// discarded or carried verbatim as bootstrap history, so annotating it
		// would re-bill tokens on content the core already has.
		newMessage = annotator.Annotate(newMessage)
	}

	parts := make([]string, 0, 3)
	if canonicalEmpty && inbound.Echo != "" {
		parts = append(parts, bootstrapEchoOpen+"\n"+inbound.Echo+"\n"+bootstrapEchoClose)
	}
	parts = append(parts, newMessage)
	if meta := inboundMetaLine(inbound); meta != "" {
		parts = append(parts, meta)
	}
	return strings.Join(parts, "\n\n")
}

// warnOnEchoDrift emits one observe-only WARN line when the OpenClaw inbound
// echo-discard path shows signs of wire drift — OpenClaw renamed the inbound
// context marker or stopped collapsing echoed turn bodies — which would otherwise
// silently re-inflate canonical history every turn, the exact harm DIRETIVA #4803
// prevents, with no signal. It only observes: it never alters what
// composeInboundUserText forwards, which is why it sits at the Run call-site
// rather than inside the pure compose.
//
// canonicalTurns is len(canonical history). A fresh session (0) preserves the echo
// as marked bootstrap rather than discarding it, so re-inflation cannot happen and
// no warning is due. marker_present reports whether OpenClaw's marker was still
// present in the raw payload (inbound.Echo != "", by splitInbound's invariant), so
// a renamed marker (absent, Echo empty) is told apart from an un-collapsed body
// (marker intact) at a glance; the body is never logged — the echo is large and
// carries user content.
func warnOnEchoDrift(diag io.Writer, sessionID string, inbound protocol.Inbound, canonicalTurns int) {
	if canonicalTurns == 0 || !inbound.SuspectedEchoDrift() {
		return
	}
	// TODO(metrics): increment inbound_echo_drift_total here once internal/ grows a
	// metrics seam; none exists today, so adding one now would be premature.
	newMessage := inbound.NewMessage
	diagf(diag,
		"session: wire-drift suspected in OpenClaw inbound echo-discard, canonical history may be silently re-inflating "+
			"(OpenClaw renamed the inbound-context marker or stopped collapsing echoed turn bodies) "+
			"session=%s canonical_turns=%d new_msg_bytes=%d new_msg_lines=%d marker_present=%t",
		sessionID, canonicalTurns, len(newMessage), strings.Count(newMessage, "\n")+1, inbound.Echo != "")
}

// inboundMetaLine renders the compact meta line appended after the new message:
// the useful per-message metadata the core should keep even once the echo is
// gone. It lists only the fields OpenClaw actually supplied, so a message that
// replies to nothing (or carries no sender/timestamp) drops those parts, and a
// message with no parseable info produces no meta line at all.
func inboundMetaLine(inbound protocol.Inbound) string {
	if !inbound.HasInfo {
		return ""
	}
	info := inbound.Info
	var parts []string
	if info.ReplyToID != "" {
		parts = append(parts, "respondendo à msg #"+info.ReplyToID)
	}
	// Name is preferred over Username for the sender label.
	if sender := protocol.FirstNonEmpty(info.Sender.Name, info.Sender.Username); sender != "" {
		parts = append(parts, "de "+sender)
	}
	if info.Timestamp != "" {
		parts = append(parts, info.Timestamp)
	}
	if len(parts) == 0 {
		return ""
	}
	return "[meta: " + strings.Join(parts, "; ") + "]"
}
