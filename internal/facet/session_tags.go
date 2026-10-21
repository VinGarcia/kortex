package facet

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// sessionTagsFacet names this producer's callLog entries distinctly from
// inputAnnotatorFacet (annotator.go): a session-tags write failure is a
// persistence problem, not an annotation problem, even though both are
// recorded through the same facetLogger/sink.
const sessionTagsFacet = "session_tags"

// This file is the Go port of the per-session emotion-tags PRODUCER that
// lives in sylphie's scripts/emotion-annotate.sh (the session_append shell
// function) + scripts/emotion-session-path.sh (the path+sanitization
// contract those scripts and hooks/memory-distill/DISTILL-RUN.md all share).
// Before this file existed, kortex's native input annotator tagged messages
// in-stream and in the facet log only; nothing durable survived per session
// for sylphie's memory-distill "cor emocional" step to read. This closes
// that gap: on every successful annotation, one JSONL line is appended to
// <SessionTagsDir>/<sanitized id>.jsonl, carrying the salient
// (highest-investment) paragraph of that message.
//
// Keying: kortex sessions are identified by the OpenClaw-assigned session id
// (passed as kortex's own --session-id/--resume argv — see
// main.go:nativeSessionID — and recorded verbatim as `sessionId` in
// sylphie's distill queue, hooks/memory-distill/DISTILL-RUN.md). This file
// keys the per-session file on THAT id, not on the bash producer's
// Telegram-style human-readable "sessionKey" (e.g.
// "agent:sylphie:telegram:direct:X"): a kortex session id is minted fresh
// per session and never reused, so keying on it sidesteps the reuse hazard
// DISTILL-RUN.md documents for sessionKey (a stable per-channel identity
// recycled across many distinct conversations over time) rather than merely
// papering over it with delete-after-read. DISTILL-RUN.md's "De onde vem a
// cor" section is updated to resolve the file by `sessionId` first (the
// kortex-native producer, this file), falling back to `sessionKey` (the
// legacy bash producer, still live for any session not routed through
// kortex) — both fields are already present on every distill-queue line
// regardless of which backend produced the session.
//
// No idempotency/dedup check is needed here (unlike the bash producer's
// session_append): the caller, InputAnnotator.annotateText, already declines
// to re-annotate — and therefore never reaches this function — for any
// message whose NEW segment already carries the "[emoções p" marker. That
// is exactly the resend/resume case the bash dedup guarded against, so
// kortex's upstream guard makes a second one here redundant.

// sessionIDSanitizer mirrors emotion-session-path.sh's sanitization: every
// character outside [A-Za-z0-9._-] becomes "_".
var sessionIDSanitizer = regexp.MustCompile(`[^A-Za-z0-9._-]`)

// sessionTagsPath resolves <dir>/<sanitized sessionID>.jsonl, rejecting a
// sessionID that sanitizes to "", ".", or ".." (ok=false) — the same
// degenerate-name guard emotion-session-path.sh applies. A kortex session id
// (a UUID, or whatever OpenClaw supplied via --session-id/--resume) is not
// expected to hit this in practice, but the guard is cheap insurance against
// ever writing outside dir or colliding on a degenerate name.
func sessionTagsPath(dir string, sessionID string) (path string, ok bool) {
	safe := sessionIDSanitizer.ReplaceAllString(sessionID, "_")
	if safe == "" || safe == "." || safe == ".." {
		return "", false
	}
	return filepath.Join(dir, safe+".jsonl"), true
}

// sessionTagsLine is one JSONL line of the per-session emotion-tags file.
// Field names intentionally diverge from the bash producer's schema in one
// place: "sessionId" here, not "sessionKey" — see the package-level comment
// above for why. sylphie's documented READ contract
// (hooks/memory-distill/DISTILL-RUN.md, "De onde vem a cor") only consumes
// polarity/valence/investment/emotions/evidence, so this rename carries no
// compatibility risk; ts/source/sessionId are provenance metadata, unread by
// distill today.
type sessionTagsLine struct {
	Ts         int64     `json:"ts"`
	Source     string    `json:"source"`
	SessionID  string    `json:"sessionId"`
	Polarity   string    `json:"polarity"`
	Valence    string    `json:"valence"`
	Investment int       `json:"investment"`
	Emotions   []Emotion `json:"emotions"`
	Evidence   string    `json:"evidence"`
}

// polarityOfValence ports the bash producer's valence→polarity mapping
// exactly (positiva→positive, negativa→negative, mista→mixed, anything
// else — i.e. "neutra" — →neutral).
func polarityOfValence(valence string) string {
	switch valence {
	case "positiva":
		return "positive"
	case "negativa":
		return "negative"
	case "mista":
		return "mixed"
	default:
		return "neutral"
	}
}

var (
	evidenceWhitespace  = regexp.MustCompile(`\s+`)
	evidenceSentenceEnd = regexp.MustCompile(`[.!?]`)
)

// evidenceFromParagraph ports emotion-annotate.sh's evidence extraction:
// collapse all whitespace (including embedded newlines/tabs) to single
// spaces, trim, cut at the first sentence-ending punctuation (. ! ?)
// inclusive, then hard-cap at 160 runes with an ellipsis.
func evidenceFromParagraph(paragraph string) string {
	collapsed := strings.TrimSpace(evidenceWhitespace.ReplaceAllString(paragraph, " "))
	if loc := evidenceSentenceEnd.FindStringIndex(collapsed); loc != nil {
		collapsed = collapsed[:loc[1]]
	}
	runes := []rune(collapsed)
	if len(runes) > 160 {
		return string(runes[:160]) + "…"
	}
	return collapsed
}

// maxInvestmentIndex returns the index of the FIRST paragraph that reaches
// the maximum investment across annotations (mirroring the bash producer's
// `jq index($m)`, which also returns the first match), or -1 for an empty
// slice. The maximum itself matches aggregate()'s definition; this helper
// additionally needs the INDEX (to pick the salient paragraph's text and
// emotions), which aggregate() does not expose.
func maxInvestmentIndex(annotations []ParagraphAnnotation) int {
	if len(annotations) == 0 {
		return -1
	}
	max := annotations[0].Investment
	for _, ann := range annotations[1:] {
		if ann.Investment > max {
			max = ann.Investment
		}
	}
	for i, ann := range annotations {
		if ann.Investment == max {
			return i
		}
	}
	return -1
}

// appendSessionTag appends one JSONL line for this message's salient
// paragraph to the per-session emotion-tags file. Persistence is best-effort
// in the sense that a write failure never corrupts or blocks the annotation
// result it is recording (the FAIL-OPEN invariant the whole facet package
// documents — package doc, annotator.go): the caller already committed to
// returning the annotated text before this runs. But "best-effort" must not
// mean "silent" — every I/O failure (marshal, mkdir, open, write) is recorded
// through the same facetLogger the rest of this facet's failure modes already
// use, via log, so an operator can tell persistence is broken instead of
// discovering a quietly-empty session-tags directory later. The two
// remaining early returns (dir == "", degenerate sessionID, no annotations)
// are not failures — they are the documented off-by-default/no-op gating —
// so they stay unlogged.
func appendSessionTag(log *facetLogger, model string, dir string, sessionID string, paragraphs []string, annotations []ParagraphAnnotation) {
	if dir == "" {
		return
	}
	path, ok := sessionTagsPath(dir, sessionID)
	if !ok {
		return
	}
	idx := maxInvestmentIndex(annotations)
	if idx == -1 {
		return
	}
	agg := aggregate(annotations)
	line := sessionTagsLine{
		Ts:         time.Now().UnixMilli(),
		Source:     "kortex",
		SessionID:  sessionID,
		Polarity:   polarityOfValence(agg.Valence),
		Valence:    agg.Valence,
		Investment: agg.Investment,
		Emotions:   annotations[idx].Emotions,
		Evidence:   evidenceFromParagraph(paragraphs[idx]),
	}
	data, err := json.Marshal(line)
	if err != nil {
		log.record(callLog{Facet: sessionTagsFacet, Model: model, Err: fmt.Sprintf("marshal session tag: %v", err)})
		return
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		log.record(callLog{Facet: sessionTagsFacet, Model: model, Err: fmt.Sprintf("mkdir session tags dir %s: %v", dir, err)})
		return
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		log.record(callLog{Facet: sessionTagsFacet, Model: model, Err: fmt.Sprintf("open session tags file %s: %v", path, err)})
		return
	}
	defer f.Close()
	line2 := append(data, '\n')
	if n, err := f.Write(line2); err != nil || n != len(line2) {
		log.record(callLog{Facet: sessionTagsFacet, Model: model, Err: fmt.Sprintf("write session tags file %s: wrote %d of %d bytes, err %v", path, n, len(line2), err)})
	}
}
