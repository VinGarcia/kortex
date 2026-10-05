package facet

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// This file builds the EPHEMERAL tone digest (design-organ-emocoes-polo-positivo
// §0.-1 item 3, fase 3): a short, human-readable summary of the current
// conversation's emotional tone, computed from the input annotator's per-message
// emotion tags, injected as the LAST item of the core model's context on every
// turn. "Ephemeral" means recomputed each turn, injected only into that turn's
// core call, and NEVER persisted into the canonical/shared history — the same
// contract as the superego's ephemeral tail (design-prefrontal-redesign #4354).
//
// SOURCE OF ANNOTATIONS. The input annotator does not retain its structured
// output in-process; it interleaves the tags into the user text as
// `[emoções pN: ...]` lines, and that annotated text is what persists in the
// canonical history (design item 2: "as falas do Vini já ficam anotadas
// `[emoções pN]` no histórico canônico que o core vê"). The only structured
// retention in kortex is the OUTPUT evaluator's TurnEmotion, which tags the
// ASSISTANT's own text, not the user's. The digest is about the USER's tone, so
// its clean in-process source is the annotation lines already carried by the
// session's user messages — re-parsed here with parseAnnotationLine, the inverse
// of contract.go's annotationLine. This is the facet's own tag format (not the
// wire protocol), so parsing it stays correctly in the facet layer.

// maxDigestEmotions bounds how many distinct emotions the digest lists. By
// design there is NO last-N window and NO expiration (design item 3: "Sem janela
// de últimos-N"), so a long session could otherwise accumulate an unbounded
// number of distinct emotions in the injected block. The cap keeps the block
// bounded; the most salient survive (see the deterministic sort in
// buildToneDigest) and any beyond the cap are summarized as an omitted count.
const maxDigestEmotions = 12

// digestEmotionNameMax bounds one rendered emotion name's length, so a single
// pathological tag cannot bloat the block.
const digestEmotionNameMax = 60

// The ephemeral digest block delimiters, in the same spirit as the superego's
// clearly-delimited ephemeral blocks (superego_context.go). They let the core
// model recognize the injected tone context as a bounded, orchestrator-authored
// section distinct from the conversation itself.
const (
	toneDigestOpen  = "<<<TOM_DA_CONVERSA_INICIO>>>"
	toneDigestClose = "<<<TOM_DA_CONVERSA_FIM>>>"
)

// toneEmotion accumulates one deduplicated emotion across the whole session: its
// highest observed level and how many times it occurred. The dedup key is
// (emotion name, valence) — the same key the design's diary drain dedups on
// (§3 Elo-3) — because the same emotion name under a different valence is a
// genuinely different tone and must not be collapsed.
type toneEmotion struct {
	name     string
	valence  string
	maxLevel int
	count    int
}

// BuildToneDigest renders the ephemeral tone digest from the session's user
// messages (each carrying the input annotator's `[emoções pN]` lines). It
// deduplicates emotions ONCE by (name, valence) keeping the maximum level and
// the occurrence count, keeps ALL valences (positiva/negativa/mista/neutra —
// the value is the full tone, design Q3 FECHADA), orders the result
// deterministically, bounds its size, and returns a clearly-delimited block.
//
// minLevel drops emotions below that level before dedup; minLevel <= 0 includes
// every emotion. The authoritative design ties a `level >= 3` cut to the DIARY
// distillation (item 1); whether that same cut applies to this digest is
// genuinely ambiguous in item 3 ("com corte >=3 + dedup, mantém TODAS as
// emoções da sessão no destilado"). Per the task's tie-breaker the caller wires
// minLevel = 0 (include all) and this parameter keeps the >=3 reading one call
// away without a code change.
//
// FAIL-OPEN: no annotations (or none surviving minLevel) returns "" so the
// caller injects nothing and the turn proceeds normally — the facet invariant.
func BuildToneDigest(userTexts []string, minLevel int) string {
	type key struct{ name, valence string }
	acc := map[key]*toneEmotion{}
	// order preserves first-seen insertion order so the final sort is the ONLY
	// source of ordering (map iteration order is random in Go); it never leaks
	// into the output on its own.
	var order []key
	for _, text := range userTexts {
		for _, line := range strings.Split(text, "\n") {
			valence, emotions, ok := parseAnnotationLine(line)
			if !ok {
				continue
			}
			for _, emo := range emotions {
				if emo.Level < minLevel {
					continue
				}
				k := key{name: emo.Emotion, valence: valence}
				e, exists := acc[k]
				if !exists {
					e = &toneEmotion{name: emo.Emotion, valence: valence}
					acc[k] = e
					order = append(order, k)
				}
				e.count++
				if emo.Level > e.maxLevel {
					e.maxLevel = emo.Level
				}
			}
		}
	}
	if len(acc) == 0 {
		return ""
	}

	list := make([]*toneEmotion, 0, len(acc))
	for _, k := range order {
		list = append(list, acc[k])
	}
	// Deterministic ordering: most salient first — highest level, then most
	// frequent, then name, then valence as final tie-breakers so the output is
	// stable for the same input regardless of map iteration order.
	sort.Slice(list, func(i, j int) bool {
		a, b := list[i], list[j]
		if a.maxLevel != b.maxLevel {
			return a.maxLevel > b.maxLevel
		}
		if a.count != b.count {
			return a.count > b.count
		}
		if a.name != b.name {
			return a.name < b.name
		}
		return a.valence < b.valence
	})

	omitted := 0
	if len(list) > maxDigestEmotions {
		omitted = len(list) - maxDigestEmotions
		list = list[:maxDigestEmotions]
	}

	var b strings.Builder
	b.WriteString(toneDigestOpen)
	b.WriteString("\nDestilado efêmero do tom emocional desta conversa (recomputado a cada turno, não faz parte do histórico):")
	for _, e := range list {
		name := boundedSnippet(e.name, digestEmotionNameMax)
		fmt.Fprintf(&b, "\n- %s — nível máx %d, valência %s, %dx", name, e.maxLevel, e.valence, e.count)
	}
	if omitted > 0 {
		fmt.Fprintf(&b, "\n- (+%d emoção(ões) de menor saliência omitida(s))", omitted)
	}
	b.WriteString("\n")
	b.WriteString(toneDigestClose)
	return b.String()
}

// ToneDigest is the stateless facet that satisfies the native path's
// session.ToneDigester port. It holds only the minLevel knob; it makes no model
// call and keeps no per-session state (every turn recomputes from the user texts
// handed in), so the composition root can construct one and share it.
type ToneDigest struct {
	minLevel int
}

// NewToneDigest builds the tone-digest facet. minLevel is the pre-dedup level
// cut (0 includes every emotion — see BuildToneDigest).
func NewToneDigest(minLevel int) *ToneDigest {
	return &ToneDigest{minLevel: minLevel}
}

// Digest renders the ephemeral tone digest for the session's user messages. It
// is fail-open by construction: BuildToneDigest is pure, bounded string work over
// the annotation lines and returns "" whenever there is nothing to digest (no
// annotations, or none surviving minLevel), so the caller injects nothing and the
// turn proceeds normally. No recover() wraps it — there is no impossible state to
// swallow here, and a silent swallow would hide a real bug rather than protect
// the turn (every other facet logs its panic; this one has none to log).
func (d *ToneDigest) Digest(userTexts []string) string {
	return BuildToneDigest(userTexts, d.minLevel)
}

// parseAnnotationLine is the inverse of contract.go's annotationLine: given one
// line of text, it recovers the paragraph valence and the list of emotions when
// the line is an annotation line, or reports ok=false otherwise. A line with the
// marker but no emotions ("emotions=nenhuma") is a valid annotation line that
// contributes no emotions (ok=true, nil slice). Each emotion inherits its
// paragraph's valence (the inline format carries valence per paragraph, not per
// emotion). Malformed emotion tokens are skipped individually (fail-open),
// never aborting the whole line.
func parseAnnotationLine(line string) (string, []Emotion, bool) {
	line = strings.TrimSpace(line)
	if !strings.HasPrefix(line, annotationMarker) || !strings.HasSuffix(line, "]") {
		return "", nil, false
	}
	inner := strings.TrimSuffix(line, "]")

	valence := fieldValue(inner, "valence=")
	if !validValences[valence] {
		return "", nil, false
	}

	emoStr := afterField(inner, "emotions=")
	if emoStr == "" || emoStr == "nenhuma" {
		return valence, nil, true
	}
	var emotions []Emotion
	for _, token := range strings.Split(emoStr, ", ") {
		name, level, ok := parseEmotionToken(token)
		if !ok {
			continue
		}
		emotions = append(emotions, Emotion{Emotion: name, Level: level})
	}
	return valence, emotions, true
}

// fieldValue returns the single-token value of "prefix" within s (the text from
// just after the prefix up to the next space), or "" when the prefix is absent.
func fieldValue(s string, prefix string) string {
	idx := strings.Index(s, prefix)
	if idx < 0 {
		return ""
	}
	rest := s[idx+len(prefix):]
	if sp := strings.IndexByte(rest, ' '); sp >= 0 {
		return rest[:sp]
	}
	return rest
}

// afterField returns everything after "prefix" within s, trimmed. The emotions
// field is the last field on an annotation line (its "]" already stripped by the
// caller), so the remainder of the line is its value.
func afterField(s string, prefix string) string {
	idx := strings.Index(s, prefix)
	if idx < 0 {
		return ""
	}
	return strings.TrimSpace(s[idx+len(prefix):])
}

// parseEmotionToken parses one "name(level)" token back into its name and
// integer level. The name is everything before the LAST '(' (emotion names may
// contain spaces, and the level parens are always the trailing group), and the
// level is the digits inside that final paren group. ok=false for a token
// without a parenthesized integer level or with an empty name.
func parseEmotionToken(token string) (string, int, bool) {
	token = strings.TrimSpace(token)
	open := strings.LastIndexByte(token, '(')
	closeIdx := strings.LastIndexByte(token, ')')
	if open <= 0 || closeIdx <= open {
		return "", 0, false
	}
	name := strings.TrimSpace(token[:open])
	if name == "" {
		return "", 0, false
	}
	level, err := strconv.Atoi(strings.TrimSpace(token[open+1 : closeIdx]))
	if err != nil {
		return "", 0, false
	}
	return name, level, true
}
