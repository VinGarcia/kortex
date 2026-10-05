package facet

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vingarcia/kortex/internal/history"
)

// This file assembles the superego's view of the conversation (design #4351
// + #4354): the canonical history converted to the API message array the
// core itself consumed — clean final text, tags stripped (tags are
// orchestrator metadata, never part of the canonical history) — with the
// turn under review entering NOT as an assistant message but as a final
// user-role message framed as an unsent draft, the F2c emotion tags
// interleaved between its paragraphs. The draft framing is DETERMINISTIC
// (fixed strings below, never model-generated).

// superegoMessages converts the history snapshot into the superego's message
// array. Turns after turnIndex are ignored; turns[turnIndex]'s user text
// closes the plain history and draft (the reframed reviewed output) is the
// final user message. maxHistoryTurns > 0 keeps only that many of the most
// recent turns (the reviewed turn included), dropping the oldest ones.
//
// Simplifications, on purpose: the HISTORY turns' tool calls are omitted (the
// superego judges the spoken exchange; replaying every past turn's tool
// traffic would multiply tokens for little judgment value), and turns whose
// user or assistant text is empty contribute no message for that side.
// Consecutive same-role messages are valid Messages API input, so no artificial
// filler is inserted. The ONE exception is the reviewed turn: a compact,
// bounded summary of only turns[turnIndex]'s tool calls is appended to the
// draft-user message so the superego can verify the provenance of facts the
// assistant claimed to have checked (#4749). Nothing is appended when the
// reviewed turn ran no tools.
func superegoMessages(snapshot history.Snapshot, turnIndex int, draft string, maxHistoryTurns int) []ReviewMessage {
	turns := snapshot.Turns[:turnIndex+1]
	start := 0
	if maxHistoryTurns > 0 && len(turns) > maxHistoryTurns {
		start = len(turns) - maxHistoryTurns
	}
	var messages []ReviewMessage
	for i := start; i <= turnIndex; i++ {
		if user := stripAnnotationLines(turns[i].UserText); user != "" {
			messages = append(messages, ReviewMessage{Role: "user", Content: user})
		}
		if i == turnIndex {
			// The reviewed turn's assistant text enters as the draft-user
			// message instead of an assistant message (draft-user decision,
			// #4354: mitigates self-leniency bias).
			break
		}
		if turns[i].AssistantText != "" {
			messages = append(messages, ReviewMessage{Role: "assistant", Content: turns[i].AssistantText})
		}
	}
	final := draft
	if summary := toolCallsSummary(turns[turnIndex].ToolCalls); summary != "" {
		final = draft + "\n\n" + summary
	}
	return append(messages, ReviewMessage{Role: "user", Content: final})
}

// superegoToolInputMax and superegoToolResultMax bound how much of each tool
// call's input and result the summary shows. The superego only needs enough to
// confirm a fact was actually fetched, not the full payload; bounding keeps the
// reviewed turn's tool block from blowing up the token budget on a large tool
// output.
const (
	superegoToolInputMax  = 200
	superegoToolResultMax = 200
)

// toolCallsSummary renders the reviewed turn's tool calls as one compact,
// deterministic, clearly-delimited block for the superego, in the same
// Portuguese framing as the draft-under-review message. Each call shows its
// name, a bounded snippet of the input and of the result, and an error marker
// when the tool failed. It returns "" when there are no tool calls, so the
// caller appends nothing (no empty block). Ordering follows ToolCalls, which
// preserves tool_use order, so the output is stable.
func toolCallsSummary(calls []history.ToolCall) string {
	if len(calls) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("<<<FERRAMENTAS_EXECUTADAS_NESTE_TURNO_INICIO>>>\n")
	b.WriteString("Ferramentas que EU realmente executei neste turno (use para verificar a procedência de fatos que afirmei ter apurado):\n")
	for i, call := range calls {
		name := call.Name
		if name == "" {
			name = "(sem nome)"
		}
		fmt.Fprintf(&b, "\n%d. %s", i+1, name)
		if call.IsError {
			b.WriteString(" [ERRO]")
		}
		if input := boundedSnippet(call.Input, superegoToolInputMax); input != "" {
			fmt.Fprintf(&b, "\n   entrada: %s", input)
		}
		if result := boundedSnippet(call.Result, superegoToolResultMax); result != "" {
			fmt.Fprintf(&b, "\n   resultado: %s", result)
		}
	}
	b.WriteString("\n<<<FERRAMENTAS_EXECUTADAS_NESTE_TURNO_FIM>>>")
	return b.String()
}

// boundedSnippet reduces s to its first non-empty line, trimmed and capped at
// maxRunes runes (an ellipsis marks truncation). It is rune-safe so a cap never
// splits a multi-byte character. "" in → "" out.
func boundedSnippet(s string, maxRunes int) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if nl := strings.IndexByte(s, '\n'); nl >= 0 {
		s = strings.TrimSpace(s[:nl])
	}
	runes := []rune(s)
	if len(runes) > maxRunes {
		return string(runes[:maxRunes]) + "…"
	}
	return s
}

// stripAnnotationLines removes the input annotator's injected tag lines
// (lines opening with annotationMarker) from a recorded user text: the
// history records what the backend received — annotated when that facet is
// on — but the superego's canonical view carries clean text only. A genuine
// user line starting with the marker is stripped too; the annotator already
// treats that marker as its own (it refuses to re-annotate such text), so
// the ambiguity is inherent to the marker, not introduced here.
func stripAnnotationLines(text string) string {
	if !strings.Contains(text, annotationMarker) {
		return text
	}
	var kept []string
	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), annotationMarker) {
			continue
		}
		kept = append(kept, line)
	}
	return strings.TrimSpace(strings.Join(kept, "\n"))
}

// draftUnderReviewMessage renders the reviewed turn as the deterministic
// draft-user message: fixed delimiters and framing, the draft paragraphs
// with their annotation lines interleaved (same renderer the core's input
// uses), and — because the inline tag format carries no `about` anchors — a
// compact JSON block after the draft listing each paragraph's emotions with
// their anchors. len(paragraphs) == len(tags) is the caller's contract.
func draftUnderReviewMessage(paragraphs []string, tags []ParagraphAnnotation) string {
	var b strings.Builder
	b.WriteString("<<<RASCUNHO_SOB_REVISAO_INICIO>>>\n")
	b.WriteString("Rascunho da minha próxima fala — AINDA NÃO ENVIADA — com as tags de emoção do avaliador intercaladas entre os parágrafos:\n\n")
	b.WriteString(interleave(paragraphs, tags))
	b.WriteString("\n<<<RASCUNHO_SOB_REVISAO_FIM>>>")
	if abouts := aboutsJSON(tags); abouts != "" {
		b.WriteString("\n\nÂncoras `about` do avaliador (a que trecho cada emoção se refere), por parágrafo, JSON: ")
		b.WriteString(abouts)
	}
	return b.String()
}

// activeDraftMessage renders the draft-under-review for the active loop: the
// deterministic draftUnderReviewMessage, prefixed with the ladder round number
// and the annotations from previous rounds. The superego prompt expects both
// (it is told "the review round number (1 to 4) and the annotations from
// previous rounds"), and the round number is what enforces the 3+1 contract —
// revise only in rounds 1-3, hold-and-ask only in round 4. prior is the
// critiques of the earlier rounds in order; empty on round 1. The header is
// deterministic (fixed strings), never model-generated.
func activeDraftMessage(paragraphs []string, tags []ParagraphAnnotation, round int, prior []SuperegoCritique) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Rodada de revisão: %d de 4.\n", round)
	if round >= 4 {
		b.WriteString("Esta é a última rodada: você só pode APROVAR ou RETER-E-PERGUNTAR-AO-HUMANO (hold_ask_human); não emita novas anotações de revisão.\n")
	}
	if len(prior) > 0 {
		b.WriteString("\nAnotações das rodadas anteriores (o rascunho abaixo já tentou incorporá-las):\n")
		for i, crit := range prior {
			fmt.Fprintf(&b, "- Rodada %d", i+1)
			if crit.Why != "" {
				fmt.Fprintf(&b, " (%s)", crit.Why)
			}
			b.WriteString(":\n")
			for _, ann := range crit.Annotations {
				fmt.Fprintf(&b, "  - parágrafo %d: %s — %s\n", ann.Paragraph, ann.Issue, ann.Why)
			}
		}
	}
	b.WriteString("\n")
	b.WriteString(draftUnderReviewMessage(paragraphs, tags))
	return b.String()
}

// paragraphAbouts is one paragraph's entry of the abouts block.
type paragraphAbouts struct {
	Paragraph int       `json:"p"`
	Emotions  []Emotion `json:"emotions"`
}

// aboutsJSON renders the per-paragraph emotions (about included) as one
// compact JSON array, skipping emotionless paragraphs. "" when no paragraph
// carries emotions (the block is then omitted entirely).
func aboutsJSON(tags []ParagraphAnnotation) string {
	var entries []paragraphAbouts
	for i, tag := range tags {
		if len(tag.Emotions) == 0 {
			continue
		}
		entries = append(entries, paragraphAbouts{Paragraph: i + 1, Emotions: tag.Emotions})
	}
	if len(entries) == 0 {
		return ""
	}
	data, err := json.Marshal(entries)
	if err != nil {
		return ""
	}
	return string(data)
}
