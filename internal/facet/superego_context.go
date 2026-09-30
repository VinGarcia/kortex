package facet

import (
	"encoding/json"
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
// Simplifications, on purpose: tool calls are omitted (the superego judges
// the spoken exchange; tool traffic would multiply tokens for little
// judgment value), and turns whose user or assistant text is empty
// contribute no message for that side. Consecutive same-role messages are
// valid Messages API input, so no artificial filler is inserted.
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
	return append(messages, ReviewMessage{Role: "user", Content: draft})
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
