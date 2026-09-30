package facet

import (
	"encoding/json"
	"fmt"
	"strings"
)

// This file is the Go port of the evaluator OUTPUT CONTRACT that lives in
// sylphie's scripts/lib/emotion-eval.sh + the deterministic mechanics of
// scripts/emotion-annotate.sh: paragraph segmentation, extraction of the
// JSON array from the raw model output, schema validation (count must match
// the paragraph count so index i annotates paragraph i), and the
// deterministic interleaving of one annotation line after each paragraph.
// The model only ever returns JSON; the user's original text is never
// rewritten by the model — kortex splices the tags in deterministically.

// ParagraphAnnotation is one element of the evaluator's output array,
// annotating the paragraph at the same index. Exported because the output
// evaluator retains it as per-turn metadata for downstream facets (F2d
// superego), `about` included.
type ParagraphAnnotation struct {
	Investment int       `json:"investment"`
	Valence    string    `json:"valence"`
	Emotions   []Emotion `json:"emotions"`
}

type Emotion struct {
	Emotion string `json:"emotion"`
	Level   int    `json:"level"`
	About   string `json:"about"`
}

var validValences = map[string]bool{
	"positiva": true,
	"negativa": true,
	"mista":    true,
	"neutra":   true,
}

// segmentParagraphs splits text into paragraphs: each block of lines
// separated by one or more blank lines (a blank line is empty or
// whitespace-only). Leading/trailing blank lines produce no paragraphs;
// internal newlines of a paragraph are preserved. This mirrors the
// evaluator prompt's segmentation rule exactly — the annotation array is
// matched to these paragraphs by index.
func segmentParagraphs(text string) []string {
	var paragraphs []string
	var current []string
	flush := func() {
		if len(current) > 0 {
			paragraphs = append(paragraphs, strings.Join(current, "\n"))
			current = nil
		}
	}
	for _, line := range strings.Split(text, "\n") {
		if strings.TrimRight(line, " \t") == "" {
			flush()
			continue
		}
		current = append(current, line)
	}
	flush()
	return paragraphs
}

// extractJSONArray slices the raw model output from the first '[' to the
// last ']' (greedy), tolerating code fences or prose around the array. ""
// when no array is present. Extraction only — validation is separate.
func extractJSONArray(raw string) string {
	start := strings.Index(raw, "[")
	end := strings.LastIndex(raw, "]")
	if start == -1 || end == -1 || end < start {
		return ""
	}
	return raw[start : end+1]
}

// parseAnnotations decodes and validates the evaluator output against the
// v2 schema for expectedCount paragraphs: investment integer 0–5, valence
// in the known set, each emotion with a non-empty name and level 0–5. A
// count mismatch is an error because the index-based interleaving would
// silently annotate the wrong paragraphs.
func parseAnnotations(jsonArray string, expectedCount int) ([]ParagraphAnnotation, error) {
	var annotations []ParagraphAnnotation
	if err := json.Unmarshal([]byte(jsonArray), &annotations); err != nil {
		return nil, fmt.Errorf("evaluator output is not the expected JSON array: %w", err)
	}
	if len(annotations) != expectedCount {
		return nil, fmt.Errorf("evaluator returned %d annotations for %d paragraphs", len(annotations), expectedCount)
	}
	for i, ann := range annotations {
		if ann.Investment < 0 || ann.Investment > 5 {
			return nil, fmt.Errorf("paragraph %d: investment %d out of range 0-5", i+1, ann.Investment)
		}
		if !validValences[ann.Valence] {
			return nil, fmt.Errorf("paragraph %d: unknown valence %q", i+1, ann.Valence)
		}
		for _, emo := range ann.Emotions {
			if emo.Emotion == "" {
				return nil, fmt.Errorf("paragraph %d: emotion with empty name", i+1)
			}
			if emo.Level < 0 || emo.Level > 5 {
				return nil, fmt.Errorf("paragraph %d: emotion level %d out of range 0-5", i+1, emo.Level)
			}
		}
	}
	return annotations, nil
}

// annotationLine renders one paragraph's annotation in the exact format the
// core already consumes from sylphie's emotion-annotate.sh:
//
//	[emoções p1: investment=4 valence=negativa emotions=exaustão(4), cobrança(3)]
//
// Newlines inside an emotion name are collapsed to spaces so the annotation
// stays a single line (a model-supplied "\n" would otherwise desync the
// line-oriented format).
func annotationLine(index int, ann ParagraphAnnotation) string {
	var emotions string
	if len(ann.Emotions) == 0 {
		emotions = "nenhuma"
	} else {
		parts := make([]string, len(ann.Emotions))
		for i, emo := range ann.Emotions {
			name := strings.Join(strings.FieldsFunc(emo.Emotion, func(r rune) bool {
				return r == '\n' || r == '\r'
			}), " ")
			parts[i] = fmt.Sprintf("%s(%d)", name, emo.Level)
		}
		emotions = strings.Join(parts, ", ")
	}
	return fmt.Sprintf("[emoções p%d: investment=%d valence=%s emotions=%s]", index+1, ann.Investment, ann.Valence, emotions)
}

// interleave rebuilds the text as each original paragraph followed by its
// annotation line, blocks separated by a blank line. Paragraph text is
// spliced verbatim — it never passes through the model.
func interleave(paragraphs []string, annotations []ParagraphAnnotation) string {
	blocks := make([]string, len(paragraphs))
	for i, para := range paragraphs {
		blocks[i] = para + "\n" + annotationLine(i, annotations[i])
	}
	return strings.Join(blocks, "\n\n")
}

// numberParagraphs renders the pre-segmented paragraphs with a [P<i>] marker
// line before each one. The evaluator receives this numbered copy instead of
// the raw text: gateway-composed messages carry envelope/context blocks whose
// paragraph count the evaluator's own judgment tends to miscount, and a count
// mismatch discards the whole annotation (fail-open). The markers pin the
// segmentation the orchestrator already committed to.
func numberParagraphs(paragraphs []string) string {
	var b strings.Builder
	for i, paragraph := range paragraphs {
		if i > 0 {
			b.WriteString("\n\n")
		}
		fmt.Fprintf(&b, "[P%d]\n%s", i+1, paragraph)
	}
	return b.String()
}

// segmentationContract is appended to the evaluator system prompt to make the
// index alignment explicit for the numbered copy built by numberParagraphs.
func segmentationContract(count int) string {
	return fmt.Sprintf("\n\n## SEGMENTAÇÃO PRÉ-FEITA (contrato estrito desta chamada)\n"+
		"A mensagem entre os delimitadores foi pré-segmentada pelo orquestrador em %d parágrafos, "+
		"cada um precedido por uma linha-marcador [P<i>]. Os marcadores NÃO fazem parte do texto original — ignore-os como conteúdo.\n"+
		"Retorne EXATAMENTE %d elementos no array JSON: o elemento de índice i anota o parágrafo [P<i+1>].\n"+
		"Parágrafo sem carga emocional (metadados, JSON de envelope, texto puramente técnico) recebe "+
		"{\"investment\": 0, \"valence\": \"neutra\", \"emotions\": []} — nunca omita um elemento.",
		count, count)
}
