// Package facet implements kortex facets: model-backed processors that run
// alongside the core conversation. First facet: the input emotional
// annotator, which tags each paragraph of a real user message with
// valence/investment annotations (the contract ported from sylphie's
// emotion-annotate.sh) before the message reaches the core model.
//
// Central invariant (FAIL-OPEN): a facet may enrich the stream, never break
// it. Any failure — API error, timeout, malformed evaluator output —
// results in the original line passing through untouched, with the failure
// recorded in the structured facet log.
package facet

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/vingarcia/kortex/internal/protocol"
)

// InputAnnotator intercepts real user messages heading to the backend and
// splices per-paragraph emotion annotations into their text. It implements
// the proxy's to-backend interception point.
type InputAnnotator struct {
	evaluator    Evaluator
	systemPrompt string
	timeout      time.Duration
	model        string // logged with each call; the evaluator owns the actual routing
	log          *facetLogger
}

// AnnotatorParams wires an InputAnnotator. All fields except LogSink are
// required; a nil LogSink disables the structured facet log (the facet
// still runs). The composition root owns opening/closing the sink.
type AnnotatorParams struct {
	Evaluator    Evaluator
	SystemPrompt string
	Timeout      time.Duration
	Model        string
	LogSink      io.Writer
}

func NewInputAnnotator(params AnnotatorParams) *InputAnnotator {
	return &InputAnnotator{
		evaluator:    params.Evaluator,
		systemPrompt: params.SystemPrompt,
		timeout:      params.Timeout,
		model:        params.Model,
		log:          &facetLogger{sink: params.LogSink},
	}
}

const inputAnnotatorFacet = "input_annotator"

// annotationMarker opens every annotation line this facet injects. The
// idempotency guard trips when the GENUINELY NEW message already carries it —
// a resend/resume of an already-annotated turn — so annotations are never
// stacked nor re-billed. The guard is scoped to the new message (see
// newMessageSegment): OpenClaw prepends a "Conversation context" echo of prior
// turns whose bodies still carry this marker from when they were annotated, and
// matching that echoed history would wrongly skip every message after the first
// in a session.
const annotationMarker = "[emoções p"

// inboundContextMarker is the provenance sentinel OpenClaw appends to the header
// line of every inbound-context block it injects ahead of the real user message
// (its INBOUND_CONTEXT_MARKER; the gateway keys its own strippers on this same
// value — openclaw dist/strip-inbound-meta). Context blocks are joined with a
// blank line and the genuinely new message is appended last, so the text after
// the final marker-bearing block is the new message. OpenClaw collapses each
// echoed turn's body to a single line (sanitizeTranscriptBody: \s+ -> " "), so a
// context block never carries an internal blank line — that is what makes the
// blank-line split below unambiguous; only the real new message spans paragraphs.
const inboundContextMarker = "⟦openclaw:ctx⟧"

// newMessageSegment returns the genuinely new user message: the text after the
// last OpenClaw inbound-context block. When no context echo is present the whole
// text is the new message. Scoping the idempotency guard to this segment keeps
// echoed annotations from prior turns from suppressing the new turn's annotation.
func newMessageSegment(text string) string {
	if !strings.Contains(text, inboundContextMarker) {
		return text
	}
	blocks := strings.Split(text, "\n\n")
	lastContext := -1
	for i, block := range blocks {
		if strings.Contains(block, inboundContextMarker) {
			lastContext = i
		}
	}
	return strings.Join(blocks[lastContext+1:], "\n\n")
}

// InterceptToBackend inspects one stdin protocol line and returns the
// replacement line to forward, or nil to forward the original untouched.
// Only a real user prompt is ever touched: control messages, handshakes,
// replays and tool-result carriers pass through by construction, and every
// failure path returns nil (fail-open).
func (a *InputAnnotator) InterceptToBackend(line []byte, event protocol.Event) []byte {
	defer func() {
		// A panic in the annotation path must degrade to passthrough, not
		// kill the stdin pump; after recover the function returns the zero
		// value (nil), i.e. forward the original line.
		if r := recover(); r != nil {
			a.log.record(callLog{Facet: inputAnnotatorFacet, Model: a.model, Err: fmt.Sprintf("panic: %v", r)})
		}
	}()
	if event.Type != protocol.TypeUser || event.IsReplay || event.Message == nil {
		return nil
	}
	if event.Message.Role != "user" {
		return nil
	}
	for _, block := range event.Message.Content {
		if block.Type == "tool_result" {
			return nil
		}
	}

	newLine, err := protocol.RewriteUserText(line, a.annotateText)
	if err != nil {
		if errors.Is(err, protocol.ErrUnsupportedContent) {
			a.log.record(callLog{Facet: inputAnnotatorFacet, Model: a.model, Skipped: err.Error()})
		} else {
			a.log.record(callLog{Facet: inputAnnotatorFacet, Model: a.model, Err: err.Error()})
		}
		return nil
	}
	return newLine
}

// Annotate is the native producer's seam (paralleling InterceptToBackend, the
// proxy's line-level seam): it runs the facet over an already-parsed user
// message and returns the text to send to the core model — the annotated text
// on success, the original text unchanged on any failure or decline
// (fail-open). A panic degrades to the original text, never a dropped message.
func (a *InputAnnotator) Annotate(text string) (result string) {
	result = text
	defer func() {
		// A panic in the annotation path must degrade to the original text,
		// not propagate and abort the turn.
		if r := recover(); r != nil {
			a.log.record(callLog{Facet: inputAnnotatorFacet, Model: a.model, Err: fmt.Sprintf("panic: %v", r)})
			result = text
		}
	}()
	if annotated, ok := a.annotateText(text); ok {
		return annotated
	}
	return text
}

// annotateText runs the evaluator over the message text and interleaves the
// returned tags. ok=false declines the rewrite (nothing to annotate, or a
// failure — already logged) and the original line passes through.
func (a *InputAnnotator) annotateText(text string) (string, bool) {
	if strings.Contains(newMessageSegment(text), annotationMarker) {
		a.log.record(callLog{Facet: inputAnnotatorFacet, Model: a.model, Skipped: "text already carries annotations"})
		return "", false
	}
	paragraphs := segmentParagraphs(text)
	if len(paragraphs) == 0 {
		return "", false
	}

	ctx, cancel := context.WithTimeout(context.Background(), a.timeout)
	defer cancel()

	start := time.Now()
	userMessage := "<<<MENSAGEM_RECEBIDA_INICIO>>>\n" + numberParagraphs(paragraphs) + "\n<<<MENSAGEM_RECEBIDA_FIM>>>"
	eval, err := a.evaluator.Evaluate(ctx, a.systemPrompt+segmentationContract(len(paragraphs)), userMessage)
	entry := callLog{
		Facet:        inputAnnotatorFacet,
		Model:        a.model,
		LatencyMs:    time.Since(start).Milliseconds(),
		Paragraphs:   len(paragraphs),
		StopReason:   eval.StopReason,
		InputTokens:  eval.InputTokens,
		OutputTokens: eval.OutputTokens,
	}
	if err != nil {
		entry.Err = err.Error()
		a.log.record(entry)
		return "", false
	}

	jsonArray := extractJSONArray(eval.Text)
	if jsonArray == "" {
		entry.Err = "no JSON array in evaluator output"
		a.log.record(entry)
		return "", false
	}
	annotations, err := parseAnnotations(jsonArray, len(paragraphs))
	if err != nil {
		entry.Err = err.Error()
		a.log.record(entry)
		return "", false
	}

	entry.OK = true
	a.log.record(entry)
	return interleave(paragraphs, annotations), true
}

// callLog is one structured facet-log line. The token never appears here.
// The last block of fields is only set by the output evaluator; pointers
// keep a meaningful zero (turn 0, gate false) distinguishable from absent.
type callLog struct {
	Time         string `json:"ts"`
	Facet        string `json:"facet"`
	Model        string `json:"model"`
	LatencyMs    int64  `json:"latencyMs,omitempty"`
	Paragraphs   int    `json:"paragraphs,omitempty"`
	InputTokens  int    `json:"inputTokens,omitempty"`
	OutputTokens int    `json:"outputTokens,omitempty"`
	StopReason   string `json:"stopReason,omitempty"`
	OK           bool   `json:"ok"`
	Err          string `json:"error,omitempty"`
	Skipped      string `json:"skipped,omitempty"`

	TurnIndex     *int                  `json:"turnIndex,omitempty"`
	Tags          []ParagraphAnnotation `json:"tags,omitempty"`
	Aggregate     *Aggregate            `json:"aggregate,omitempty"`
	GateWouldFire *bool                 `json:"gateWouldFire,omitempty"`

	// Superego-only fields: the shadow critique in full, the structured
	// verdict when parseable, and how many messages the assembled context
	// carried (cost visibility for maxHistoryTurns tuning).
	HistoryMessages     int                  `json:"historyMessages,omitempty"`
	Critique            string               `json:"critique,omitempty"`
	Verdict             string               `json:"verdict,omitempty"`
	SuperegoAnnotations []SuperegoAnnotation `json:"superegoAnnotations,omitempty"`
	SuperegoWhy         string               `json:"superegoWhy,omitempty"`

	// Active-loop-only fields (superego mode "active"): the ladder round (1-4)
	// a review ran at, the loop's path ("primary" — the deterministic tag gate;
	// kortex has no model-backed fallback gate), and the loop's decision for the
	// turn ("deliver", "revise", "hold", "overrun", or "fail_open").
	Round    int    `json:"round,omitempty"`
	Path     string `json:"path,omitempty"`
	Decision string `json:"decision,omitempty"`
}

// facetLogger appends one JSON object per facet call to the sink. A nil
// sink makes it a no-op (facet keeps running).
type facetLogger struct {
	mu   sync.Mutex
	sink io.Writer
}

func (l *facetLogger) record(entry callLog) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.sink == nil {
		return
	}
	entry.Time = time.Now().Format(time.RFC3339Nano)
	data, err := json.Marshal(entry)
	if err != nil {
		return
	}
	l.sink.Write(append(data, '\n'))
}
