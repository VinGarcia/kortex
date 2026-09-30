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

const facetName = "input_annotator"

// annotationMarker opens every annotation line this facet injects. A text
// already carrying it is never annotated again: re-sent messages (session
// resume/replay paths) must not stack annotations or re-bill the evaluator.
const annotationMarker = "[emoções p"

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
			a.log.record(callLog{Facet: facetName, Model: a.model, Err: fmt.Sprintf("panic: %v", r)})
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
			a.log.record(callLog{Facet: facetName, Model: a.model, Skipped: err.Error()})
		} else {
			a.log.record(callLog{Facet: facetName, Model: a.model, Err: err.Error()})
		}
		return nil
	}
	return newLine
}

// annotateText runs the evaluator over the message text and interleaves the
// returned tags. ok=false declines the rewrite (nothing to annotate, or a
// failure — already logged) and the original line passes through.
func (a *InputAnnotator) annotateText(text string) (string, bool) {
	if strings.Contains(text, annotationMarker) {
		a.log.record(callLog{Facet: facetName, Model: a.model, Skipped: "text already carries annotations"})
		return "", false
	}
	paragraphs := segmentParagraphs(text)
	if len(paragraphs) == 0 {
		return "", false
	}

	ctx, cancel := context.WithTimeout(context.Background(), a.timeout)
	defer cancel()

	start := time.Now()
	userMessage := "<<<MENSAGEM_RECEBIDA_INICIO>>>\n" + text + "\n<<<MENSAGEM_RECEBIDA_FIM>>>"
	eval, err := a.evaluator.Evaluate(ctx, a.systemPrompt, userMessage)
	entry := callLog{
		Facet:        facetName,
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
