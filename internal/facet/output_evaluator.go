package facet

import (
	"context"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/vingarcia/kortex/internal/history"
)

// OutputEvaluator is the F2c facet: it runs the output emotion evaluator
// over each COMPLETED assistant turn (the full final text — never partial
// stream deltas), aggregates the per-paragraph tags into the gate input, and
// records the hypothetical gate decision.
//
// OBSERVABILITY MODE (the F2c rollout decision): nothing about the stream
// changes — the assistant output reaches the gateway untouched and the
// evaluation runs asynchronously after the turn's terminal result event.
// The outcome goes to the structured facet log and is retained in memory as
// per-turn metadata (TurnEmotion, `about` included) for the F2d superego to
// consume. Fail-open like every facet: any failure is a log entry, never a
// behavior change.
type OutputEvaluator struct {
	evaluator         Evaluator
	systemPrompt      string
	timeout           time.Duration
	model             string // logged with each call; the evaluator owns the actual routing
	gateMinInvestment int
	superego          *Superego
	log               *facetLogger

	inFlight sync.WaitGroup
	mu       sync.Mutex
	turns    map[int]TurnEmotion
}

// OutputEvaluatorParams wires an OutputEvaluator. All fields except LogSink
// are required; a nil LogSink disables the structured facet log (the facet
// still runs). The composition root owns opening/closing the sink.
type OutputEvaluatorParams struct {
	Evaluator         Evaluator
	SystemPrompt      string
	Timeout           time.Duration
	Model             string
	GateMinInvestment int
	// Superego, when non-nil, is fired once per turn whose hypothetical
	// gate decision is true (the F2d shadow trigger). Review dispatches its
	// own goroutine and returns promptly.
	Superego *Superego
	LogSink  io.Writer
}

func NewOutputEvaluator(params OutputEvaluatorParams) *OutputEvaluator {
	return &OutputEvaluator{
		evaluator:         params.Evaluator,
		systemPrompt:      params.SystemPrompt,
		timeout:           params.Timeout,
		model:             params.Model,
		gateMinInvestment: params.GateMinInvestment,
		superego:          params.Superego,
		log:               &facetLogger{sink: params.LogSink},
		turns:             map[int]TurnEmotion{},
	}
}

const outputEvaluatorFacet = "output_evaluator"

// TurnEmotion is the retained per-turn metadata: the full per-paragraph
// tags (each emotion with its `about` anchor — it never appears in any
// inline tag, but the superego needs it), the gate aggregate, and the
// hypothetical gate decision. This is the structure the F2d superego will
// consume alongside the canonical history; the history itself never carries
// tags (design decision #4354: tags are orchestrator metadata, kept out of
// the canonical history).
type TurnEmotion struct {
	TurnIndex     int                   `json:"turnIndex"`
	Tags          []ParagraphAnnotation `json:"tags"`
	Aggregate     Aggregate             `json:"aggregate"`
	GateWouldFire bool                  `json:"gateWouldFire"`
}

// EvaluateCompletedTurn is the proxy's turn-completion hook. It dispatches
// the evaluation on its own goroutine and returns immediately: the caller
// sits on the stdout pump, and the assistant output must reach the gateway
// with zero added latency.
func (e *OutputEvaluator) EvaluateCompletedTurn(turnIndex int, snapshot history.Snapshot) {
	e.inFlight.Add(1)
	go func() {
		defer e.inFlight.Done()
		e.evaluateTurn(turnIndex, snapshot)
	}()
}

// WaitInFlight blocks until every dispatched evaluation finished. The
// composition root calls it after the child exited (output long delivered)
// so a one-shot run does not kill the final turn's evaluation by exiting.
func (e *OutputEvaluator) WaitInFlight() {
	e.inFlight.Wait()
}

// TurnEmotions returns a copy of the retained per-turn metadata, keyed by
// turn index. Turns whose evaluation failed or was skipped are absent.
func (e *OutputEvaluator) TurnEmotions() map[int]TurnEmotion {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make(map[int]TurnEmotion, len(e.turns))
	for k, v := range e.turns {
		out[k] = v
	}
	return out
}

// DraftEvaluation is the synchronous output-evaluation result the active
// superego loop consumes: the paragraph segmentation the evaluator committed
// to (so index i tags paragraph i), the per-paragraph tags, the gate aggregate
// and its decision.
type DraftEvaluation struct {
	Paragraphs    []string
	Tags          []ParagraphAnnotation
	Aggregate     Aggregate
	GateWouldFire bool
}

// EvaluateDraft runs the output emotion evaluator over one draft text
// synchronously and returns the gate decision. The bool is false (with a
// logged reason) when the evaluator could not produce a decision; the active
// superego loop fails open on false. It logs one call line exactly like the
// async path. turnIndex is for logging only (the draft may be a revised
// candidate not yet in any snapshot). The caller owns the context deadline.
func (e *OutputEvaluator) EvaluateDraft(ctx context.Context, turnIndex int, draft string) (DraftEvaluation, bool) {
	paragraphs := segmentParagraphs(draft)
	if len(paragraphs) == 0 {
		e.log.record(callLog{Facet: outputEvaluatorFacet, Model: e.model, TurnIndex: &turnIndex, Skipped: "turn has no assistant text"})
		return DraftEvaluation{}, false
	}

	ctx, cancel := context.WithTimeout(ctx, e.timeout)
	defer cancel()

	start := time.Now()
	userMessage := "<<<RESPOSTA_PROPOSTA_INICIO>>>\n" + numberParagraphs(paragraphs) + "\n<<<RESPOSTA_PROPOSTA_FIM>>>"
	eval, err := e.evaluator.Evaluate(ctx, e.systemPrompt+segmentationContract(len(paragraphs)), userMessage)
	entry := callLog{
		Facet:        outputEvaluatorFacet,
		Model:        e.model,
		TurnIndex:    &turnIndex,
		LatencyMs:    time.Since(start).Milliseconds(),
		Paragraphs:   len(paragraphs),
		StopReason:   eval.StopReason,
		InputTokens:  eval.InputTokens,
		OutputTokens: eval.OutputTokens,
	}
	if err != nil {
		entry.Err = err.Error()
		e.log.record(entry)
		return DraftEvaluation{}, false
	}

	jsonArray := extractJSONArray(eval.Text)
	if jsonArray == "" {
		entry.Err = "no JSON array in evaluator output"
		e.log.record(entry)
		return DraftEvaluation{}, false
	}
	annotations, err := parseAnnotations(jsonArray, len(paragraphs))
	if err != nil {
		entry.Err = err.Error()
		e.log.record(entry)
		return DraftEvaluation{}, false
	}

	agg := aggregate(annotations)
	fire := gateWouldFire(agg, e.gateMinInvestment)

	entry.OK = true
	entry.Tags = annotations
	entry.Aggregate = &agg
	entry.GateWouldFire = &fire
	e.log.record(entry)
	return DraftEvaluation{Paragraphs: paragraphs, Tags: annotations, Aggregate: agg, GateWouldFire: fire}, true
}

// evaluateTurn runs one async (shadow-mode) evaluation end to end over a
// completed turn's text: it reuses EvaluateDraft for the evaluator call and
// gate decision, retains the per-turn metadata, and — on a gate hit — fires
// the shadow superego review. Every failure path ends in a log entry (inside
// EvaluateDraft) and nothing else.
func (e *OutputEvaluator) evaluateTurn(turnIndex int, snapshot history.Snapshot) {
	defer func() {
		// A panic here runs on a private goroutine: without this recover it
		// would kill the whole proxy, which is the one thing a facet must
		// never do.
		if r := recover(); r != nil {
			e.log.record(callLog{Facet: outputEvaluatorFacet, Model: e.model, TurnIndex: &turnIndex, Err: fmt.Sprintf("panic: %v", r)})
		}
	}()

	if turnIndex < 0 || turnIndex >= len(snapshot.Turns) {
		e.log.record(callLog{Facet: outputEvaluatorFacet, Model: e.model, TurnIndex: &turnIndex, Err: fmt.Sprintf("turn index %d outside snapshot of %d turns", turnIndex, len(snapshot.Turns))})
		return
	}

	de, ok := e.EvaluateDraft(context.Background(), turnIndex, snapshot.Turns[turnIndex].AssistantText)
	if !ok {
		return
	}

	emotion := TurnEmotion{
		TurnIndex:     turnIndex,
		Tags:          de.Tags,
		Aggregate:     de.Aggregate,
		GateWouldFire: de.GateWouldFire,
	}
	e.mu.Lock()
	e.turns[turnIndex] = emotion
	e.mu.Unlock()

	// The F2d shadow trigger: one superego review per gate hit. Review
	// dispatches its own goroutine, so this evaluation goroutine's lifetime
	// never stretches to the (much longer) superego call.
	if de.GateWouldFire && e.superego != nil {
		e.superego.Review(turnIndex, snapshot, de.Paragraphs, de.Tags)
	}
}
