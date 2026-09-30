package facet

import (
	"context"
	"fmt"
	"io"
	"sync"
	"time"
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
	LogSink           io.Writer
}

func NewOutputEvaluator(params OutputEvaluatorParams) *OutputEvaluator {
	return &OutputEvaluator{
		evaluator:         params.Evaluator,
		systemPrompt:      params.SystemPrompt,
		timeout:           params.Timeout,
		model:             params.Model,
		gateMinInvestment: params.GateMinInvestment,
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
func (e *OutputEvaluator) EvaluateCompletedTurn(turnIndex int, assistantText string) {
	e.inFlight.Add(1)
	go func() {
		defer e.inFlight.Done()
		e.evaluateTurn(turnIndex, assistantText)
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

// evaluateTurn runs one evaluation end to end: evaluator call, contract
// validation, aggregate + hypothetical gate decision, log + metadata
// retention. Every failure path ends in a log entry and nothing else.
func (e *OutputEvaluator) evaluateTurn(turnIndex int, assistantText string) {
	defer func() {
		// A panic here runs on a private goroutine: without this recover it
		// would kill the whole proxy, which is the one thing a facet must
		// never do.
		if r := recover(); r != nil {
			e.log.record(callLog{Facet: outputEvaluatorFacet, Model: e.model, TurnIndex: &turnIndex, Err: fmt.Sprintf("panic: %v", r)})
		}
	}()

	paragraphs := segmentParagraphs(assistantText)
	if len(paragraphs) == 0 {
		e.log.record(callLog{Facet: outputEvaluatorFacet, Model: e.model, TurnIndex: &turnIndex, Skipped: "turn has no assistant text"})
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), e.timeout)
	defer cancel()

	start := time.Now()
	userMessage := "<<<RESPOSTA_PROPOSTA_INICIO>>>\n" + assistantText + "\n<<<RESPOSTA_PROPOSTA_FIM>>>"
	eval, err := e.evaluator.Evaluate(ctx, e.systemPrompt, userMessage)
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
		return
	}

	jsonArray := extractJSONArray(eval.Text)
	if jsonArray == "" {
		entry.Err = "no JSON array in evaluator output"
		e.log.record(entry)
		return
	}
	annotations, err := parseAnnotations(jsonArray, len(paragraphs))
	if err != nil {
		entry.Err = err.Error()
		e.log.record(entry)
		return
	}

	agg := aggregate(annotations)
	fire := gateWouldFire(agg, e.gateMinInvestment)
	emotion := TurnEmotion{
		TurnIndex:     turnIndex,
		Tags:          annotations,
		Aggregate:     agg,
		GateWouldFire: fire,
	}
	e.mu.Lock()
	e.turns[turnIndex] = emotion
	e.mu.Unlock()

	entry.OK = true
	entry.Tags = annotations
	entry.Aggregate = &agg
	entry.GateWouldFire = &fire
	e.log.record(entry)
}
