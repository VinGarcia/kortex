package facet

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

// newTestOutputEvaluator wires an evaluator with the default threshold; a
// nil logSink (typed-nil guarded) keeps the facet log off.
func newTestOutputEvaluator(evaluator Evaluator, logSink *bytes.Buffer) *OutputEvaluator {
	params := OutputEvaluatorParams{
		Evaluator:         evaluator,
		SystemPrompt:      "avalie a carga da resposta",
		Timeout:           time.Second,
		Model:             "claude-haiku-4-5-20251001",
		GateMinInvestment: 4,
	}
	if logSink != nil {
		params.LogSink = logSink
	}
	return NewOutputEvaluator(params)
}

// evaluateAndWait drives the real async entry point and waits for the
// evaluation to land, so assertions see the final state.
func evaluateAndWait(e *OutputEvaluator, turnIndex int, text string) {
	e.EvaluateCompletedTurn(turnIndex, text)
	e.WaitInFlight()
}

func lastLogEntry(t *testing.T, log *bytes.Buffer) map[string]any {
	t.Helper()
	lines := strings.Split(strings.TrimSpace(log.String()), "\n")
	var entry map[string]any
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &entry); err != nil {
		t.Fatalf("facet log line is not JSON: %v: %q", err, lines[len(lines)-1])
	}
	return entry
}

func TestOutputEvaluator_happyPathMultiParagraph(t *testing.T) {
	evaluator := &fakeEvaluator{response: evalText(`[
		{"investment":4,"valence":"negativa","emotions":[{"emotion":"recusa","level":4,"about":"não vou fazer isso"}]},
		{"investment":0,"valence":"neutra","emotions":[]}
	]`)}
	var log bytes.Buffer
	e := newTestOutputEvaluator(evaluator, &log)

	evaluateAndWait(e, 2, "não vou fazer isso\n\nrodei o script, saiu verde")

	if !strings.Contains(evaluator.gotUser, "<<<RESPOSTA_PROPOSTA_INICIO>>>") ||
		!strings.Contains(evaluator.gotUser, "<<<RESPOSTA_PROPOSTA_FIM>>>") {
		t.Errorf("user message missing output delimiters: %q", evaluator.gotUser)
	}
	turns := e.TurnEmotions()
	emotion, ok := turns[2]
	if !ok {
		t.Fatalf("no metadata retained for turn 2: %+v", turns)
	}
	if emotion.TurnIndex != 2 || len(emotion.Tags) != 2 {
		t.Fatalf("unexpected metadata: %+v", emotion)
	}
	// The about anchor must survive into the retained metadata — the
	// superego needs it even though no inline tag ever carries it.
	if emotion.Tags[0].Emotions[0].About != "não vou fazer isso" {
		t.Errorf("about not retained: %+v", emotion.Tags[0])
	}
	if emotion.Aggregate != (Aggregate{Investment: 4, Valence: "negativa"}) {
		t.Errorf("aggregate = %+v", emotion.Aggregate)
	}
	if !emotion.GateWouldFire {
		t.Error("gate should hypothetically fire at investment 4")
	}

	entry := lastLogEntry(t, &log)
	if entry["ok"] != true || entry["facet"] != "output_evaluator" {
		t.Errorf("log entry = %v", entry)
	}
	if entry["turnIndex"] != float64(2) || entry["gateWouldFire"] != true {
		t.Errorf("log entry missing turn/gate fields: %v", entry)
	}
	if _, hasTags := entry["tags"]; !hasTags {
		t.Errorf("log entry missing tags: %v", entry)
	}
	if agg, _ := entry["aggregate"].(map[string]any); agg["investment"] != float64(4) || agg["valence"] != "negativa" {
		t.Errorf("log aggregate = %v", entry["aggregate"])
	}
}

func TestOutputEvaluator_gateStaysQuietBelowThreshold(t *testing.T) {
	evaluator := &fakeEvaluator{response: oneParagraphEval()} // investment 2, positiva
	e := newTestOutputEvaluator(evaluator, nil)

	evaluateAndWait(e, 0, "boa notícia pequena")

	emotion, ok := e.TurnEmotions()[0]
	if !ok {
		t.Fatal("expected metadata for turn 0")
	}
	if emotion.GateWouldFire {
		t.Errorf("gate fired for %+v with threshold 4", emotion.Aggregate)
	}
}

func TestOutputEvaluator_configurableThreshold(t *testing.T) {
	evaluator := &fakeEvaluator{response: oneParagraphEval()} // investment 2, positiva
	e := newTestOutputEvaluator(evaluator, nil)
	e.gateMinInvestment = 2

	evaluateAndWait(e, 0, "boa notícia pequena")

	if emotion := e.TurnEmotions()[0]; !emotion.GateWouldFire {
		t.Errorf("gate should fire at threshold 2 for %+v", emotion.Aggregate)
	}
}

func TestOutputEvaluator_countMismatchFailsOpen(t *testing.T) {
	// Two paragraphs in, one annotation out: index-based matching would lie.
	evaluator := &fakeEvaluator{response: oneParagraphEval()}
	var log bytes.Buffer
	e := newTestOutputEvaluator(evaluator, &log)

	evaluateAndWait(e, 0, "primeiro parágrafo\n\nsegundo parágrafo")

	if len(e.TurnEmotions()) != 0 {
		t.Errorf("no metadata should be retained on contract violation: %+v", e.TurnEmotions())
	}
	entry := lastLogEntry(t, &log)
	if entry["ok"] != false || !strings.Contains(entry["error"].(string), "1 annotations for 2 paragraphs") {
		t.Errorf("log entry = %v", entry)
	}
}

func TestOutputEvaluator_evaluatorErrorFailsOpen(t *testing.T) {
	evaluator := &fakeEvaluator{err: errors.New("api exploded")}
	var log bytes.Buffer
	e := newTestOutputEvaluator(evaluator, &log)

	evaluateAndWait(e, 1, "um parágrafo qualquer")

	if len(e.TurnEmotions()) != 0 {
		t.Errorf("no metadata on evaluator error: %+v", e.TurnEmotions())
	}
	entry := lastLogEntry(t, &log)
	if entry["ok"] != false || entry["error"] != "api exploded" {
		t.Errorf("log entry = %v", entry)
	}
}

// TestOutputEvaluator_dispatchNeverBlocksOnSlowEvaluator is the
// observability-mode guarantee: EvaluateCompletedTurn returns immediately
// even when the model call is slower than the facet timeout, so the stdout
// pump (and therefore delivery to the gateway) never waits on evaluation.
func TestOutputEvaluator_dispatchNeverBlocksOnSlowEvaluator(t *testing.T) {
	evaluator := &fakeEvaluator{response: oneParagraphEval(), delay: 5 * time.Second}
	var log bytes.Buffer
	e := newTestOutputEvaluator(evaluator, &log)
	e.timeout = 50 * time.Millisecond

	start := time.Now()
	e.EvaluateCompletedTurn(0, "um parágrafo qualquer")
	if elapsed := time.Since(start); elapsed > 100*time.Millisecond {
		t.Fatalf("dispatch blocked for %v", elapsed)
	}

	e.WaitInFlight()
	if len(e.TurnEmotions()) != 0 {
		t.Errorf("no metadata on timeout: %+v", e.TurnEmotions())
	}
	entry := lastLogEntry(t, &log)
	if entry["ok"] != false || !strings.Contains(entry["error"].(string), "context deadline exceeded") {
		t.Errorf("log entry = %v", entry)
	}
}

func TestOutputEvaluator_emptyTurnIsSkipped(t *testing.T) {
	evaluator := &fakeEvaluator{response: oneParagraphEval()}
	var log bytes.Buffer
	e := newTestOutputEvaluator(evaluator, &log)

	evaluateAndWait(e, 0, "   \n\n  ")

	if evaluator.calls != 0 {
		t.Errorf("evaluator called %d times for an empty turn", evaluator.calls)
	}
	entry := lastLogEntry(t, &log)
	if entry["skipped"] != "turn has no assistant text" {
		t.Errorf("log entry = %v", entry)
	}
}

// panicEvaluator triggers the goroutine's recover path: without it a facet
// panic would kill the whole proxy process.
type panicEvaluator struct{}

func (panicEvaluator) Evaluate(_ context.Context, _ string, _ string) (Evaluation, error) {
	panic("boom")
}

func TestOutputEvaluator_panicIsRecoveredAndLogged(t *testing.T) {
	var log bytes.Buffer
	e := newTestOutputEvaluator(panicEvaluator{}, &log)

	evaluateAndWait(e, 0, "um parágrafo")

	entry := lastLogEntry(t, &log)
	if !strings.Contains(entry["error"].(string), "panic: boom") {
		t.Errorf("log entry = %v", entry)
	}
}
