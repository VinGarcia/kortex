package facet

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vingarcia/kortex/internal/history"
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
// evaluation to land, so assertions see the final state. The snapshot holds
// the evaluated text at turnIndex, padded with filler turns before it.
func evaluateAndWait(e *OutputEvaluator, turnIndex int, text string) {
	turns := make([]history.Turn, turnIndex+1)
	turns[turnIndex] = history.Turn{AssistantText: text, Completed: true}
	e.EvaluateCompletedTurn(turnIndex, history.Snapshot{Turns: turns})
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

// TestOutputEvaluator_failureModes drives every fail-open branch through a
// single table (the local form — see the annotator's failure-modes table):
// each failure retains NO metadata and ends in a log entry carrying the
// error, and nothing else happens.
func TestOutputEvaluator_failureModes(t *testing.T) {
	tests := []struct {
		desc      string
		evaluator *fakeEvaluator
		text      string
		wantErr   string
	}{
		{
			desc:      "evaluator API error",
			evaluator: &fakeEvaluator{err: errors.New("api exploded")},
			text:      "um parágrafo qualquer",
			wantErr:   "api exploded",
		},
		{
			desc:      "evaluator slower than the facet timeout",
			evaluator: &fakeEvaluator{response: oneParagraphEval(), delay: 5 * time.Second},
			text:      "um parágrafo qualquer",
			wantErr:   "context deadline exceeded",
		},
		{
			desc:      "no JSON array in the output",
			evaluator: &fakeEvaluator{response: evalText("desculpa, não consigo")},
			text:      "um parágrafo qualquer",
			wantErr:   "no JSON array in evaluator output",
		},
		{
			desc:      "malformed JSON array",
			evaluator: &fakeEvaluator{response: evalText(`[{"investment":"alta"}]`)},
			text:      "um parágrafo qualquer",
			wantErr:   "not the expected JSON array",
		},
		{
			// Two paragraphs in, one annotation out: index matching would lie.
			desc:      "annotation count mismatch",
			evaluator: &fakeEvaluator{response: oneParagraphEval()},
			text:      "primeiro parágrafo\n\nsegundo parágrafo",
			wantErr:   "1 annotations for 2 paragraphs",
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			var log bytes.Buffer
			e := newTestOutputEvaluator(test.evaluator, &log)
			e.timeout = 50 * time.Millisecond

			evaluateAndWait(e, 0, test.text)

			if len(e.TurnEmotions()) != 0 {
				t.Errorf("no metadata should be retained on failure: %+v", e.TurnEmotions())
			}
			entry := lastLogEntry(t, &log)
			if entry["ok"] != false || !strings.Contains(entry["error"].(string), test.wantErr) {
				t.Errorf("log entry = %v, want error containing %q", entry, test.wantErr)
			}
		})
	}
}

// TestOutputEvaluator_dispatchNeverBlocksOnSlowEvaluator is the
// observability-mode guarantee: EvaluateCompletedTurn returns immediately
// even when the model call is slower than the facet timeout, so the stdout
// pump (and therefore delivery to the gateway) never waits on evaluation.
func TestOutputEvaluator_dispatchNeverBlocksOnSlowEvaluator(t *testing.T) {
	evaluator := &fakeEvaluator{response: oneParagraphEval(), delay: 5 * time.Second}
	e := newTestOutputEvaluator(evaluator, nil)
	e.timeout = 50 * time.Millisecond

	start := time.Now()
	e.EvaluateCompletedTurn(0, history.Snapshot{Turns: []history.Turn{{AssistantText: "um parágrafo qualquer", Completed: true}}})
	if elapsed := time.Since(start); elapsed > 100*time.Millisecond {
		t.Fatalf("dispatch blocked for %v", elapsed)
	}
	e.WaitInFlight()
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

// TestOutputEvaluator_superegoTrigger covers the F2d dispatch decision end
// to end through the real constructor plumbing: a real Superego (fake
// conversant behind it) fires exactly once per gate hit and receives the
// draft-user with the evaluator's own tags interleaved; below the threshold
// the conversant is never called.
func TestOutputEvaluator_superegoTrigger(t *testing.T) {
	tests := []struct {
		desc     string
		response Evaluation
		text     string
		wantFire bool
	}{
		{
			desc: "gate fires, superego reviews the draft with matching tags",
			response: evalText(`[
				{"investment":4,"valence":"negativa","emotions":[{"emotion":"recusa","level":4,"about":"não vou"}]},
				{"investment":0,"valence":"neutra","emotions":[]}
			]`),
			text:     "não vou fazer isso\n\nrodei o script",
			wantFire: true,
		},
		{
			desc:     "gate quiet, superego never runs",
			response: oneParagraphEval(), // investment 2, positiva
			text:     "boa notícia pequena",
			wantFire: false,
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			conversant := &fakeConversant{response: evalText(`{"verdict":"approve","annotations":[],"why":"ok"}`)}
			superego := newTestSuperego(conversant, nil)
			e := NewOutputEvaluator(OutputEvaluatorParams{
				Evaluator:         &fakeEvaluator{response: test.response},
				SystemPrompt:      "avalie a carga da resposta",
				Timeout:           time.Second,
				Model:             "claude-haiku-4-5-20251001",
				GateMinInvestment: 4,
				Superego:          superego,
			})

			evaluateAndWait(e, 0, test.text)
			superego.WaitInFlight()

			if !test.wantFire {
				if conversant.calls != 0 {
					t.Fatalf("superego conversant called %d times, want 0", conversant.calls)
				}
				return
			}
			if conversant.calls != 1 {
				t.Fatalf("superego conversant called %d times, want 1", conversant.calls)
			}
			final := conversant.gotMessages[len(conversant.gotMessages)-1]
			if !strings.Contains(final.Content, "<<<RASCUNHO_SOB_REVISAO_INICIO>>>") ||
				!strings.Contains(final.Content, "não vou fazer isso\n[emoções p1: investment=4 valence=negativa emotions=recusa(4)]") {
				t.Errorf("draft-user message = %q", final.Content)
			}
			if critique, ok := superego.Critiques()[0]; !ok || critique.Verdict != "approve" {
				t.Errorf("critique for turn 0 = %+v, %v", critique, ok)
			}
		})
	}
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
