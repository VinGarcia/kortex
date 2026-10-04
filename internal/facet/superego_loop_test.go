package facet

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/history"
)

// scriptStep is one canned model reply: an evaluation or an error. A scripted
// fake walks its steps per call and repeats the last one past the end, so a
// test only lists the steps that differ.
type scriptStep struct {
	eval Evaluation
	err  error
}

// scriptedEvaluator feeds EvaluateDraft a fixed sequence of evaluator replies.
type scriptedEvaluator struct {
	steps []scriptStep
	calls int
}

func (s *scriptedEvaluator) Evaluate(_ context.Context, _ string, _ string) (Evaluation, error) {
	step := s.steps[min(s.calls, len(s.steps)-1)]
	s.calls++
	return step.eval, step.err
}

// scriptedConversant feeds ReviewDraft a fixed sequence of superego replies and
// records each round's assembled messages so a test can assert on the draft-under
// review and the prior-round annotations the ladder surfaces.
type scriptedConversant struct {
	steps       []scriptStep
	calls       int
	gotMessages [][]ReviewMessage
}

func (s *scriptedConversant) Converse(_ context.Context, _ string, messages []ReviewMessage) (Evaluation, error) {
	s.gotMessages = append(s.gotMessages, messages)
	step := s.steps[min(s.calls, len(s.steps)-1)]
	s.calls++
	return step.eval, step.err
}

// redrafter is the core redraft callback double: it returns a fixed sequence of
// redrafts (or errors) and records every ephemeral tail it was handed.
type redrafter struct {
	drafts []string
	errs   []error
	calls  int
	tails  [][]anthropic.Message
}

func (r *redrafter) fn(_ context.Context, tail []anthropic.Message) (string, error) {
	r.tails = append(r.tails, tail)
	i := r.calls
	r.calls++
	if i < len(r.errs) && r.errs[i] != nil {
		return "", r.errs[i]
	}
	return r.drafts[min(i, len(r.drafts)-1)], nil
}

// gateFires / gateQuiet are single-paragraph evaluator replies whose aggregate
// does (investment 4, negativa) or does not (investment 0, neutra) trip the gate.
func gateFires() scriptStep {
	return scriptStep{eval: evalText(`[{"investment":4,"valence":"negativa","emotions":[]}]`)}
}
func gateQuiet() scriptStep {
	return scriptStep{eval: evalText(`[{"investment":0,"valence":"neutra","emotions":[]}]`)}
}

// superegoVerdictStep is a superego reply carrying one structured verdict.
func superegoVerdictStep(verdict string) scriptStep {
	return scriptStep{eval: evalText(fmt.Sprintf(
		`{"verdict":%q,"annotations":[{"paragraph":1,"issue":"promessa vaga","why":"sem caminho"}],"why":"motivo %s"}`,
		verdict, verdict))}
}

// newTestGovernor wires an ActiveSuperego over the two scripted fakes, with a
// short per-call timeout and an optional log sink.
func newTestGovernor(evalSteps []scriptStep, seSteps []scriptStep, logSink *bytes.Buffer) (*ActiveSuperego, *scriptedEvaluator, *scriptedConversant) {
	ev := &scriptedEvaluator{steps: evalSteps}
	co := &scriptedConversant{steps: seSteps}
	eval := NewOutputEvaluator(OutputEvaluatorParams{
		Evaluator: ev, SystemPrompt: "avalie", Timeout: time.Second,
		Model: "haiku", GateMinInvestment: 4,
	})
	se := NewSuperego(SuperegoParams{
		Conversant: co, SystemPrompt: "superego", Timeout: time.Second, Model: "opus",
	})
	gov := NewActiveSuperego(ActiveSuperegoParams{
		Evaluator: eval, Superego: se, Model: "opus",
	})
	if logSink != nil {
		// Re-point every facet log at the shared sink so the test can read the
		// gate and ladder decisions the loop records.
		eval.log.sink = logSink
		se.log.sink = logSink
		gov.log.sink = logSink
	}
	return gov, ev, co
}

func govSnapshot(draft string) history.Snapshot {
	return history.Snapshot{Turns: []history.Turn{{UserText: "oi", AssistantText: draft, Completed: true}}}
}

func TestActiveSuperego_ladder(t *testing.T) {
	const draft = "rascunho original"
	tests := []struct {
		desc       string
		evalSteps  []scriptStep
		seSteps    []scriptStep
		redrafts   []string
		redraftErr []error
		wantText   string
		wantHeld   bool
		wantRedraw int
		wantReview int
	}{
		{
			desc:       "gate quiet delivers the draft without any review",
			evalSteps:  []scriptStep{gateQuiet()},
			seSteps:    []scriptStep{superegoVerdictStep("revise")},
			wantText:   draft,
			wantReview: 0,
		},
		{
			desc:       "gate fires, superego approves round 1, draft delivered",
			evalSteps:  []scriptStep{gateFires()},
			seSteps:    []scriptStep{superegoVerdictStep("approve")},
			wantText:   draft,
			wantReview: 1,
		},
		{
			desc:       "revise then approve delivers the redraft",
			evalSteps:  []scriptStep{gateFires(), gateFires()},
			seSteps:    []scriptStep{superegoVerdictStep("revise"), superegoVerdictStep("approve")},
			redrafts:   []string{"rascunho revisado"},
			wantText:   "rascunho revisado",
			wantRedraw: 1,
			wantReview: 2,
		},
		{
			desc:       "hold_ask_human delivers the hold message, not the draft",
			evalSteps:  []scriptStep{gateFires()},
			seSteps:    []scriptStep{superegoVerdictStep("hold_ask_human")},
			wantText:   "[kortex] Segurei",
			wantHeld:   true,
			wantReview: 1,
		},
		{
			desc:       "revise every round overruns at the cap and delivers the last redraft",
			evalSteps:  []scriptStep{gateFires(), gateFires(), gateFires(), gateFires()},
			seSteps:    []scriptStep{superegoVerdictStep("revise"), superegoVerdictStep("revise"), superegoVerdictStep("revise"), superegoVerdictStep("revise")},
			redrafts:   []string{"rev2", "rev3", "rev4"},
			wantText:   "rev4",
			wantRedraw: 3,
			wantReview: 4,
		},
		{
			desc:       "exact-identity redraft trips the anti-cycle guard and delivers it",
			evalSteps:  []scriptStep{gateFires()},
			seSteps:    []scriptStep{superegoVerdictStep("revise")},
			redrafts:   []string{draft}, // byte-equal to the original
			wantText:   draft,
			wantRedraw: 1,
			wantReview: 1,
		},
		{
			desc:       "evaluator failure before the gate fails open to the draft",
			evalSteps:  []scriptStep{{err: errors.New("eval down")}},
			seSteps:    []scriptStep{superegoVerdictStep("revise")},
			wantText:   draft,
			wantReview: 0,
		},
		{
			desc:       "superego failure fails open to the draft",
			evalSteps:  []scriptStep{gateFires()},
			seSteps:    []scriptStep{{err: errors.New("superego down")}},
			wantText:   draft,
			wantReview: 1,
		},
		{
			desc:       "redraft failure fails open to the current draft",
			evalSteps:  []scriptStep{gateFires()},
			seSteps:    []scriptStep{superegoVerdictStep("revise")},
			redraftErr: []error{errors.New("core down")},
			redrafts:   []string{"never used"},
			wantText:   draft,
			wantRedraw: 1,
			wantReview: 1,
		},
		{
			desc:       "unparseable verdict fails open to the current draft",
			evalSteps:  []scriptStep{gateFires()},
			seSteps:    []scriptStep{{eval: evalText("prosa sem verdict")}},
			wantText:   draft,
			wantReview: 1,
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			gov, _, co := newTestGovernor(test.evalSteps, test.seSteps, nil)
			rd := &redrafter{drafts: test.redrafts, errs: test.redraftErr}

			text, held := gov.GovernOutput(context.Background(), 0, govSnapshot(draft), draft, rd.fn)

			if !strings.Contains(text, test.wantText) {
				t.Errorf("delivered text = %q, want containing %q", text, test.wantText)
			}
			if held != test.wantHeld {
				t.Errorf("held = %v, want %v", held, test.wantHeld)
			}
			if rd.calls != test.wantRedraw {
				t.Errorf("redraft called %d times, want %d", rd.calls, test.wantRedraw)
			}
			if co.calls != test.wantReview {
				t.Errorf("superego reviewed %d times, want %d", co.calls, test.wantReview)
			}
		})
	}
}

// TestActiveSuperego_ephemeralTailGrows asserts the redraft tail accumulates the
// prior critiques as user feedback interleaved with the prior redrafts as
// assistant turns, and that the original draft is NOT repeated in the tail (it is
// already the last canonical assistant message the callback appends to).
func TestActiveSuperego_ephemeralTailGrows(t *testing.T) {
	const draft = "rascunho original"
	gov, _, _ := newTestGovernor(
		[]scriptStep{gateFires(), gateFires(), gateFires(), gateFires()},
		[]scriptStep{superegoVerdictStep("revise"), superegoVerdictStep("revise"), superegoVerdictStep("revise"), superegoVerdictStep("revise")},
		nil,
	)
	rd := &redrafter{drafts: []string{"rev2", "rev3", "rev4"}}

	gov.GovernOutput(context.Background(), 0, govSnapshot(draft), draft, rd.fn)

	if len(rd.tails) != 3 {
		t.Fatalf("expected 3 redraft tails, got %d", len(rd.tails))
	}
	// Tail lengths grow 1, 3, 5: each round adds a (user critique, assistant
	// redraft) pair, and the newest critique closes the tail with no assistant
	// after it (that is what the next redraft answers).
	wantLens := []int{1, 3, 5}
	for i, tail := range rd.tails {
		if len(tail) != wantLens[i] {
			t.Errorf("tail %d length = %d, want %d: %+v", i, len(tail), wantLens[i], tail)
		}
		if tail[len(tail)-1].Role != "user" {
			t.Errorf("tail %d must end with the critique as a user message, got role %q", i, tail[len(tail)-1].Role)
		}
	}
	// The original draft must never appear verbatim in any tail — it lives only
	// in the canonical messages the callback appends the tail to.
	for i, tail := range rd.tails {
		for _, msg := range tail {
			if content, ok := msg.Content.(string); ok && content == draft {
				t.Errorf("tail %d leaked the original draft as a message: %+v", i, msg)
			}
		}
	}
	// The second tail interleaves: [user crit1, assistant rev2, user crit2].
	second := rd.tails[1]
	if second[0].Role != "user" || !strings.Contains(second[0].Content.(string), "Revisão do superego") {
		t.Errorf("tail 1 position 0 should be the round-1 critique: %+v", second[0])
	}
	if second[1].Role != "assistant" || second[1].Content.(string) != "rev2" {
		t.Errorf("tail 1 position 1 should be the first redraft rev2: %+v", second[1])
	}
}

// TestActiveSuperego_logsGateAndDecisions checks the loop logs the entry gate
// decision and the per-round ladder decision with the path tag.
func TestActiveSuperego_logsGateAndDecisions(t *testing.T) {
	const draft = "rascunho original"
	var log bytes.Buffer
	gov, _, _ := newTestGovernor(
		[]scriptStep{gateFires()},
		[]scriptStep{superegoVerdictStep("approve")},
		&log,
	)

	gov.GovernOutput(context.Background(), 0, govSnapshot(draft), draft, (&redrafter{}).fn)

	out := log.String()
	// The gate line: active_superego facet, primary path, gateWouldFire true.
	if !strings.Contains(out, `"facet":"active_superego"`) || !strings.Contains(out, `"path":"primary"`) {
		t.Errorf("missing active_superego/primary log line:\n%s", out)
	}
	if !strings.Contains(out, `"gateWouldFire":true`) {
		t.Errorf("gate decision not logged:\n%s", out)
	}
	if !strings.Contains(out, `"decision":"deliver"`) {
		t.Errorf("approve round should log a deliver decision:\n%s", out)
	}
}

// TestActiveSuperego_panicFailsOpen ensures a panic deep in the loop degrades to
// delivering the original draft, never a crash.
func TestActiveSuperego_panicFailsOpen(t *testing.T) {
	const draft = "rascunho original"
	eval := NewOutputEvaluator(OutputEvaluatorParams{
		Evaluator: &scriptedEvaluator{steps: []scriptStep{gateFires()}},
		SystemPrompt: "avalie", Timeout: time.Second, Model: "haiku", GateMinInvestment: 4,
	})
	se := NewSuperego(SuperegoParams{
		Conversant: panicConversant{}, SystemPrompt: "superego", Timeout: time.Second, Model: "opus",
	})
	gov := NewActiveSuperego(ActiveSuperegoParams{Evaluator: eval, Superego: se, Model: "opus"})

	text, held := gov.GovernOutput(context.Background(), 0, govSnapshot(draft), draft, (&redrafter{}).fn)
	if text != draft || held {
		t.Errorf("panic must fail open to the original draft: text=%q held=%v", text, held)
	}
}
