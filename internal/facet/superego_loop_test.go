package facet

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
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
// review, the carried-over tool blocks, and the prior-round annotations the
// ladder surfaces.
type scriptedConversant struct {
	steps       []scriptStep
	calls       int
	gotMessages [][]anthropic.Message
}

func (s *scriptedConversant) Converse(_ context.Context, _ string, messages []anthropic.Message) (Evaluation, error) {
	s.gotMessages = append(s.gotMessages, messages)
	step := s.steps[min(s.calls, len(s.steps)-1)]
	s.calls++
	return step.eval, step.err
}

// redrafter is the core redraft callback double: it returns a fixed sequence of
// redrafts (or errors) and records every ephemeral tail it was handed.
// redraftMsgs, when set, is returned alongside the matching redraft as the
// message blocks that redraft appended while acting (its own tool_use/
// tool_result traffic, excluding its draft), simulating an acting redraft (#4753).
type redrafter struct {
	drafts      []string
	errs        []error
	redraftMsgs [][]anthropic.Message
	calls       int
	tails       [][]anthropic.Message
}

func (r *redrafter) fn(_ context.Context, tail []anthropic.Message) (string, []anthropic.Message, error) {
	r.tails = append(r.tails, tail)
	i := r.calls
	r.calls++
	if i < len(r.errs) && r.errs[i] != nil {
		return "", nil, r.errs[i]
	}
	var msgs []anthropic.Message
	if i < len(r.redraftMsgs) {
		msgs = r.redraftMsgs[i]
	}
	return r.drafts[min(i, len(r.drafts)-1)], msgs, nil
}

// messagesContain reports whether any message's content carries substr, flattening
// string content directly and JSON-marshaling block content (tool_use raw blocks,
// tool_result blocks) so a test can assert on the tool provenance the superego saw.
func messagesContain(msgs []anthropic.Message, substr string) bool {
	for _, m := range msgs {
		if s, ok := m.Content.(string); ok {
			if strings.Contains(s, substr) {
				return true
			}
			continue
		}
		if b, err := json.Marshal(m.Content); err == nil && strings.Contains(string(b), substr) {
			return true
		}
	}
	return false
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

// govMessages is the core's own conversation array for the reviewed turn,
// EXCLUDING the final assistant draft (the governor is handed exactly that, the
// draft arriving separately). A single prior user message is enough for the
// ladder tests that do not exercise carried-over tool provenance.
func govMessages() []anthropic.Message {
	return []anthropic.Message{{Role: "user", Content: "oi"}}
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

			text, held := gov.GovernOutput(context.Background(), 0, govMessages(), draft, rd.fn)

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

	gov.GovernOutput(context.Background(), 0, govMessages(), draft, rd.fn)

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

// TestActiveSuperego_redraftToolCallsReachNextReview locks in the provenance
// fix for acting redrafts (#4753): the original turn's tool_use/tool_result
// blocks ride the core's message array into round 1's review, and when a redraft
// runs its own tools the NEXT round's review must carry those new blocks too —
// as the real provider-shaped messages the core consumed, not a summary — so the
// superego judges the redraft against the evidence it just gathered. The caller's
// message slice must stay untouched (the governor clones before appending).
func TestActiveSuperego_redraftToolCallsReachNextReview(t *testing.T) {
	const draft = "rascunho original"
	gov, _, co := newTestGovernor(
		[]scriptStep{gateFires(), gateFires()},
		[]scriptStep{superegoVerdictStep("revise"), superegoVerdictStep("approve")},
		nil,
	)
	// The redraft "acts": it returns the tool_use/tool_result blocks it ran (the
	// governor folds them into the review view for the next round).
	rd := &redrafter{
		drafts: []string{"rev2 com evidência"},
		redraftMsgs: [][]anthropic.Message{{
			{Role: "assistant", Content: []json.RawMessage{json.RawMessage(
				`{"type":"tool_use","id":"t-redraft","name":"web_fetch","input":{"url":"https://arxiv.org/abs/2604.14228"}}`)}},
			{Role: "user", Content: []anthropic.ToolResultBlock{
				anthropic.NewToolResultBlock("t-redraft", "HTTP 200: título real", false)}},
		}},
	}
	// The core's own array for the turn (excluding the final draft): a user turn,
	// then the original turn's web_search tool_use + tool_result blocks.
	messages := []anthropic.Message{
		{Role: "user", Content: "oi"},
		{Role: "assistant", Content: []json.RawMessage{json.RawMessage(
			`{"type":"tool_use","id":"t-orig","name":"web_search","input":{"query":"arxiv"}}`)}},
		{Role: "user", Content: []anthropic.ToolResultBlock{
			anthropic.NewToolResultBlock("t-orig", "resultados", false)}},
	}

	text, held := gov.GovernOutput(context.Background(), 0, messages, draft, rd.fn)

	if held || text != "rev2 com evidência" {
		t.Fatalf("GovernOutput = (%q, %v), want the approved redraft", text, held)
	}
	if len(co.gotMessages) != 2 {
		t.Fatalf("superego reviewed %d times, want 2", len(co.gotMessages))
	}
	round1, round2 := co.gotMessages[0], co.gotMessages[1]
	// Round 1 sees the original turn's tool blocks only.
	if !messagesContain(round1, "web_search") {
		t.Errorf("round 1 missing the original turn's tool provenance: %+v", round1)
	}
	if messagesContain(round1, "web_fetch") {
		t.Errorf("round 1 must not see the not-yet-run redraft tool: %+v", round1)
	}
	// Round 2 sees the original blocks PLUS the redraft's new evidence.
	if !messagesContain(round2, "web_search") || !messagesContain(round2, "web_fetch") {
		t.Errorf("round 2 missing original or redraft tool provenance: %+v", round2)
	}
	// The reviewed draft always enters as the closing draft-user message, never as
	// an assistant message — the one preserved asymmetry (#4354).
	for _, round := range [][]anthropic.Message{round1, round2} {
		final := round[len(round)-1]
		s, _ := final.Content.(string)
		if final.Role != "user" || !strings.Contains(s, "<<<RASCUNHO_SOB_REVISAO_INICIO>>>") {
			t.Errorf("closing message is not the draft-user: %+v", final)
		}
	}
	// The caller's message slice was never mutated by the fold.
	if got := len(messages); got != 3 {
		t.Errorf("caller messages mutated: now %d messages, want 3", got)
	}
	if messagesContain(messages, "web_fetch") {
		t.Errorf("caller messages mutated: redraft tool leaked into the caller's slice: %+v", messages)
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

	gov.GovernOutput(context.Background(), 0, govMessages(), draft, (&redrafter{}).fn)

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

	text, held := gov.GovernOutput(context.Background(), 0, govMessages(), draft, (&redrafter{}).fn)
	if text != draft || held {
		t.Errorf("panic must fail open to the original draft: text=%q held=%v", text, held)
	}
}
