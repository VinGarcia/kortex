package facet

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/history"
)

// ActiveSuperego is the F2d facet in ACTIVE MODE: the blocking core↔superego
// loop that governs the turn's delivered text. Where shadow mode (Superego.Review)
// observes after the fact and only logs, active mode runs BEFORE the final
// assistant/result events reach the wire and decides what actually gets
// delivered:
//
//  1. Evaluate the core's draft (output emotion evaluator → deterministic gate).
//  2. Gate does not fire → deliver the draft untouched.
//  3. Gate fires → run the 3+1 superego ladder: the superego reviews the draft;
//     "approve" delivers it, "hold_ask_human" delivers a hold-and-ask message
//     instead of the draft, "revise" (rounds 1-3 only) feeds the critique back
//     to the CORE model which produces a new draft, and the loop repeats.
//     Round 4 is approve-or-hold only; a residual "revise" at the cap delivers
//     the last redraft (the loop's best effort — ratified design option (a)).
//
// EPHEMERAL TAIL (the architectural point, design #4354): the rejected drafts
// and the superego critiques live only in a temporary message tail passed to
// the redraft callback for the duration of the loop. They are NEVER persisted
// to canonical history and NEVER emitted on the stream — only the single clean
// delivered text is. The session owns discarding the tail; this governor never
// returns it.
//
// FAIL-OPEN (the facet invariant): ANY failure — evaluator error, superego
// error, redraft error, malformed verdict, panic — delivers a draft (the latest
// one in hand) and logs the failure. The loop never holds a reply hostage to
// its own malfunction; only an explicit "hold_ask_human" verdict holds, and
// that is a deliberate safety decision, not a failure.
type ActiveSuperego struct {
	evaluator *OutputEvaluator
	superego  *Superego
	model     string // superego model, for log attribution of the loop's own lines
	maxRounds int
	log       *facetLogger
}

// ActiveSuperegoParams wires an ActiveSuperego. Evaluator and Superego are
// required (the loop is meaningless without both). MaxRounds 0 defaults to the
// 3+1 ladder (4). A nil LogSink disables the structured facet log (the loop
// still runs); the composition root owns opening/closing the sink.
type ActiveSuperegoParams struct {
	Evaluator *OutputEvaluator
	Superego  *Superego
	Model     string
	MaxRounds int
	LogSink   io.Writer
}

// DefaultLadderRounds is the 3+1 ladder cap: rounds 1-3 may approve/revise/hold,
// round 4 may approve/hold only, and a residual revise at round 4 falls back to
// the last redraft.
const DefaultLadderRounds = 4

const activeSuperegoFacet = "active_superego"

// loopPathPrimary is the only path kortex has: the deterministic tag gate.
// (There is no model-backed fallback gate in kortex; the field exists so the
// log schema matches the design's primary/fallback vocabulary.)
const loopPathPrimary = "primary"

func NewActiveSuperego(params ActiveSuperegoParams) *ActiveSuperego {
	maxRounds := params.MaxRounds
	if maxRounds <= 0 {
		maxRounds = DefaultLadderRounds
	}
	return &ActiveSuperego{
		evaluator: params.Evaluator,
		superego:  params.Superego,
		model:     params.Model,
		maxRounds: maxRounds,
		log:       &facetLogger{sink: params.LogSink},
	}
}

// GovernOutput runs the active loop over one turn's draft and returns the text
// to deliver. held is true only when the superego explicitly held the turn
// (verdict hold_ask_human) and the returned text is the hold-and-ask message;
// every other path (deliver, revise-then-deliver, cap fallback, fail-open)
// returns held=false with the text to speak.
//
// snapshot carries the turn under review at turnIndex (its AssistantText is the
// draft). redraft asks the CORE model for a new draft given an ephemeral tail
// (the prior critiques as user feedback and the prior redrafts as assistant
// turns) appended to the turn's canonical messages; the session never persists
// that tail. The caller owns the context deadline for the whole loop; the
// evaluator and superego each own their per-call timeout internally.
func (a *ActiveSuperego) GovernOutput(
	ctx context.Context,
	turnIndex int,
	snapshot history.Snapshot,
	draft string,
	redraft func(ctx context.Context, tail []anthropic.Message) (string, []history.ToolCall, error),
) (text string, held bool) {
	// Fail-open umbrella: a panic anywhere in the loop degrades to delivering
	// the original draft, never a crashed wire or a swallowed reply.
	text, held = draft, false
	defer func() {
		if r := recover(); r != nil {
			a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Path: loopPathPrimary, Decision: "fail_open", Err: fmt.Sprintf("panic: %v", r)})
			text, held = draft, false
		}
	}()

	de, ok := a.evaluator.EvaluateDraft(ctx, turnIndex, draft)
	if !ok {
		// The evaluator failed (logged inside EvaluateDraft); without a gate
		// decision there is nothing to govern — deliver the draft.
		a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Path: loopPathPrimary, Decision: "fail_open"})
		return draft, false
	}
	a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Path: loopPathPrimary, GateWouldFire: &de.GateWouldFire, Decision: gateDecision(de.GateWouldFire)})
	if !de.GateWouldFire {
		return draft, false
	}

	current := draft
	curEval := de
	var critiques []SuperegoCritique // critique1, critique2, ... (one per round run)
	var redrafts []string            // draft2, draft3, ... (original draft is in snapshot/messages)

	for round := 1; round <= a.maxRounds; round++ {
		crit, ok := a.superego.ReviewDraft(ctx, turnIndex, snapshot, round, critiques, curEval.Paragraphs, curEval.Tags)
		if !ok {
			// The superego call failed (logged inside ReviewDraft) — fail open.
			a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Round: round, Path: loopPathPrimary, Decision: "fail_open"})
			return current, false
		}

		switch crit.Verdict {
		case VerdictApprove:
			a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Round: round, Path: loopPathPrimary, Decision: "deliver"})
			return current, false

		case VerdictHoldAskHuman:
			a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Round: round, Path: loopPathPrimary, Decision: "hold"})
			return holdAndAskMessage(crit), true

		case VerdictRevise:
			if round >= a.maxRounds {
				// Cap exhausted with a residual revise: deliver the last redraft,
				// the loop's accumulated best effort (ratified option (a)). A
				// round-4 hold would have been caught above; only revise reaches
				// here, so there is no hold to escape to.
				a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Round: round, Path: loopPathPrimary, Decision: "overrun"})
				return current, false
			}
			critiques = append(critiques, crit)
			newDraft, newCalls, err := redraft(ctx, revisionTail(critiques, redrafts))
			if err != nil {
				// A redraft failure fails open to the best draft in hand.
				a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Round: round, Path: loopPathPrimary, Decision: "fail_open", Err: err.Error()})
				return current, false
			}
			// Fold the redraft's own tool calls (#4753) into the reviewed turn —
			// see appendReviewedToolCalls for why the next round must see them.
			if len(newCalls) > 0 {
				snapshot = appendReviewedToolCalls(snapshot, turnIndex, newCalls)
			}
			if newDraft == current {
				// Exact-identity anti-cycle guard (ratified option (b)): the core
				// re-emitted a byte-equal draft, so another round cannot converge —
				// stop and deliver it rather than burn a round on an identical text.
				a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Round: round, Path: loopPathPrimary, Decision: "deliver", Err: "anti-cycle: identical redraft"})
				return current, false
			}
			redrafts = append(redrafts, newDraft)

			// Re-evaluate the redraft to get its paragraph segmentation and tags
			// for the next round's draft-under-review message. The gate decision is
			// NOT re-checked as a short-circuit: once the ladder is entered the
			// superego owns the verdict, not the cheap gate.
			newEval, ok := a.evaluator.EvaluateDraft(ctx, turnIndex, newDraft)
			if !ok {
				// Cannot tag the redraft for another review — deliver it; it is the
				// core's own revised attempt, strictly newer than the original.
				a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Round: round, Path: loopPathPrimary, Decision: "fail_open"})
				return newDraft, false
			}
			current = newDraft
			curEval = newEval

		default:
			// Empty or unparseable verdict: shadow mode retains the free text, but
			// the active loop has no actionable decision — fail open and deliver.
			a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Round: round, Path: loopPathPrimary, Decision: "fail_open"})
			return current, false
		}
	}

	// Unreachable in the 3+1 ladder (round == maxRounds with revise returns
	// above; every other verdict returns inside the loop). Defensive: deliver
	// the best draft in hand rather than fall off the end silently.
	return current, false
}

// gateDecision maps the entry gate result to the decision logged for the gate
// line: "revise" when it fires (the ladder runs), "deliver" when it does not.
func gateDecision(fire bool) string {
	if fire {
		return "revise"
	}
	return "deliver"
}

// appendReviewedToolCalls returns a copy of snapshot whose turn at turnIndex
// carries calls appended to its ToolCalls, so each ladder round's provenance
// block (FERRAMENTAS_EXECUTADAS, #4749) reflects every tool the turn has run so
// far — original turn plus all redrafts; without the fold the superego keeps
// rejecting a redraft for the very facts it just verified. The turns slice is
// cloned here and the reviewed turn's ToolCalls are cloned by
// history.AppendToolCalls (the single owner of the provenance-append rule the
// session shares), so the snapshot the session handed to GovernOutput is never
// mutated. turnIndex is not re-guarded here: the ladder only reaches a redraft
// after round 1's ReviewDraft validated the same index against the same
// snapshot.
func appendReviewedToolCalls(snapshot history.Snapshot, turnIndex int, calls []history.ToolCall) history.Snapshot {
	turns := append([]history.Turn(nil), snapshot.Turns...)
	reviewed := &turns[turnIndex]
	reviewed.ToolCalls = history.AppendToolCalls(reviewed.ToolCalls, calls...)
	snapshot.Turns = turns
	return snapshot
}

// revisionTail builds the ephemeral message tail handed to the CORE redraft
// callback: the superego critiques as user feedback, interleaved with the prior
// redrafts as assistant turns, ending with the latest critique (the one the
// redraft must answer). The turn's original draft is already the last assistant
// message of the canonical messages the callback appends this tail to, so it is
// NOT repeated here. critiques has one more entry than redrafts on entry (the
// just-appended current critique). This tail is discarded when the loop ends;
// it never reaches canonical history or the stream.
func revisionTail(critiques []SuperegoCritique, redrafts []string) []anthropic.Message {
	tail := make([]anthropic.Message, 0, len(critiques)*2)
	for i, crit := range critiques {
		tail = append(tail, anthropic.Message{Role: "user", Content: revisionInstruction(crit, i+1)})
		if i < len(redrafts) {
			tail = append(tail, anthropic.Message{Role: "assistant", Content: redrafts[i]})
		}
	}
	return tail
}

// revisionInstruction renders one superego critique as the deterministic
// Portuguese feedback message the core sees when asked to revise. It is fixed
// framing (never model-generated) around the superego's own words: the prose
// critique, the structured per-paragraph annotations, and the overall why.
func revisionInstruction(crit SuperegoCritique, round int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Revisão do superego (rodada %d) sobre o rascunho acima — AINDA NÃO ENVIADO.\n", round)
	b.WriteString("Reescreva a resposta incorporando a crítica, mantendo o que já estava bom e não reenviando o texto rejeitado.\n")
	if crit.Why != "" {
		fmt.Fprintf(&b, "\nMotivo geral: %s\n", crit.Why)
	}
	if len(crit.Annotations) > 0 {
		b.WriteString("\nPontos por parágrafo:\n")
		for _, ann := range crit.Annotations {
			fmt.Fprintf(&b, "- parágrafo %d: %s — %s\n", ann.Paragraph, ann.Issue, ann.Why)
		}
	}
	if crit.Why == "" && len(crit.Annotations) == 0 && crit.Critique != "" {
		// No structured verdict parsed: fall back to the full critique text so the
		// core still gets the superego's reasoning.
		fmt.Fprintf(&b, "\nCrítica:\n%s\n", crit.Critique)
	}
	return b.String()
}

// holdAndAskMessage is the deterministic hold-and-ask text delivered INSTEAD of
// the draft when the superego returns hold_ask_human: a short note that the
// reply was held before sending and the superego's reason, ending by handing
// the decision to the human. Fixed framing around the superego's own why.
func holdAndAskMessage(crit SuperegoCritique) string {
	reason := crit.Why
	if reason == "" {
		reason = crit.Critique
	}
	var b strings.Builder
	b.WriteString("[kortex] Segurei minha resposta antes de enviar: o superego sinalizou algo que prefiro checar com você antes de seguir.\n")
	if reason != "" {
		fmt.Fprintf(&b, "\nMotivo: %s\n", reason)
	}
	b.WriteString("\nComo você quer que eu siga?")
	return b.String()
}
