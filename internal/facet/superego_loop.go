package facet

import (
	"context"
	"fmt"
	"io"
	"slices"
	"strings"

	"github.com/vingarcia/kortex/internal/anthropic"
)

// ActiveSuperego is the F2d facet in ACTIVE MODE: the blocking core↔superego
// loop that governs the turn's delivered text. Where shadow mode (Superego.Review)
// observes after the fact and only logs, active mode runs BEFORE the final
// assistant/result events reach the wire and decides what actually gets
// delivered:
//
//  1. Evaluate the core's draft (output emotion evaluator → deterministic gate).
//  2. Gate does not fire → deliver the draft untouched.
//  3. Gate fires → run the superego ladder: the superego reviews the draft;
//     "approve" delivers it, "revise" feeds the critique back to the CORE model
//     which produces a new draft, and the loop repeats. Verdicts are
//     approve/revise in EVERY round (there is no hold — superego aconselha, ego
//     decide).
//  4. Round cap reached with a residual "revise" → the ladder is spent, so the
//     EGO (the core) authors the final message: it is handed the full ephemeral
//     context plus a deterministic instruction to explain the impasse to the
//     user in its own words (finalRoundInstruction), and that text is delivered
//     WITHOUT a further review. The cap itself stays (an unbounded loop would be
//     both cruel and costly).
//
// EPHEMERAL TAIL (the architectural point, design #4354): the rejected drafts
// and the superego critiques live only in a temporary message tail passed to
// the redraft/finalize callbacks for the duration of the loop. They are NEVER
// persisted to canonical history and NEVER emitted on the stream — only the
// single clean delivered text is. The session owns discarding the tail; this
// governor never returns it.
//
// FAIL-OPEN (the facet invariant): ANY failure — evaluator error, superego
// error, redraft error, finalize error, malformed verdict, panic — delivers a
// draft (the latest one in hand) and logs the failure. The loop never holds a
// reply hostage to its own malfunction.
//
// The failure is deliberately NOT surfaced in the delivered text (the
// u-never-fail-silently tension): an in-band "[review failed]" notice would
// break both the EPHEMERAL TAIL invariant above — the superego machinery must
// stay invisible to the user — and the persona of an intimate assistant reply.
// So fail-open is silent to the END USER by design, but never to operations:
// every path records a structured facet-log line (Decision: "fail_open" with
// the error), which is where the degradation is observed.
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

// DefaultLadderRounds is the ladder cap: every round may approve or revise, and
// a residual revise at the cap spends the ladder and hands the core the final
// message (see GovernOutput). The cap is kept deliberately — an unbounded
// core↔superego loop would be both cruel (≈ pain) and costly.
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
// to deliver. Every path (deliver, revise-then-deliver, core-authored final
// message, fail-open) returns the text to speak; the loop never holds a reply.
//
// messages is the core's OWN conversation array for this turn — the exact
// []anthropic.Message the core consumed, EXCLUDING the final assistant draft
// (design #4351: the superego reviews from the identical context, tool_use and
// tool_result blocks and all; draft is passed separately and re-enters as a
// draft-user message inside ReviewDraft).
//
// ephemeralDigest is the turn's tone digest — the SAME ephemeral block the core
// received — forwarded into every review so the superego judges against the
// user's emotional tone too, not only the draft's own output tags. It is
// ephemeral in the review exactly as in the core (see ReviewDraft).
//
// redraft asks the CORE model for a new draft given an ephemeral tail (the prior
// critiques as user feedback and the prior redrafts as assistant turns, each
// critique preceded by the tagged draft it reviewed) appended to the turn's
// canonical messages; the session never persists that tail. Alongside the new
// draft text, redraft returns the message blocks it appended while acting
// (#4753: the redraft's own tool_use/tool_result traffic, EXCLUDING its draft),
// which this loop folds into the running review view so the next round's
// superego sees the evidence the redraft just gathered.
//
// finalize asks the CORE model for the final message when the ladder is spent (a
// residual revise at the round cap): it is handed the ephemeral tail ending with
// the deterministic finalRoundInstruction and returns the text to deliver, with
// no tools and no further review. The caller owns the context deadline for the
// whole loop; the evaluator and superego each own their per-call timeout
// internally.
func (a *ActiveSuperego) GovernOutput(
	ctx context.Context,
	turnIndex int,
	messages []anthropic.Message,
	ephemeralDigest string,
	draft string,
	redraft func(ctx context.Context, tail []anthropic.Message) (string, []anthropic.Message, error),
	finalize func(ctx context.Context, tail []anthropic.Message) (string, error),
) (text string) {
	// Fail-open umbrella: a panic anywhere in the loop degrades to delivering
	// the original draft, never a crashed wire or a swallowed reply.
	text = draft
	defer func() {
		if r := recover(); r != nil {
			a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Path: loopPathPrimary, Decision: "fail_open", Err: fmt.Sprintf("panic: %v", r)})
			text = draft
		}
	}()

	de, ok := a.evaluator.EvaluateDraft(ctx, turnIndex, draft)
	if !ok {
		// The evaluator failed (logged inside EvaluateDraft); without a gate
		// decision there is nothing to govern — deliver the draft.
		a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Path: loopPathPrimary, Decision: "fail_open"})
		return draft
	}
	a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Path: loopPathPrimary, GateWouldFire: &de.GateWouldFire, Decision: gateDecision(de.GateWouldFire)})
	if !de.GateWouldFire {
		return draft
	}

	current := draft
	curEval := de
	var critiques []SuperegoCritique // critique1, critique2, ... (one per round run)
	var redrafts []string            // draft2, draft3, ... (original draft is passed as `draft`)
	// reviewedTagged parallels critiques: the tagged draft-under-review rendering
	// (the exact text the superego judged that round) the redraft tail replays to
	// the core before each critique, so the core and the superego discuss the same
	// tagged facts. Ephemeral like the rest of the tail (#4354).
	var reviewedTagged []string
	// reviewMessages is the running review view: the core's array, grown each
	// revise round with the redraft's own tool traffic so the next review sees
	// the evidence it gathered (#4753). The draft itself is never in here — it
	// always enters as the draft-user message inside ReviewDraft.
	reviewMessages := messages

	for round := 1; round <= a.maxRounds; round++ {
		crit, ok := a.superego.ReviewDraft(ctx, turnIndex, reviewMessages, ephemeralDigest, round, critiques, curEval.Paragraphs, curEval.Tags)
		if !ok {
			// The superego call failed (logged inside ReviewDraft) — fail open.
			a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Round: round, Path: loopPathPrimary, Decision: "fail_open"})
			return current
		}

		switch crit.Verdict {
		case VerdictApprove:
			a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Round: round, Path: loopPathPrimary, Decision: "deliver"})
			return current

		case VerdictRevise:
			// Record the critique and the tagged draft it judged (kept parallel) so
			// the redraft tail — and the final round below — replay the same facts.
			critiques = append(critiques, crit)
			reviewedTagged = append(reviewedTagged, draftUnderReviewMessage(curEval.Paragraphs, curEval.Tags))

			if round >= a.maxRounds {
				// Cap reached with a residual revise: the ladder is spent, so the ego
				// (the core) — not the superego — authors the final message. It gets
				// the full ephemeral context plus the deterministic instruction to
				// explain the impasse in its own words, and that text is delivered
				// with NO further review.
				tail := append(revisionTail(critiques, redrafts, reviewedTagged), anthropic.Message{
					Role:    "user",
					Content: finalRoundInstruction(),
				})
				finalText, err := finalize(ctx, tail)
				if err != nil {
					// A finalize failure fails open to the best draft in hand.
					a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Round: round, Path: loopPathPrimary, Decision: "fail_open", Err: err.Error()})
					return current
				}
				a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Round: round, Path: loopPathPrimary, Decision: "final_round"})
				return finalText
			}
			newDraft, newMessages, err := redraft(ctx, revisionTail(critiques, redrafts, reviewedTagged))
			if err != nil {
				// A redraft failure fails open to the best draft in hand.
				a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Round: round, Path: loopPathPrimary, Decision: "fail_open", Err: err.Error()})
				return current
			}
			// Fold the redraft's own tool traffic (#4753) into the review view so
			// the next round's superego sees the evidence this redraft gathered;
			// without it the superego keeps rejecting a redraft for the very facts
			// it just verified. slices.Clone keeps the caller's array unmutated.
			if len(newMessages) > 0 {
				reviewMessages = append(slices.Clone(reviewMessages), newMessages...)
			}
			if newDraft == current {
				// Exact-identity anti-cycle guard (ratified option (b)): the core
				// re-emitted a byte-equal draft, so another round cannot converge —
				// stop and deliver it rather than burn a round on an identical text.
				a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Round: round, Path: loopPathPrimary, Decision: "deliver", Err: "anti-cycle: identical redraft"})
				return current
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
				return newDraft
			}
			current = newDraft
			curEval = newEval

		default:
			// Empty or unparseable verdict: shadow mode retains the free text, but
			// the active loop has no actionable decision — fail open and deliver.
			a.log.record(callLog{Facet: activeSuperegoFacet, Model: a.model, TurnIndex: &turnIndex, Round: round, Path: loopPathPrimary, Decision: "fail_open"})
			return current
		}
	}

	// Unreachable: round == maxRounds with revise authors the final message above,
	// and every other verdict returns inside the loop. Defensive: deliver the best
	// draft in hand rather than fall off the end silently.
	return current
}

// gateDecision maps the entry gate result to the decision logged for the gate
// line: "revise" when it fires (the ladder runs), "deliver" when it does not.
func gateDecision(fire bool) string {
	if fire {
		return "revise"
	}
	return "deliver"
}

// revisionTail builds the ephemeral message tail handed to the CORE redraft
// callback. Each round contributes, in order: the TAGGED draft-under-review (the
// same rendering the superego judged, from reviewedTagged[i]) as a user message,
// then the superego critique of that draft as user feedback, then — for rounds
// already answered — the redraft as an assistant turn. The tail ends with the
// latest critique (the one the redraft must answer). The tagged draft is
// replayed so the core and the superego discuss the SAME tagged facts: the
// turn's own untagged draft is already the last assistant message of the
// canonical messages this tail is appended to, but it carries no F2c tags, which
// the superego saw. critiques and reviewedTagged are parallel and have one more
// entry than redrafts on entry (the just-appended current round). This tail is
// discarded when the loop ends; it never reaches canonical history or the
// stream (the tags must never contaminate the delivered message or history).
func revisionTail(critiques []SuperegoCritique, redrafts []string, reviewedTagged []string) []anthropic.Message {
	// The two user messages per round (tagged draft, then critique) are emitted
	// as consecutive same-role turns on purpose: they are a single logical unit
	// the core should read together, but kept separate so the critique can stay
	// the LAST tail message the redraft tool loop answers (redraftToolUserMessage).
	// The Anthropic Messages API, which kortex calls directly (internal/anthropic),
	// accepts consecutive same-role messages — this is not an alternation error.
	tail := make([]anthropic.Message, 0, len(critiques)*3)
	for i, crit := range critiques {
		tail = append(tail, anthropic.Message{Role: "user", Content: reviewedTagged[i]})
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

// finalRoundInstruction is the deterministic instruction handed to the CORE
// model for the final round, when the ladder is spent (a residual revise at the
// round cap). It is read by the core model, so it is written as explicit,
// full-sentence Portuguese. It tells the core to STOP revising and instead write
// a single honest message to the user, in its own words and human tone, that
// makes three things clear: (1) there was an internal doubt/conflict about this
// reply that it could not resolve on its own; (2) what it had wanted to say (the
// draft's intent); and (3) what the problem was. It must NOT expose "superego"
// or any internal process/entity as such. The ego decides here — the message is
// delivered with no further review.
func finalRoundInstruction() string {
	return "O limite interno de rodadas de revisão desta resposta foi atingido e um conflito interno sobre ela não foi resolvido. " +
		"Em vez de enviar o rascunho como está ou ficar em silêncio, escreva AGORA, com as suas próprias palavras e em tom humano e pessoal (sem mencionar \"superego\" nem qualquer processo ou entidade interna), uma única mensagem ao usuário que deixe claro, de forma natural: " +
		"(1) que você ficou insegura/em dúvida sobre um ponto desta resposta e não conseguiu resolver sozinha; " +
		"(2) o que você queria ter dito (a intenção do rascunho); e " +
		"(3) qual era o problema que te impediu de enviar a resposta assim. " +
		"Peça ajuda ou mais contexto se fizer sentido. " +
		"Exemplo de tom (não copie literalmente): \"Vini, estou insegura sobre um ponto — eu estava pensando em te falar X, mas acho que essa mensagem não está apropriada por causa de Y e não consegui pensar num jeito melhor de dizer isso ou em como consertar.\" " +
		"Responda apenas com essa mensagem final ao usuário."
}
