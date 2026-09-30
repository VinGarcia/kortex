package facet

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/history"
)

// Superego is the F2d facet in SHADOW MODE: when the output evaluator's
// hypothetical gate fires on a completed turn, the superego re-reads the
// conversation with its own system prompt — the same CANONICAL history the
// core's context is built from (design #4351; canonical means clean final
// text: wire-level extras like input tags and tool traffic are absent by
// design, see superego_context.go) — plus the turn under review reframed as a user-role
// "draft not yet sent" carrying the F2c emotion tags (design #4354), and
// produces one critique. Shadow contract: the critique goes only to the
// facet log and to in-memory per-turn metadata (the structure the F2e
// block→revise loop will consume); nothing blocks, nothing is revised, the
// stream is never touched. One critique per gate hit — no loop here.
//
// Fail-open like every facet: any failure — API error, timeout, malformed
// input — ends in a log entry and nothing else.
type Superego struct {
	conversant      Conversant
	systemPrompt    string
	timeout         time.Duration
	model           string // logged with each call; the conversant owns the actual routing
	maxHistoryTurns int
	log             *facetLogger

	inFlight sync.WaitGroup
	mu       sync.Mutex
	turns    map[int]SuperegoCritique
}

// SuperegoParams wires a Superego. All fields except LogSink are required; a
// nil LogSink disables the structured facet log (the facet still runs). The
// composition root owns opening/closing the sink. MaxHistoryTurns 0 means
// the whole history.
type SuperegoParams struct {
	Conversant      Conversant
	SystemPrompt    string
	Timeout         time.Duration
	Model           string
	MaxHistoryTurns int
	LogSink         io.Writer
}

func NewSuperego(params SuperegoParams) *Superego {
	return &Superego{
		conversant:      params.Conversant,
		systemPrompt:    params.SystemPrompt,
		timeout:         params.Timeout,
		model:           params.Model,
		maxHistoryTurns: params.MaxHistoryTurns,
		log:             &facetLogger{sink: params.LogSink},
		turns:           map[int]SuperegoCritique{},
	}
}

const superegoFacet = "superego"

// superegoMaxTokens bounds the superego response: a prose critique plus a
// small JSON verdict — 4096 tokens is ample without letting a runaway
// response bill an essay.
const superegoMaxTokens = 4096

// ReviewMessage is one message of the superego's assembled context, in
// facet-owned shape: the adapter behind the Conversant port translates it
// to its provider's wire type, keeping the context-assembly service
// decoupled from any provider.
type ReviewMessage struct {
	Role    string // "user" or "assistant"
	Content string
}

// Conversant is the multi-message model call the superego makes: its own
// system prompt over the full conversation message array. It is a separate
// port from Evaluator (single user message) because the superego's whole
// point is seeing the conversation the core saw. Tests substitute a fake.
type Conversant interface {
	Converse(ctx context.Context, system string, messages []ReviewMessage) (Evaluation, error)
}

// AnthropicConversant adapts the anthropic client to the Conversant port.
type AnthropicConversant struct {
	client *anthropic.Client
	model  string
}

func NewAnthropicConversant(client *anthropic.Client, model string) *AnthropicConversant {
	return &AnthropicConversant{client: client, model: model}
}

func (c *AnthropicConversant) Converse(ctx context.Context, system string, messages []ReviewMessage) (Evaluation, error) {
	wire := make([]anthropic.Message, len(messages))
	for i, m := range messages {
		wire[i] = anthropic.Message{Role: m.Role, Content: m.Content}
	}
	resp, err := c.client.CreateMessage(ctx, anthropic.MessageRequest{
		Model:     c.model,
		System:    system,
		MaxTokens: superegoMaxTokens,
		Messages:  wire,
	})
	if err != nil {
		return Evaluation{}, err
	}
	return Evaluation{
		Text:         resp.Text,
		StopReason:   resp.StopReason,
		InputTokens:  resp.Usage.InputTokens,
		OutputTokens: resp.Usage.OutputTokens,
	}, nil
}

// SuperegoCritique is the retained per-turn shadow outcome: the full model
// text always, plus the structured verdict when the response carried the
// parseable JSON object of the superego prompt contract
// ({"verdict","annotations","why"}). This is the structure the F2e loop
// will consume alongside TurnEmotion.
type SuperegoCritique struct {
	TurnIndex int    `json:"turnIndex"`
	Critique  string `json:"critique"`
	// Verdict is "approve", "revise" or "hold_ask_human" when parsed; ""
	// when the response carried no valid verdict object (the critique text
	// is still retained — shadow mode never discards the model's judgment).
	Verdict     string               `json:"verdict,omitempty"`
	Annotations []SuperegoAnnotation `json:"annotations,omitempty"`
	Why         string               `json:"why,omitempty"`
}

// SuperegoAnnotation is one paragraph-level issue of a structured verdict.
type SuperegoAnnotation struct {
	Paragraph int    `json:"paragraph"`
	Issue     string `json:"issue"`
	Why       string `json:"why"`
}

// Review is the gate-hit hook the output evaluator calls. It dispatches the
// review on its own goroutine and returns immediately (the caller is itself
// an async facet goroutine, but its lifetime must not stretch to a 60s
// superego call). draftParagraphs and tags come from the SAME segmentation
// the evaluator committed to, so index i tags paragraph i.
func (s *Superego) Review(turnIndex int, snapshot history.Snapshot, draftParagraphs []string, tags []ParagraphAnnotation) {
	s.inFlight.Add(1)
	go func() {
		defer s.inFlight.Done()
		s.review(turnIndex, snapshot, draftParagraphs, tags)
	}()
}

// WaitInFlight blocks until every dispatched review finished. The
// composition root calls it after the child exited so a one-shot run does
// not kill the final turn's review by exiting.
func (s *Superego) WaitInFlight() {
	s.inFlight.Wait()
}

// Critiques returns a copy of the retained per-turn critiques, keyed by turn
// index. Turns whose review failed are absent.
func (s *Superego) Critiques() map[int]SuperegoCritique {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make(map[int]SuperegoCritique, len(s.turns))
	for k, v := range s.turns {
		out[k] = v
	}
	return out
}

// review runs one shadow review end to end: context assembly, model call,
// verdict parsing, log + metadata retention. Every failure path ends in a
// log entry and nothing else.
func (s *Superego) review(turnIndex int, snapshot history.Snapshot, draftParagraphs []string, tags []ParagraphAnnotation) {
	defer func() {
		// A panic here runs on a private goroutine: without this recover it
		// would kill the whole proxy, which is the one thing a facet must
		// never do.
		if r := recover(); r != nil {
			s.log.record(callLog{Facet: superegoFacet, Model: s.model, TurnIndex: &turnIndex, Err: fmt.Sprintf("panic: %v", r)})
		}
	}()

	if turnIndex < 0 || turnIndex >= len(snapshot.Turns) {
		s.log.record(callLog{Facet: superegoFacet, Model: s.model, TurnIndex: &turnIndex, Err: fmt.Sprintf("turn index %d outside snapshot of %d turns", turnIndex, len(snapshot.Turns))})
		return
	}
	if len(draftParagraphs) == 0 || len(draftParagraphs) != len(tags) {
		s.log.record(callLog{Facet: superegoFacet, Model: s.model, TurnIndex: &turnIndex, Err: fmt.Sprintf("draft/tags mismatch: %d paragraphs, %d tags", len(draftParagraphs), len(tags))})
		return
	}

	messages := superegoMessages(snapshot, turnIndex, draftUnderReviewMessage(draftParagraphs, tags), s.maxHistoryTurns)

	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()

	start := time.Now()
	eval, err := s.conversant.Converse(ctx, s.systemPrompt, messages)
	entry := callLog{
		Facet:           superegoFacet,
		Model:           s.model,
		TurnIndex:       &turnIndex,
		LatencyMs:       time.Since(start).Milliseconds(),
		HistoryMessages: len(messages),
		StopReason:      eval.StopReason,
		InputTokens:     eval.InputTokens,
		OutputTokens:    eval.OutputTokens,
	}
	if err != nil {
		entry.Err = err.Error()
		s.log.record(entry)
		return
	}

	critique := SuperegoCritique{TurnIndex: turnIndex, Critique: eval.Text}
	if verdict, ok := parseSuperegoVerdict(eval.Text); ok {
		critique.Verdict = verdict.Verdict
		critique.Annotations = verdict.Annotations
		critique.Why = verdict.Why
	}
	s.mu.Lock()
	s.turns[turnIndex] = critique
	s.mu.Unlock()

	entry.OK = true
	entry.Critique = critique.Critique
	entry.Verdict = critique.Verdict
	entry.SuperegoAnnotations = critique.Annotations
	entry.SuperegoWhy = critique.Why
	s.log.record(entry)
}

// superegoVerdict is the JSON object of the superego prompt contract.
type superegoVerdict struct {
	Verdict     string               `json:"verdict"`
	Annotations []SuperegoAnnotation `json:"annotations"`
	Why         string               `json:"why"`
}

var validVerdicts = map[string]bool{
	"approve":        true,
	"revise":         true,
	"hold_ask_human": true,
}

// parseSuperegoVerdict best-effort extracts the verdict object from the raw
// model text. ok=false means the text carries no valid verdict — never an
// error: in shadow mode a free-text critique is a fully acceptable outcome.
func parseSuperegoVerdict(raw string) (superegoVerdict, bool) {
	jsonObject := extractJSONObject(raw)
	if jsonObject == "" {
		return superegoVerdict{}, false
	}
	var v superegoVerdict
	if err := json.Unmarshal([]byte(jsonObject), &v); err != nil {
		return superegoVerdict{}, false
	}
	if !validVerdicts[v.Verdict] {
		return superegoVerdict{}, false
	}
	return v, true
}
