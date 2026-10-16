package facet

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/history"
)

// fakeConversant is the Conversant double: it records the call and returns a
// canned evaluation, honoring the context like the real client would.
type fakeConversant struct {
	response    Evaluation
	err         error
	delay       time.Duration
	calls       int
	gotSys      string
	gotMessages []anthropic.Message
}

func (f *fakeConversant) Converse(ctx context.Context, system string, messages []anthropic.Message) (Evaluation, error) {
	f.calls++
	f.gotSys = system
	f.gotMessages = messages
	if f.delay > 0 {
		select {
		case <-ctx.Done():
			return Evaluation{}, ctx.Err()
		case <-time.After(f.delay):
		}
	}
	return f.response, f.err
}

func newTestSuperego(conversant Conversant, logSink *bytes.Buffer) *Superego {
	params := SuperegoParams{
		Conversant:   conversant,
		SystemPrompt: "você é o superego",
		Timeout:      time.Second,
		Model:        "claude-sonnet-5",
		// Fixed clock so the dated system prompt is deterministic in assertions.
		Now: func() time.Time { return time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC) },
	}
	if logSink != nil {
		params.LogSink = logSink
	}
	return NewSuperego(params)
}

// reviewAndWait drives the real async entry point and waits for the review
// to land, so assertions see the final state.
func reviewAndWait(s *Superego, turnIndex int, snapshot history.Snapshot, paragraphs []string, tags []ParagraphAnnotation) {
	s.Review(turnIndex, snapshot, paragraphs, tags)
	s.WaitInFlight()
}

// TestSuperegoMessages_contextAssembly drives the history→messages
// conversion through a table: clean alternating messages, tag stripping,
// draft-user as the closing message, empty-side and truncation behavior.
// Each row states its own history inline.
func TestSuperegoMessages_contextAssembly(t *testing.T) {
	draft := "DRAFT"
	tests := []struct {
		desc            string
		snapshot        history.Snapshot
		turnIndex       int
		maxHistoryTurns int
		want            []anthropic.Message
	}{
		{
			desc: "two turns: prior exchange (annotation line stripped) plus reviewed turn's user text, draft closes",
			snapshot: history.Snapshot{Turns: []history.Turn{
				{
					UserText:      "contava com você ontem\n[emoções p1: investment=4 valence=negativa emotions=cobrança(3)]",
					AssistantText: "desculpa, falhei mesmo",
					Completed:     true,
				},
				{UserText: "e agora?", AssistantText: "vou consertar hoje", Completed: true},
			}},
			turnIndex: 1,
			want: []anthropic.Message{
				{Role: "user", Content: "contava com você ontem"},
				{Role: "assistant", Content: "desculpa, falhei mesmo"},
				{Role: "user", Content: "e agora?"},
				{Role: "user", Content: draft},
			},
		},
		{
			desc: "turn with empty user text contributes no user message",
			snapshot: history.Snapshot{Turns: []history.Turn{
				{AssistantText: "resposta sem prompt", Completed: true},
				{UserText: "ok", AssistantText: "reviewed", Completed: true},
			}},
			turnIndex: 1,
			want: []anthropic.Message{
				{Role: "assistant", Content: "resposta sem prompt"},
				{Role: "user", Content: "ok"},
				{Role: "user", Content: draft},
			},
		},
		{
			desc: "maxHistoryTurns keeps only the most recent turns, reviewed included",
			snapshot: history.Snapshot{Turns: []history.Turn{
				{UserText: "velho", AssistantText: "antigo", Completed: true},
				{UserText: "meio", AssistantText: "recente", Completed: true},
				{UserText: "atual", AssistantText: "reviewed", Completed: true},
			}},
			turnIndex:       2,
			maxHistoryTurns: 2,
			want: []anthropic.Message{
				{Role: "user", Content: "meio"},
				{Role: "assistant", Content: "recente"},
				{Role: "user", Content: "atual"},
				{Role: "user", Content: draft},
			},
		},
		{
			desc: "maxHistoryTurns larger than the history truncates nothing",
			snapshot: history.Snapshot{Turns: []history.Turn{
				{UserText: "oi", AssistantText: "reviewed", Completed: true},
			}},
			turnIndex:       0,
			maxHistoryTurns: 10,
			want: []anthropic.Message{
				{Role: "user", Content: "oi"},
				{Role: "user", Content: draft},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			got := superegoMessages(test.snapshot, test.turnIndex, draft, test.maxHistoryTurns)
			if len(got) != len(test.want) {
				t.Fatalf("got %d messages, want %d: %+v", len(got), len(test.want), got)
			}
			for i := range got {
				if got[i].Role != test.want[i].Role || got[i].Content != test.want[i].Content {
					t.Errorf("message %d = %+v, want %+v", i, got[i], test.want[i])
				}
			}
		})
	}
}

// TestSuperegoMessages_shadowPathIsTextOnly pins the retired #4749 behavior:
// the SHADOW path reconstructs from the snapshot's final text only, so the
// closing message is the bare draft and no tool-provenance block is emitted —
// the reviewed turn's tool traffic reaches the superego on the ACTIVE path via
// the real tool_use/tool_result blocks (TestActiveSuperego_* in the loop test),
// not a summary here.
func TestSuperegoMessages_shadowPathIsTextOnly(t *testing.T) {
	snapshot := history.Snapshot{Turns: []history.Turn{
		{
			UserText:      "checa o PR #4749",
			AssistantText: "reviewed",
			Completed:     true,
			ToolCalls: []history.ToolCall{
				{ID: "t1", Name: "Grep", Input: `{"pattern":"approved"}`, Result: "PR #4749 approved"},
			},
		},
	}}
	msgs := superegoMessages(snapshot, 0, "DRAFT", 0)
	final := msgs[len(msgs)-1]
	if final.Role != "user" || final.Content != "DRAFT" {
		t.Errorf("closing message = %+v, want the bare draft-user with no tool block", final)
	}
	for _, m := range msgs {
		if s, ok := m.Content.(string); ok && strings.Contains(s, "FERRAMENTAS") {
			t.Errorf("shadow path must emit no tool-provenance block:\n%s", s)
		}
	}
}

func TestDraftUnderReviewMessage_formatsTagsAndAbouts(t *testing.T) {
	draft := draftUnderReviewMessage(
		[]string{"vou consertar hoje", "sem falta"},
		[]ParagraphAnnotation{
			{Investment: 4, Valence: "negativa", Emotions: []Emotion{{Emotion: "culpa", Level: 4, About: "vou consertar hoje"}}},
			{Investment: 2, Valence: "neutra", Emotions: []Emotion{}},
		})

	if !strings.HasPrefix(draft, "<<<RASCUNHO_SOB_REVISAO_INICIO>>>") ||
		!strings.Contains(draft, "<<<RASCUNHO_SOB_REVISAO_FIM>>>") {
		t.Errorf("draft missing delimiters:\n%s", draft)
	}
	// The tags come interleaved in the same inline format the core's input
	// consumes; the about anchors ride the JSON block after the draft.
	if !strings.Contains(draft, "vou consertar hoje\n[emoções p1: investment=4 valence=negativa emotions=culpa(4)]") {
		t.Errorf("draft missing interleaved tag after paragraph 1:\n%s", draft)
	}
	if !strings.Contains(draft, "sem falta\n[emoções p2: investment=2 valence=neutra emotions=nenhuma]") {
		t.Errorf("draft missing interleaved tag after paragraph 2:\n%s", draft)
	}
	if !strings.Contains(draft, `[{"p":1,"emotions":[{"emotion":"culpa","level":4,"about":"vou consertar hoje"}]}]`) {
		t.Errorf("draft missing abouts JSON block:\n%s", draft)
	}
}

func TestDraftUnderReviewMessage_noEmotionsOmitsAboutsBlock(t *testing.T) {
	draft := draftUnderReviewMessage([]string{"texto técnico"}, []ParagraphAnnotation{{Investment: 0, Valence: "neutra"}})
	if strings.Contains(draft, "Âncoras") {
		t.Errorf("abouts block should be absent when no paragraph carries emotions:\n%s", draft)
	}
}

// TestSuperego_reviewSystemPromptAdvancesWithClock proves the date line is
// rendered per call, not captured at construction: a long-lived session whose
// clock crosses midnight must hand the superego the NEW day on the next review.
func TestSuperego_reviewSystemPromptAdvancesWithClock(t *testing.T) {
	conversant := &fakeConversant{response: evalText("ok")}
	days := []time.Time{
		time.Date(2026, 10, 5, 23, 0, 0, 0, time.UTC),
		time.Date(2026, 10, 6, 1, 0, 0, 0, time.UTC),
	}
	call := 0
	s := NewSuperego(SuperegoParams{
		Conversant: conversant, SystemPrompt: "você é o superego",
		Timeout: time.Second, Model: "claude-sonnet-5",
		Now: func() time.Time { d := days[min(call, len(days)-1)]; call++; return d },
	})
	snapshot := history.Snapshot{Turns: []history.Turn{{UserText: "oi", AssistantText: "resposta", Completed: true}}}
	tags := []ParagraphAnnotation{{Investment: 4, Valence: "negativa"}}

	reviewAndWait(s, 0, snapshot, []string{"resposta"}, tags)
	if !strings.HasSuffix(conversant.gotSys, "Data de hoje: 2026-10-05.") {
		t.Errorf("first review system prompt = %q, want day one", conversant.gotSys)
	}
	reviewAndWait(s, 0, snapshot, []string{"resposta"}, tags)
	if !strings.HasSuffix(conversant.gotSys, "Data de hoje: 2026-10-06.") {
		t.Errorf("second review system prompt = %q, want the advanced day", conversant.gotSys)
	}
}

// TestSuperego_reviewSystemPrompt_calibratedLeniency locks in change #2: the
// calibrated-leniency flag appends the leniency guidance to the ACTIVE review
// system prompt (and the extra last-round block only on the final ladder round),
// never to the shadow path, while the date line ALWAYS stays the final suffix
// and the flag-off prompt is exactly base+date (what the other tests and the
// temporal-claim logic depend on).
func TestSuperego_reviewSystemPrompt_calibratedLeniency(t *testing.T) {
	const base = "você é o superego"
	const dateSuffix = "\n\nData de hoje: 2026-10-05."
	tests := []struct {
		desc              string
		leniency          bool
		active            bool
		lastRound         bool
		wantLeniency      bool
		wantLastRound     bool
		wantExactBaseDate bool
	}{
		{desc: "flag off, active: base + date only", leniency: false, active: true, wantExactBaseDate: true},
		{desc: "flag off, active, lastRound ignored", leniency: false, active: true, lastRound: true, wantExactBaseDate: true},
		{desc: "flag on, shadow (not active): no leniency", leniency: true, active: false, wantExactBaseDate: true},
		{desc: "flag on, active, not last round: leniency only", leniency: true, active: true, wantLeniency: true},
		{desc: "flag on, active, last round: leniency plus last-round block", leniency: true, active: true, lastRound: true, wantLeniency: true, wantLastRound: true},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			s := newTestSuperego(&fakeConversant{response: evalText("ok")}, nil)
			s.calibratedLeniency = test.leniency

			got := s.reviewSystemPrompt(test.active, test.lastRound)

			if !strings.HasPrefix(got, base) {
				t.Errorf("prompt must start with the base system prompt:\n%s", got)
			}
			if !strings.HasSuffix(got, dateSuffix) {
				t.Errorf("date line must stay the final suffix:\n%s", got)
			}
			if test.wantExactBaseDate && got != base+dateSuffix {
				t.Errorf("want exactly base+date, got:\n%q", got)
			}
			if gotLeniency := strings.Contains(got, leniencyGuidance); gotLeniency != test.wantLeniency {
				t.Errorf("leniency guidance present = %v, want %v:\n%s", gotLeniency, test.wantLeniency, got)
			}
			if gotLast := strings.Contains(got, lastRoundLeniencyGuidance); gotLast != test.wantLastRound {
				t.Errorf("last-round guidance present = %v, want %v:\n%s", gotLast, test.wantLastRound, got)
			}
		})
	}
}

func TestSuperego_reviewHappyPathWithStructuredVerdict(t *testing.T) {
	conversant := &fakeConversant{response: evalText(`Analisei o rascunho.
{"verdict":"revise","annotations":[{"paragraph":1,"issue":"promessa sem caminho","why":"cria expectativa"}],"why":"promessa vaga"}`)}
	var log bytes.Buffer
	s := newTestSuperego(conversant, &log)

	snapshot := history.Snapshot{Turns: []history.Turn{
		{UserText: "contava com você ontem", AssistantText: "desculpa, falhei mesmo", Completed: true},
		{UserText: "e agora?", AssistantText: "vou consertar hoje", Completed: true},
	}}
	reviewAndWait(s, 1, snapshot, []string{"vou consertar hoje"},
		[]ParagraphAnnotation{{Investment: 4, Valence: "negativa", Emotions: []Emotion{{Emotion: "culpa", Level: 4, About: "vou consertar hoje"}}}})

	if conversant.gotSys != "você é o superego\n\nData de hoje: 2026-10-05." {
		t.Errorf("system prompt = %q", conversant.gotSys)
	}
	if len(conversant.gotMessages) != 4 {
		t.Fatalf("superego saw %d messages, want 4: %+v", len(conversant.gotMessages), conversant.gotMessages)
	}
	final := conversant.gotMessages[3]
	finalText, _ := final.Content.(string)
	if final.Role != "user" || !strings.Contains(finalText, "<<<RASCUNHO_SOB_REVISAO_INICIO>>>") {
		t.Errorf("final message is not the draft-user: %+v", final)
	}

	critique, ok := s.Critiques()[1]
	if !ok {
		t.Fatalf("no critique retained: %+v", s.Critiques())
	}
	if critique.Verdict != "revise" || critique.Why != "promessa vaga" {
		t.Errorf("critique = %+v", critique)
	}
	if len(critique.Annotations) != 1 || critique.Annotations[0].Issue != "promessa sem caminho" {
		t.Errorf("annotations = %+v", critique.Annotations)
	}
	if !strings.Contains(critique.Critique, "Analisei o rascunho.") {
		t.Errorf("full critique text not retained: %q", critique.Critique)
	}

	entry := lastLogEntry(t, &log)
	if entry["ok"] != true || entry["facet"] != "superego" {
		t.Errorf("log entry = %v", entry)
	}
	if entry["verdict"] != "revise" || entry["historyMessages"] != float64(4) {
		t.Errorf("log entry missing verdict/context fields: %v", entry)
	}
	if !strings.Contains(entry["critique"].(string), "Analisei o rascunho.") {
		t.Errorf("log entry missing critique text: %v", entry)
	}
}

func TestSuperego_freeTextCritiqueIsRetainedWithoutVerdict(t *testing.T) {
	conversant := &fakeConversant{response: evalText("crítica em prosa, sem JSON nenhum")}
	s := newTestSuperego(conversant, nil)

	reviewAndWait(s, 0,
		history.Snapshot{Turns: []history.Turn{{UserText: "oi", AssistantText: "prometo tudo", Completed: true}}},
		[]string{"prometo tudo"},
		[]ParagraphAnnotation{{Investment: 4, Valence: "mista"}})

	critique, ok := s.Critiques()[0]
	if !ok {
		t.Fatal("free-text critique must still be retained in shadow mode")
	}
	if critique.Verdict != "" || critique.Why != "" || len(critique.Annotations) != 0 {
		t.Errorf("unparseable response must leave structured fields empty: %+v", critique)
	}
	if critique.Critique != "crítica em prosa, sem JSON nenhum" {
		t.Errorf("critique text = %q", critique.Critique)
	}
}

// TestSuperego_failureModes drives every fail-open branch through a single
// table: each failure retains NO critique and ends in a log entry carrying
// the error, and nothing else happens. Each row states its own history and
// draft inline.
func TestSuperego_failureModes(t *testing.T) {
	tests := []struct {
		desc       string
		conversant *fakeConversant
		turnIndex  int
		snapshot   history.Snapshot
		paragraphs []string
		tags       []ParagraphAnnotation
		wantErr    string
		wantCalls  int
	}{
		{
			desc:       "conversant API error",
			conversant: &fakeConversant{err: errors.New("api exploded")},
			turnIndex:  0,
			snapshot:   history.Snapshot{Turns: []history.Turn{{UserText: "oi", AssistantText: "falo demais", Completed: true}}},
			paragraphs: []string{"falo demais"},
			tags:       []ParagraphAnnotation{{Investment: 4, Valence: "negativa"}},
			wantErr:    "api exploded",
			wantCalls:  1,
		},
		{
			desc:       "conversant slower than the facet timeout",
			conversant: &fakeConversant{response: evalText("tarde demais"), delay: 5 * time.Second},
			turnIndex:  0,
			snapshot:   history.Snapshot{Turns: []history.Turn{{UserText: "oi", AssistantText: "falo demais", Completed: true}}},
			paragraphs: []string{"falo demais"},
			tags:       []ParagraphAnnotation{{Investment: 4, Valence: "negativa"}},
			wantErr:    "context deadline exceeded",
			wantCalls:  1,
		},
		{
			desc:       "turn index outside the snapshot",
			conversant: &fakeConversant{response: evalText("nunca chamado")},
			turnIndex:  3,
			snapshot:   history.Snapshot{Turns: []history.Turn{{UserText: "oi", AssistantText: "só um turno", Completed: true}}},
			paragraphs: []string{"só um turno"},
			tags:       []ParagraphAnnotation{{Investment: 4, Valence: "negativa"}},
			wantErr:    "turn index 3 outside snapshot of 1 turns",
			wantCalls:  0,
		},
		{
			desc:       "paragraph/tag count mismatch breaks the interleave contract",
			conversant: &fakeConversant{response: evalText("nunca chamado")},
			turnIndex:  0,
			snapshot:   history.Snapshot{Turns: []history.Turn{{UserText: "oi", AssistantText: "dois\n\nparágrafos", Completed: true}}},
			paragraphs: []string{"dois", "parágrafos"},
			tags:       []ParagraphAnnotation{{Investment: 4, Valence: "negativa"}},
			wantErr:    "draft/tags mismatch: 2 paragraphs, 1 tags",
			wantCalls:  0,
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			var log bytes.Buffer
			s := newTestSuperego(test.conversant, &log)
			s.timeout = 50 * time.Millisecond

			reviewAndWait(s, test.turnIndex, test.snapshot, test.paragraphs, test.tags)

			if len(s.Critiques()) != 0 {
				t.Errorf("no critique should be retained on failure: %+v", s.Critiques())
			}
			if test.conversant.calls != test.wantCalls {
				t.Errorf("conversant called %d times, want %d", test.conversant.calls, test.wantCalls)
			}
			entry := lastLogEntry(t, &log)
			if entry["ok"] != false || !strings.Contains(entry["error"].(string), test.wantErr) {
				t.Errorf("log entry = %v, want error containing %q", entry, test.wantErr)
			}
		})
	}
}

// panicConversant triggers the goroutine's recover path: without it a facet
// panic would kill the whole proxy process.
type panicConversant struct{}

func (panicConversant) Converse(_ context.Context, _ string, _ []anthropic.Message) (Evaluation, error) {
	panic("boom")
}

func TestSuperego_panicIsRecoveredAndLogged(t *testing.T) {
	var log bytes.Buffer
	s := newTestSuperego(panicConversant{}, &log)

	reviewAndWait(s, 0,
		history.Snapshot{Turns: []history.Turn{{UserText: "oi", AssistantText: "bum", Completed: true}}},
		[]string{"bum"},
		[]ParagraphAnnotation{{Investment: 4, Valence: "negativa"}})

	entry := lastLogEntry(t, &log)
	if !strings.Contains(entry["error"].(string), "panic: boom") {
		t.Errorf("log entry = %v", entry)
	}
}

func TestSuperego_dispatchNeverBlocksOnSlowConversant(t *testing.T) {
	conversant := &fakeConversant{response: evalText("lento"), delay: 5 * time.Second}
	s := newTestSuperego(conversant, nil)
	s.timeout = 50 * time.Millisecond

	start := time.Now()
	s.Review(0,
		history.Snapshot{Turns: []history.Turn{{UserText: "oi", AssistantText: "devagar", Completed: true}}},
		[]string{"devagar"},
		[]ParagraphAnnotation{{Investment: 4, Valence: "negativa"}})
	if elapsed := time.Since(start); elapsed > 100*time.Millisecond {
		t.Fatalf("dispatch blocked for %v", elapsed)
	}
	s.WaitInFlight()
}
