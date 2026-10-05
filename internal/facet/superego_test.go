package facet

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
	"time"

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
	gotMessages []ReviewMessage
}

func (f *fakeConversant) Converse(ctx context.Context, system string, messages []ReviewMessage) (Evaluation, error) {
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
		want            []ReviewMessage
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
			want: []ReviewMessage{
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
			want: []ReviewMessage{
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
			want: []ReviewMessage{
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
			want: []ReviewMessage{
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
				if got[i] != test.want[i] {
					t.Errorf("message %d = %+v, want %+v", i, got[i], test.want[i])
				}
			}
		})
	}
}

// TestSuperegoMessages_appendsReviewedTurnToolSummary covers #4749: only the
// reviewed turn's tool calls are summarized into the closing draft-user message
// (name, bounded input/result snippets, error marker), and the summary rides
// AFTER the draft, not in place of it.
func TestSuperegoMessages_appendsReviewedTurnToolSummary(t *testing.T) {
	snapshot := history.Snapshot{Turns: []history.Turn{
		{
			UserText:      "checa o PR #4749",
			AssistantText: "reviewed",
			Completed:     true,
			ToolCalls: []history.ToolCall{
				{ID: "t1", Name: "Grep", Input: `{"pattern":"approved"}`, Result: "PR #4749 approved"},
				{ID: "t2", Name: "Bash", Input: `{"cmd":"gh pr view"}`, Result: "boom", IsError: true},
			},
		},
	}}
	msgs := superegoMessages(snapshot, 0, "DRAFT", 0)
	final := msgs[len(msgs)-1]
	if final.Role != "user" {
		t.Fatalf("closing message role = %q, want user", final.Role)
	}
	// The draft is preserved and precedes the tool block.
	if !strings.HasPrefix(final.Content, "DRAFT\n\n") {
		t.Errorf("tool summary must be appended after the draft, got:\n%s", final.Content)
	}
	for _, want := range []string{
		"<<<FERRAMENTAS_EXECUTADAS_NESTE_TURNO_INICIO>>>",
		"<<<FERRAMENTAS_EXECUTADAS_NESTE_TURNO_FIM>>>",
		"Grep",
		`{"pattern":"approved"}`,
		"PR #4749 approved",
		"Bash",
		"[ERRO]", // t2 failed
		"boom",
	} {
		if !strings.Contains(final.Content, want) {
			t.Errorf("tool summary missing %q:\n%s", want, final.Content)
		}
	}
}

// TestSuperegoMessages_noToolBlockWhenNoCalls pins the empty case: a reviewed
// turn that ran no tools appends nothing — the closing message is the bare
// draft, with no delimiter block.
func TestSuperegoMessages_noToolBlockWhenNoCalls(t *testing.T) {
	snapshot := history.Snapshot{Turns: []history.Turn{
		{UserText: "oi", AssistantText: "reviewed", Completed: true},
	}}
	msgs := superegoMessages(snapshot, 0, "DRAFT", 0)
	final := msgs[len(msgs)-1]
	if final.Content != "DRAFT" {
		t.Errorf("closing message = %q, want the bare draft with no tool block", final.Content)
	}
	if strings.Contains(final.Content, "FERRAMENTAS_EXECUTADAS") {
		t.Errorf("no tool block must be emitted when the reviewed turn ran no tools:\n%s", final.Content)
	}
}

// TestToolCallsSummary_truncatesAndSkipsEmpty covers the snippet bounds: an
// input/result longer than the cap is cut to superegoTool*Max runes with an
// ellipsis, and nil tool calls yield "" (no block).
func TestToolCallsSummary_truncatesAndSkipsEmpty(t *testing.T) {
	if s := toolCallsSummary(nil); s != "" {
		t.Errorf("toolCallsSummary(nil) = %q, want \"\"", s)
	}

	long := strings.Repeat("a", 500)
	summary := toolCallsSummary([]history.ToolCall{{Name: "Big", Input: long, Result: long}})
	if strings.Contains(summary, long) {
		t.Errorf("summary carried the full 500-rune payload, want it truncated:\n%s", summary)
	}
	// Input and result are capped at 200 runes then ellipsized.
	wantInput := strings.Repeat("a", superegoToolInputMax) + "…"
	wantResult := strings.Repeat("a", superegoToolResultMax) + "…"
	if !strings.Contains(summary, wantInput) {
		t.Errorf("input not truncated to %d runes + ellipsis:\n%s", superegoToolInputMax, summary)
	}
	if !strings.Contains(summary, wantResult) {
		t.Errorf("result not truncated to %d runes + ellipsis:\n%s", superegoToolResultMax, summary)
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

	if conversant.gotSys != "você é o superego" {
		t.Errorf("system prompt = %q", conversant.gotSys)
	}
	if len(conversant.gotMessages) != 4 {
		t.Fatalf("superego saw %d messages, want 4: %+v", len(conversant.gotMessages), conversant.gotMessages)
	}
	final := conversant.gotMessages[3]
	if final.Role != "user" || !strings.Contains(final.Content, "<<<RASCUNHO_SOB_REVISAO_INICIO>>>") {
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

func (panicConversant) Converse(_ context.Context, _ string, _ []ReviewMessage) (Evaluation, error) {
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
