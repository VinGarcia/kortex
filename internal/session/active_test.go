package session

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/history"
	"github.com/vingarcia/kortex/internal/protocol"
	"github.com/vingarcia/kortex/internal/tools"
)

// fakeGovernor stands in for the active superego loop: it records what it saw
// and returns a fixed delivered text. When callRedraftWith is non-nil it drives
// the real redraft callback once with that ephemeral tail and delivers the
// callback's result, so a test can prove the tail reaches the core model yet
// never persists.
type fakeGovernor struct {
	gotDraft        string
	gotTurnIndex    int
	gotSnapshot     history.Snapshot
	deliver         string
	held            bool
	callRedraftWith []anthropic.Message
	redraftResult   string
}

func (f *fakeGovernor) GovernOutput(
	ctx context.Context,
	turnIndex int,
	snapshot history.Snapshot,
	draft string,
	redraft func(ctx context.Context, tail []anthropic.Message) (string, error),
) (string, bool) {
	f.gotDraft = draft
	f.gotTurnIndex = turnIndex
	f.gotSnapshot = snapshot
	if f.callRedraftWith != nil {
		text, err := redraft(ctx, f.callRedraftWith)
		if err != nil {
			return draft, false
		}
		f.redraftResult = text
		return text, f.held
	}
	return f.deliver, f.held
}

// sequencedResponder replies to each Messages API call with the next text in
// replies (repeating the last), recording every call's message array so a test
// can inspect what the redraft call carried.
func sequencedResponder(t *testing.T, replies ...string) (*httptest.Server, *[][]anthropic.Message) {
	t.Helper()
	var calls [][]anthropic.Message
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []anthropic.Message `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		reply := replies[min(len(calls), len(replies)-1)]
		calls = append(calls, body.Messages)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"content":[{"type":"text","text":%q}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`, reply)
	}))
	t.Cleanup(server.Close)
	return server, &calls
}

// TestRun_ActiveGovernorDeliversGovernedText proves the active path emits and
// persists the GOVERNED text, and that the core's raw draft never reaches the
// wire or the canonical history.
func TestRun_ActiveGovernorDeliversGovernedText(t *testing.T) {
	server, _ := sequencedResponder(t, "raw-draft")
	client := anthropic.NewClient("secret", server.URL, time.Second)
	store := NewStore(t.TempDir())
	gov := &fakeGovernor{deliver: "governed-reply"}

	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin:  strings.NewReader(initializeLine("sys") + "\n" + userLine("sess-1", "hi") + "\n"),
		Stdout: &out, Client: client, Dispatcher: tools.NewDispatcher(),
		Store: store, SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		Governor: gov,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if gov.gotDraft != "raw-draft" || gov.gotTurnIndex != 0 {
		t.Errorf("governor saw draft=%q turnIndex=%d, want raw-draft/0", gov.gotDraft, gov.gotTurnIndex)
	}

	// The emitted stream must show the governed text in both the assistant event
	// and the result, and never leak the raw draft.
	if strings.Contains(out.String(), "raw-draft") {
		t.Errorf("the rejected draft leaked onto the wire:\n%s", out.String())
	}
	var assistantText, resultText string
	for _, ev := range parseLines(t, out.String()) {
		switch ev.Type {
		case protocol.TypeAssistant:
			if ev.Message != nil {
				assistantText = ev.Message.TextContent()
			}
		case protocol.TypeResult:
			resultText = ev.Result.Text
		}
	}
	if assistantText != "governed-reply" {
		t.Errorf("emitted assistant text = %q, want governed-reply", assistantText)
	}
	if resultText != "governed-reply" {
		t.Errorf("emitted result text = %q, want governed-reply", resultText)
	}

	// The persisted canonical history records only the delivered text as the
	// final assistant turn — never the raw draft.
	msgs, err := store.Load("sess-1")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(msgs) != 2 {
		t.Fatalf("persisted %d messages, want 2 (user + governed assistant): %+v", len(msgs), msgs)
	}
	if content, ok := msgs[1].Content.(string); !ok || content != "governed-reply" {
		t.Errorf("persisted final assistant = %+v, want the governed text", msgs[1].Content)
	}
}

// TestRun_ActiveGovernorEphemeralTailNotPersisted proves the redraft tail (the
// rejected drafts and critiques) reaches the CORE model during the loop but is
// never written to canonical history or the stream.
func TestRun_ActiveGovernorEphemeralTailNotPersisted(t *testing.T) {
	server, calls := sequencedResponder(t, "raw-draft", "revised-draft")
	client := anthropic.NewClient("secret", server.URL, time.Second)
	store := NewStore(t.TempDir())
	gov := &fakeGovernor{
		callRedraftWith: []anthropic.Message{{Role: "user", Content: "CRITIQUE-TAIL-TEXT"}},
	}

	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin:  strings.NewReader(initializeLine("sys") + "\n" + userLine("sess-1", "hi") + "\n"),
		Stdout: &out, Client: client, Dispatcher: tools.NewDispatcher(),
		Store: store, SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		RedraftTimeout: time.Second, Governor: gov,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// The core model made two calls: the initial draft and the redraft. The
	// redraft call must carry the ephemeral tail appended after the canonical
	// messages (so the core sees the critique), proving the loop reached it.
	if len(*calls) != 2 {
		t.Fatalf("core model calls = %d, want 2 (draft + redraft)", len(*calls))
	}
	redraftMsgs := (*calls)[1]
	last := redraftMsgs[len(redraftMsgs)-1]
	if content, ok := last.Content.(string); !ok || content != "CRITIQUE-TAIL-TEXT" {
		t.Errorf("redraft call did not carry the ephemeral tail as its last message: %+v", redraftMsgs)
	}
	if gov.redraftResult != "revised-draft" {
		t.Errorf("redraft returned %q, want revised-draft", gov.redraftResult)
	}

	// Neither the raw draft, the critique tail, nor anything but the delivered
	// revised text may appear on the wire.
	if strings.Contains(out.String(), "raw-draft") || strings.Contains(out.String(), "CRITIQUE-TAIL-TEXT") {
		t.Errorf("ephemeral tail or rejected draft leaked onto the wire:\n%s", out.String())
	}

	// The persisted history carries only user + the delivered revised draft;
	// the tail's critique message is NOT persisted.
	msgs, err := store.Load("sess-1")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(msgs) != 2 {
		t.Fatalf("persisted %d messages, want 2: %+v", len(msgs), msgs)
	}
	if content, ok := msgs[1].Content.(string); !ok || content != "revised-draft" {
		t.Errorf("persisted final assistant = %+v, want the revised draft", msgs[1].Content)
	}
	for _, m := range msgs {
		if content, ok := m.Content.(string); ok && strings.Contains(content, "CRITIQUE-TAIL-TEXT") {
			t.Errorf("ephemeral critique leaked into canonical history: %+v", m)
		}
	}
}

// TestRun_ActiveGovernorRedraftTimesOutFailsOpen proves a hung redraft call is
// bounded by Config.RedraftTimeout rather than blocking the turn forever: when
// the redraft endpoint never responds, the redraft returns (with an error)
// within roughly the deadline, the governor fails open, and the turn still
// completes by delivering the raw draft. The client is built with no HTTP
// timeout (the production wiring from main.go), so the only bound on the hung
// call is the per-call deadline this test exercises.
func TestRun_ActiveGovernorRedraftTimesOutFailsOpen(t *testing.T) {
	const redraftTimeout = 100 * time.Millisecond

	// The first call (the draft) answers normally; the redraft call (the second)
	// hangs, so the only thing that ends the redraft is RedraftTimeout firing on
	// the caller side. teardown is closed before server.Close runs (defer beats
	// t.Cleanup) so the hung handler never blocks the server shutdown even if it
	// didn't observe the client-side cancellation.
	teardown := make(chan struct{})
	defer close(teardown)
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"content":[{"type":"text","text":"raw-draft"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
			return
		}
		select {
		case <-r.Context().Done():
		case <-teardown:
		}
	}))
	t.Cleanup(server.Close)

	client := anthropic.NewClient("secret", server.URL, 0)
	store := NewStore(t.TempDir())
	gov := &fakeGovernor{
		callRedraftWith: []anthropic.Message{{Role: "user", Content: "CRITIQUE-TAIL-TEXT"}},
	}

	var out strings.Builder
	start := time.Now()
	err := Run(context.Background(), Config{
		Stdin:  strings.NewReader(initializeLine("sys") + "\n" + userLine("sess-1", "hi") + "\n"),
		Stdout: &out, Client: client, Dispatcher: tools.NewDispatcher(),
		Store: store, SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		RedraftTimeout: redraftTimeout, Governor: gov,
	})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Bounded: the turn finished far sooner than the hung endpoint would ever
	// allow. The ceiling is generous versus the 100ms deadline yet nowhere near
	// the forever a missing bound would produce.
	if elapsed > 5*time.Second {
		t.Fatalf("turn took %s, want bounded completion near the %s redraft deadline", elapsed, redraftTimeout)
	}
	if n := calls.Load(); n != 2 {
		t.Fatalf("core model calls = %d, want 2 (draft + hung redraft)", n)
	}
	// Fail-open: the redraft errored (so the governor never recorded a redraft
	// result) and the raw draft is what reached the wire.
	if gov.redraftResult != "" {
		t.Errorf("redraft result = %q, want empty (the hung redraft must have errored)", gov.redraftResult)
	}
	var resultText string
	for _, ev := range parseLines(t, out.String()) {
		if ev.Type == protocol.TypeResult {
			resultText = ev.Result.Text
		}
	}
	if resultText != "raw-draft" {
		t.Errorf("delivered result = %q, want raw-draft (fail-open to the draft)", resultText)
	}
}

// TestRun_ActiveGovernorZeroRedraftTimeoutUsesDefault proves that leaving
// Config.RedraftTimeout at its zero value does NOT silently disable the
// superego. Before the clamp, context.WithTimeout(ctx, 0) handed the redraft an
// already-expired deadline, so every redraft errored instantly and the governor
// failed open to the raw draft on every turn. With the DefaultRedraftTimeout
// floor the redraft gets a real (non-expired) deadline: it reaches the core
// model, succeeds, and the revised draft is delivered.
func TestRun_ActiveGovernorZeroRedraftTimeoutUsesDefault(t *testing.T) {
	server, calls := sequencedResponder(t, "raw-draft", "revised-draft")
	client := anthropic.NewClient("secret", server.URL, time.Second)
	store := NewStore(t.TempDir())
	gov := &fakeGovernor{
		callRedraftWith: []anthropic.Message{{Role: "user", Content: "CRITIQUE-TAIL-TEXT"}},
	}

	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin:  strings.NewReader(initializeLine("sys") + "\n" + userLine("sess-1", "hi") + "\n"),
		Stdout: &out, Client: client, Dispatcher: tools.NewDispatcher(),
		Store: store, SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		// RedraftTimeout deliberately omitted (zero): the clamp must supply the
		// default floor rather than an already-expired deadline.
		Governor: gov,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Both core calls happened (draft + redraft): the redraft was not instantly
	// cancelled by a zero deadline.
	if len(*calls) != 2 {
		t.Fatalf("core model calls = %d, want 2 (draft + redraft); a zero deadline would have aborted the redraft", len(*calls))
	}
	// The redraft resolved against the core model rather than erroring out, so
	// the governor delivered the revised draft instead of failing open.
	if gov.redraftResult != "revised-draft" {
		t.Errorf("redraft result = %q, want revised-draft (a zero deadline would error and leave it empty)", gov.redraftResult)
	}
	var resultText string
	for _, ev := range parseLines(t, out.String()) {
		if ev.Type == protocol.TypeResult {
			resultText = ev.Result.Text
		}
	}
	if resultText != "revised-draft" {
		t.Errorf("delivered result = %q, want revised-draft (superego executed, not a silent fail-open to raw-draft)", resultText)
	}
}

// TestRun_ActiveGovernorEmptyRedraftFailsOpen proves the governor-path backstop
// for the live 2026-10-04 failure class: a redraft that comes back empty (the
// thinking model spent its whole budget on thinking) must NOT be delivered as an
// empty turn. The redraft closure reports it as a failure, the governor fails
// open to the non-empty first draft, and that draft is what reaches the wire —
// never an empty assistant turn OpenClaw would reject.
func TestRun_ActiveGovernorEmptyRedraftFailsOpen(t *testing.T) {
	// First call (draft) is non-empty; the redraft (second call) returns empty.
	server, _ := sequencedResponder(t, "raw-draft", "")
	client := anthropic.NewClient("secret", server.URL, time.Second)
	store := NewStore(t.TempDir())
	gov := &fakeGovernor{
		callRedraftWith: []anthropic.Message{{Role: "user", Content: "CRITIQUE-TAIL-TEXT"}},
	}

	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin:  strings.NewReader(initializeLine("sys") + "\n" + userLine("sess-1", "hi") + "\n"),
		Stdout: &out, Client: client, Dispatcher: tools.NewDispatcher(),
		Store: store, SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		RedraftTimeout: time.Second, Governor: gov,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// The empty redraft must have been reported as a failure, so the governor
	// never recorded a redraft result and fell back to the draft.
	if gov.redraftResult != "" {
		t.Errorf("redraft result = %q, want empty (an empty redraft must error, not deliver)", gov.redraftResult)
	}
	var assistantText, resultText string
	sawError := false
	for _, ev := range parseLines(t, out.String()) {
		switch ev.Type {
		case protocol.TypeAssistant:
			if ev.Message != nil {
				assistantText = ev.Message.TextContent()
			}
		case protocol.TypeResult:
			resultText = ev.Result.Text
			if ev.Result.IsError {
				sawError = true
			}
		}
	}
	if sawError {
		t.Errorf("turn surfaced an error result; fail-open to the non-empty draft should have delivered cleanly:\n%s", out.String())
	}
	if assistantText != "raw-draft" || resultText != "raw-draft" {
		t.Errorf("delivered assistant=%q result=%q, want raw-draft (fail-open to the non-empty first draft)", assistantText, resultText)
	}
}

// TestRun_ActiveGovernorHoldDeliversHoldMessage proves a held turn delivers the
// governor's hold-and-ask text instead of the draft, and persists that text.
func TestRun_ActiveGovernorHoldDeliversHoldMessage(t *testing.T) {
	server, _ := sequencedResponder(t, "raw-draft")
	client := anthropic.NewClient("secret", server.URL, time.Second)
	store := NewStore(t.TempDir())
	gov := &fakeGovernor{deliver: "[kortex] held — como seguir?", held: true}

	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin:  strings.NewReader(initializeLine("sys") + "\n" + userLine("sess-1", "hi") + "\n"),
		Stdout: &out, Client: client, Dispatcher: tools.NewDispatcher(),
		Store: store, SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		Governor: gov,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	var resultText string
	for _, ev := range parseLines(t, out.String()) {
		if ev.Type == protocol.TypeResult {
			resultText = ev.Result.Text
		}
	}
	if resultText != "[kortex] held — como seguir?" {
		t.Errorf("held turn delivered %q, want the hold-and-ask message", resultText)
	}
	if strings.Contains(out.String(), "raw-draft") {
		t.Errorf("held turn leaked the draft onto the wire:\n%s", out.String())
	}
}
