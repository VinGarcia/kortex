package session

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
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
	gotDraft         string
	gotTurnIndex     int
	gotSnapshot      history.Snapshot
	deliver          string
	held             bool
	callRedraftWith  []anthropic.Message
	redraftResult    string
	redraftToolCalls []history.ToolCall
}

func (f *fakeGovernor) GovernOutput(
	ctx context.Context,
	turnIndex int,
	snapshot history.Snapshot,
	draft string,
	redraft func(ctx context.Context, tail []anthropic.Message) (string, []history.ToolCall, error),
) (string, bool) {
	f.gotDraft = draft
	f.gotTurnIndex = turnIndex
	f.gotSnapshot = snapshot
	if f.callRedraftWith != nil {
		text, calls, err := redraft(ctx, f.callRedraftWith)
		if err != nil {
			return draft, false
		}
		f.redraftResult = text
		f.redraftToolCalls = calls
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

// recordingGovernor records what the governor saw on EVERY turn (not just the
// last), so a multi-turn test can assert the superego's per-turn input is
// unchanged by an unrelated feature.
type recordingGovernor struct {
	turnIndices  []int
	snapshotLens []int
	deliver      string
}

func (g *recordingGovernor) GovernOutput(
	ctx context.Context,
	turnIndex int,
	snapshot history.Snapshot,
	draft string,
	redraft func(ctx context.Context, tail []anthropic.Message) (string, []history.ToolCall, error),
) (string, bool) {
	g.turnIndices = append(g.turnIndices, turnIndex)
	g.snapshotLens = append(g.snapshotLens, len(snapshot.Turns))
	return g.deliver, false
}

// TestRun_ToneDigestInvisibleToGovernor locks in FIX 1: wiring the tone digest
// must NOT change what the active superego/governor receives. The governor and
// the TurnEvaluator are mutually exclusive, so on the active path `turns` is
// never accumulated and the governor must keep seeing turnIndex 0 and a
// single-turn snapshot on EVERY turn — the digest draws from its own
// priorUserTexts accumulator instead. A two-turn session run twice (with the
// digest wired, and as a no-digest baseline) must produce identical governor
// input, proving the feature is invisible to the superego (same turnIndex, same
// snapshot size, so same per-turn token cost).
func TestRun_ToneDigestInvisibleToGovernor(t *testing.T) {
	run := func(digester ToneDigester) *recordingGovernor {
		server, _ := sequencedResponder(t, "raw-draft")
		client := anthropic.NewClient("secret", server.URL, time.Second)
		gov := &recordingGovernor{deliver: "governed"}
		in := strings.Join([]string{
			initializeLine("sys"),
			userLine("sess-1", "estou exausto"),
			userLine("sess-1", "e agora?"),
		}, "\n") + "\n"
		var out strings.Builder
		err := Run(context.Background(), Config{
			Stdin: strings.NewReader(in), Stdout: &out, Client: client,
			Dispatcher: tools.NewDispatcher(), Store: NewStore(t.TempDir()), SessionID: "sess-1",
			Model: "m", MaxTokens: 1024, NewUUID: seqUUID(), Governor: gov, ToneDigester: digester,
		})
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		return gov
	}

	withDigest := run(&fakeToneDigester{digest: "<<<TOM_DA_CONVERSA_INICIO>>>\n- x\n<<<TOM_DA_CONVERSA_FIM>>>"})
	baseline := run(nil)

	// The governor path never accumulates turns, so it sees turnIndex 0 and a
	// single-turn snapshot on both turns — with or without the digest.
	wantTurnIndices := []int{0, 0}
	wantSnapshotLens := []int{1, 1}
	for _, c := range []struct {
		name string
		gov  *recordingGovernor
	}{{"with-digest", withDigest}, {"baseline", baseline}} {
		if !slices.Equal(c.gov.turnIndices, wantTurnIndices) {
			t.Errorf("%s: governor turnIndices = %v, want %v", c.name, c.gov.turnIndices, wantTurnIndices)
		}
		if !slices.Equal(c.gov.snapshotLens, wantSnapshotLens) {
			t.Errorf("%s: governor snapshot sizes = %v, want %v", c.name, c.gov.snapshotLens, wantSnapshotLens)
		}
	}
	// And the two runs are byte-for-byte identical in what the governor saw: the
	// digest is invisible to the superego.
	if !slices.Equal(withDigest.turnIndices, baseline.turnIndices) ||
		!slices.Equal(withDigest.snapshotLens, baseline.snapshotLens) {
		t.Errorf("digest changed governor input: with-digest (idx=%v lens=%v) vs baseline (idx=%v lens=%v)",
			withDigest.turnIndices, withDigest.snapshotLens, baseline.turnIndices, baseline.snapshotLens)
	}
}

// scriptedCoreServer replies to each Messages API call with the next raw JSON
// body in bodies (repeating the last), and records every call's raw request
// body so a test can assert what the redraft tool loop sent (the critique, the
// guard-rail, and the tool_results fed back). Unlike sequencedResponder it lets
// a test script tool_use bodies, which the redraft now drives.
func scriptedCoreServer(t *testing.T, bodies ...string) (*httptest.Server, *[]string) {
	t.Helper()
	var reqs []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		i := min(len(reqs), len(bodies)-1)
		reqs = append(reqs, string(raw))
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, bodies[i])
	}))
	t.Cleanup(server.Close)
	return server, &reqs
}

const (
	endTurnDraftBody = `{"content":[{"type":"text","text":"raw-draft"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
)

func toolUseBody(command string) string {
	return fmt.Sprintf(`{"content":[{"type":"tool_use","id":"toolu_1","name":"bash","input":{"command":%q}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`, command)
}

func endTurnTextBody(text string) string {
	return fmt.Sprintf(`{"content":[{"type":"text","text":%q}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`, text)
}

// TestRun_ActiveGovernorRedraftActsWithTools proves directive #4753: a redraft
// whose finding needs ACTION drives the real bounded tool loop — it runs the
// check (a bash tool_use the dispatcher executes) and answers with the evidence,
// instead of merely rewording. The redraft's opening message carries the
// superego critique plus the deterministic bounded-action guard-rail, and the
// tool_result is fed back into the loop before the evidence-backed draft is
// delivered and persisted.
func TestRun_ActiveGovernorRedraftActsWithTools(t *testing.T) {
	server, reqs := scriptedCoreServer(t,
		endTurnDraftBody,                         // call 1: the initial draft
		toolUseBody("echo evidence-123"),         // call 2: redraft acts (runs the check)
		endTurnTextBody("verified evidence-123"), // call 3: redraft answers with evidence
	)
	client := anthropic.NewClient("secret", server.URL, time.Second)
	store := NewStore(t.TempDir())
	gov := &fakeGovernor{
		callRedraftWith: []anthropic.Message{{Role: "user", Content: "CRITIQUE-X"}},
	}
	eval := &fakeTurnEvaluator{}

	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin:  strings.NewReader(initializeLine("sys") + "\n" + userLine("sess-1", "hi") + "\n"),
		Stdout: &out, Client: client, Dispatcher: tools.NewDispatcher(),
		Store: store, SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		RedraftTimeout: time.Second, Governor: gov, TurnEvaluator: eval,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Draft + two redraft tool-loop calls (tool_use, then the evidence answer).
	if len(*reqs) != 3 {
		t.Fatalf("core model calls = %d, want 3 (draft + redraft tool round + redraft answer)", len(*reqs))
	}
	// The redraft's opening user message carried the critique and the guard-rail.
	if !strings.Contains((*reqs)[1], "CRITIQUE-X") || !strings.Contains((*reqs)[1], "sem retrabalho gratuito") {
		t.Errorf("redraft opening call missing critique or guard-rail:\n%s", (*reqs)[1])
	}
	// The second redraft-loop call carried the real bash tool_result back — proof
	// the loop actually executed the check rather than rewording around it.
	if !strings.Contains((*reqs)[2], "evidence-123") {
		t.Errorf("redraft loop did not feed the tool evidence back:\n%s", (*reqs)[2])
	}
	if gov.redraftResult != "verified evidence-123" {
		t.Errorf("redraft result = %q, want the evidence-backed answer", gov.redraftResult)
	}
	// The redraft also hands its own tool calls back to the governor so the next
	// review round's provenance block covers the evidence it just gathered.
	if len(gov.redraftToolCalls) != 1 || gov.redraftToolCalls[0].Name != "bash" ||
		!strings.Contains(gov.redraftToolCalls[0].Result, "evidence-123") {
		t.Errorf("redraft tool calls = %+v, want the bash evidence call with its result", gov.redraftToolCalls)
	}

	var resultText string
	for _, ev := range parseLines(t, out.String()) {
		if ev.Type == protocol.TypeResult {
			resultText = ev.Result.Text
		}
	}
	if resultText != "verified evidence-123" {
		t.Errorf("delivered result = %q, want the evidence-backed redraft", resultText)
	}
	msgs, err := store.Load("sess-1")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if content, ok := msgs[len(msgs)-1].Content.(string); !ok || content != "verified evidence-123" {
		t.Errorf("persisted final assistant = %+v, want the evidence-backed redraft", msgs[len(msgs)-1].Content)
	}
	// The canonical Turn records the redraft's tool call too: the delivered text
	// rests on that evidence, so history must list it, not just a diag line.
	if len(eval.snapshots) != 1 {
		t.Fatalf("turn evaluator fired %d times, want 1", len(eval.snapshots))
	}
	turn := eval.snapshots[0].Turns[0]
	if len(turn.ToolCalls) != 1 || turn.ToolCalls[0].Name != "bash" ||
		!strings.Contains(turn.ToolCalls[0].Result, "evidence-123") {
		t.Errorf("canonical turn tool calls = %+v, want the redraft's bash evidence call", turn.ToolCalls)
	}
}

// slowGovernor blocks for delay before delivering, simulating a long superego
// ladder, so the keepalive test can observe the ticker firing mid-ladder.
type slowGovernor struct {
	delay   time.Duration
	deliver string
}

func (g *slowGovernor) GovernOutput(
	ctx context.Context,
	turnIndex int,
	snapshot history.Snapshot,
	draft string,
	redraft func(ctx context.Context, tail []anthropic.Message) (string, []history.ToolCall, error),
) (string, bool) {
	time.Sleep(g.delay)
	return g.deliver, false
}

// TestRun_GovernorKeepaliveDuringLadder locks in the no-output-watchdog fix: a
// long-running governor ladder must put benign system/keepalive lines on the
// wire while it deliberates, and the stream must still end with the normal
// final assistant + result events after the last keepalive (the ticker stops
// before the final emission, so the two can never interleave).
func TestRun_GovernorKeepaliveDuringLadder(t *testing.T) {
	server, _ := scriptedCoreServer(t, endTurnDraftBody)
	client := anthropic.NewClient("secret", server.URL, time.Second)
	gov := &slowGovernor{delay: 120 * time.Millisecond, deliver: "governed"}

	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin:  strings.NewReader(initializeLine("sys") + "\n" + userLine("sess-1", "hi") + "\n"),
		Stdout: &out, Client: client, Dispatcher: tools.NewDispatcher(),
		Store: NewStore(t.TempDir()), SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		RedraftTimeout: time.Second, Governor: gov,
		GovernKeepaliveInterval: 20 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	keepalives, lastKeepalive, resultIndex := 0, -1, -1
	for i, line := range lines {
		if strings.Contains(line, `"subtype":"keepalive"`) {
			keepalives++
			lastKeepalive = i
			if !strings.Contains(line, `"type":"system"`) || !strings.Contains(line, `"session_id":"sess-1"`) {
				t.Errorf("malformed keepalive line: %s", line)
			}
		}
		if strings.Contains(line, `"type":"result"`) {
			resultIndex = i
		}
	}
	if keepalives < 2 {
		t.Errorf("saw %d keepalive lines during a 120ms ladder at 20ms interval, want >= 2:\n%s", keepalives, out.String())
	}
	if resultIndex == -1 || lastKeepalive > resultIndex {
		t.Errorf("keepalive after the terminal result (last keepalive line %d, result line %d)", lastKeepalive, resultIndex)
	}
	// The governed text still arrives intact as the terminal result.
	var resultText string
	for _, ev := range parseLines(t, out.String()) {
		if ev.Type == protocol.TypeResult {
			resultText = ev.Result.Text
		}
	}
	if resultText != "governed" {
		t.Errorf("delivered result = %q, want the governed text", resultText)
	}
}

// TestRun_GovernorSnapshotCarriesTurnToolCalls locks in the provenance fix for
// the active path (#4749): the reviewed turn the governor receives must carry
// the tool calls the core actually ran this turn, result attached — that is
// what feeds the superego's FERRAMENTAS_EXECUTADAS block. Before the fix the
// reviewed turn was built without ToolCalls, so the superego saw no provenance
// and rejected genuinely tool-backed drafts as fabricated.
func TestRun_GovernorSnapshotCarriesTurnToolCalls(t *testing.T) {
	server, _ := scriptedCoreServer(t,
		toolUseBody("echo prov-42"),        // call 1: the core runs a real check
		endTurnTextBody("checked prov-42"), // call 2: the evidence-backed draft
	)
	client := anthropic.NewClient("secret", server.URL, time.Second)
	gov := &fakeGovernor{deliver: "ok"}

	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin:  strings.NewReader(initializeLine("sys") + "\n" + userLine("sess-1", "hi") + "\n"),
		Stdout: &out, Client: client, Dispatcher: tools.NewDispatcher(),
		Store: NewStore(t.TempDir()), SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		RedraftTimeout: time.Second, Governor: gov,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	reviewed := gov.gotSnapshot.Turns[gov.gotTurnIndex]
	if len(reviewed.ToolCalls) != 1 {
		t.Fatalf("reviewed turn carries %d tool calls, want 1: %+v", len(reviewed.ToolCalls), reviewed.ToolCalls)
	}
	call := reviewed.ToolCalls[0]
	if call.Name != "bash" || !strings.Contains(call.Input, "echo prov-42") {
		t.Errorf("reviewed tool call = %+v, want the bash echo", call)
	}
	if !strings.Contains(call.Result, "prov-42") || call.IsError {
		t.Errorf("reviewed tool call result = %q (isError %v), want the real output attached", call.Result, call.IsError)
	}
}

// TestRun_ActiveGovernorRedraftToolBudgetEnforced proves the per-redraft tool
// budget caps the loop: with RedraftToolBudget 2 the redraft runs exactly two
// tool rounds and then finalizes with a text answer (FinalizeOnBudget), so a
// core that keeps asking for tools can never run an unbounded loop and the turn
// still completes with a deliverable draft rather than hanging or erroring.
func TestRun_ActiveGovernorRedraftToolBudgetEnforced(t *testing.T) {
	server, reqs := scriptedCoreServer(t,
		endTurnDraftBody,                    // call 1: the initial draft
		toolUseBody("echo loop"),            // call 2: redraft tool round 1
		toolUseBody("echo loop"),            // call 3: redraft tool round 2 (budget = 2)
		endTurnTextBody("budget-finalized"), // call 4: forced toolless finalize
	)
	client := anthropic.NewClient("secret", server.URL, time.Second)
	store := NewStore(t.TempDir())
	gov := &fakeGovernor{
		callRedraftWith: []anthropic.Message{{Role: "user", Content: "CRITIQUE-X"}},
	}

	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin:  strings.NewReader(initializeLine("sys") + "\n" + userLine("sess-1", "hi") + "\n"),
		Stdout: &out, Client: client, Dispatcher: tools.NewDispatcher(),
		Store: store, SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		RedraftTimeout: 5 * time.Second, RedraftToolBudget: 2, Governor: gov,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Draft + 2 budgeted tool rounds + 1 finalize = 4 calls. If the budget were
	// not enforced the server would keep handing back tool_use forever.
	if len(*reqs) != 4 {
		t.Fatalf("core model calls = %d, want 4 (draft + 2 tool rounds + finalize)", len(*reqs))
	}
	if gov.redraftResult != "budget-finalized" {
		t.Errorf("redraft result = %q, want the finalized answer", gov.redraftResult)
	}
	var resultText string
	sawError := false
	for _, ev := range parseLines(t, out.String()) {
		if ev.Type == protocol.TypeResult {
			resultText = ev.Result.Text
			sawError = ev.Result.IsError
		}
	}
	if sawError {
		t.Errorf("budget exhaustion surfaced an error result; it must end gracefully:\n%s", out.String())
	}
	if resultText != "budget-finalized" {
		t.Errorf("delivered result = %q, want budget-finalized (graceful finalize)", resultText)
	}
}

// TestRun_ActiveGovernorZeroRedraftToolBudgetUsesDefault proves that leaving
// Config.RedraftToolBudget at its zero value runs EXACTLY the default
// (DefaultRedraftToolBudget = 3) tool-use rounds, not RunLoop's far larger
// DefaultMaxTurns (25). Before the clamp a 0 budget was handed straight to
// RunLoop as MaxTurns 0, which silently resolved to 25 — an 8x-larger safety
// budget that only main's resolution masked. The clamp at the consumption point
// (mirroring the RedraftTimeout floor) makes a 0-budget Config honor 3 rounds
// regardless of how it was built. The server keeps offering tools on every
// redraft round, so an unclamped budget would run all 25; it finalizes with
// text only on the toolless FinalizeOnBudget call.
func TestRun_ActiveGovernorZeroRedraftToolBudgetUsesDefault(t *testing.T) {
	var reqN, toolRounds atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		n := reqN.Add(1)
		w.Header().Set("Content-Type", "application/json")
		switch {
		case n == 1:
			// The initial draft turn: stop immediately so only the redraft drives
			// the tool loop that this test counts.
			fmt.Fprint(w, endTurnDraftBody)
		case bytes.Contains(raw, []byte(`"tools":[`)):
			// A redraft tool-loop round (tools offered): keep asking for a tool so
			// an unclamped 25-round budget would run every round, not stop early.
			toolRounds.Add(1)
			fmt.Fprint(w, toolUseBody("echo loop"))
		default:
			// The toolless FinalizeOnBudget call: the budget is spent, so close the
			// round with a deliverable text answer.
			fmt.Fprint(w, endTurnTextBody("budget-finalized"))
		}
	}))
	t.Cleanup(server.Close)

	client := anthropic.NewClient("secret", server.URL, time.Second)
	store := NewStore(t.TempDir())
	gov := &fakeGovernor{
		callRedraftWith: []anthropic.Message{{Role: "user", Content: "CRITIQUE-X"}},
	}

	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin:  strings.NewReader(initializeLine("sys") + "\n" + userLine("sess-1", "hi") + "\n"),
		Stdout: &out, Client: client, Dispatcher: tools.NewDispatcher(),
		Store: store, SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		// RedraftTimeout is a real value so only the budget is under test;
		// RedraftToolBudget deliberately omitted (zero): the clamp must supply the
		// DefaultRedraftToolBudget floor rather than RunLoop's DefaultMaxTurns.
		RedraftTimeout: 5 * time.Second, Governor: gov,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// The redraft ran exactly DefaultRedraftToolBudget rounds. Without the clamp
	// this is RunLoop's DefaultMaxTurns (25) — the silent 8x drift under test.
	if got := int(toolRounds.Load()); got != DefaultRedraftToolBudget {
		t.Fatalf("redraft tool rounds = %d, want %d (a 0 budget must clamp to the default, not RunLoop's 25)", got, DefaultRedraftToolBudget)
	}
	if gov.redraftResult != "budget-finalized" {
		t.Errorf("redraft result = %q, want the finalized answer", gov.redraftResult)
	}
	var resultText string
	sawError := false
	for _, ev := range parseLines(t, out.String()) {
		if ev.Type == protocol.TypeResult {
			resultText = ev.Result.Text
			sawError = ev.Result.IsError
		}
	}
	if sawError {
		t.Errorf("budget exhaustion surfaced an error result; it must end gracefully:\n%s", out.String())
	}
	if resultText != "budget-finalized" {
		t.Errorf("delivered result = %q, want budget-finalized (graceful finalize)", resultText)
	}
}

// TestRun_ActiveGovernorRedraftToolErrorFallsBackToText proves the fail-safe: if
// the redraft tool loop errors for a non-deadline reason (here the first loop
// call 500s), the redraft degrades to the pre-#4753 text-only revision rather
// than breaking the turn, and that reworded draft is delivered cleanly.
func TestRun_ActiveGovernorRedraftToolErrorFallsBackToText(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch calls.Add(1) {
		case 1:
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, endTurnDraftBody)
		case 2:
			// The redraft tool loop's first call fails (not a deadline) -> RunLoop
			// errors -> the closure falls back to the text-only redraft.
			http.Error(w, "boom", http.StatusInternalServerError)
		default:
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, endTurnTextBody("fallback-reworded"))
		}
	}))
	t.Cleanup(server.Close)

	client := anthropic.NewClient("secret", server.URL, time.Second)
	store := NewStore(t.TempDir())
	gov := &fakeGovernor{
		callRedraftWith: []anthropic.Message{{Role: "user", Content: "CRITIQUE-X"}},
	}

	var out strings.Builder
	err := Run(context.Background(), Config{
		Stdin:  strings.NewReader(initializeLine("sys") + "\n" + userLine("sess-1", "hi") + "\n"),
		Stdout: &out, Client: client, Dispatcher: tools.NewDispatcher(),
		Store: store, SessionID: "sess-1", Model: "m", MaxTokens: 1024, NewUUID: seqUUID(),
		RedraftTimeout: 5 * time.Second, Governor: gov,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Draft, failed tool-loop call, text-only fallback = 3 calls.
	if n := calls.Load(); n != 3 {
		t.Fatalf("core model calls = %d, want 3 (draft + failed tool loop + text fallback)", n)
	}
	if gov.redraftResult != "fallback-reworded" {
		t.Errorf("redraft result = %q, want the text-only fallback draft", gov.redraftResult)
	}
	var resultText string
	sawError := false
	for _, ev := range parseLines(t, out.String()) {
		if ev.Type == protocol.TypeResult {
			resultText = ev.Result.Text
			sawError = ev.Result.IsError
		}
	}
	if sawError {
		t.Errorf("tool-loop error broke the turn; the fallback should deliver cleanly:\n%s", out.String())
	}
	if resultText != "fallback-reworded" {
		t.Errorf("delivered result = %q, want fallback-reworded", resultText)
	}
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
	// The redraft now drives the tool loop, so its opening user message is the
	// critique followed by the deterministic bounded-action guard-rail. It must
	// still carry the critique verbatim (the loop reached it) and the guard-rail.
	if content, ok := last.Content.(string); !ok ||
		!strings.Contains(content, "CRITIQUE-TAIL-TEXT") ||
		!strings.Contains(content, redraftActionInstruction) {
		t.Errorf("redraft call did not carry the critique tail + guard-rail as its last message: %+v", redraftMsgs)
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
