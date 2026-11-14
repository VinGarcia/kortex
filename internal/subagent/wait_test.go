package subagent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

// liveEntry builds one active[]/recent[] element in the shape observed live
// (2026-11-14): status is the human label, state is execution.state.
func liveEntry(id string, status string, state string) string {
	return `{"index":1,"runId":"` + id + `","sessionKey":"s-` + id + `","label":"l","status":"` + status + `","execution":{"state":"` + state + `"}}`
}

// liveListRaw wraps raw active/recent entries in a list response, plus a
// tasks[] array keyed by an unrelated taskId that must be ignored.
func liveListRaw(active []string, recent []string) string {
	return `{"status":"ok","action":"list","tasks":[{"taskId":"task-uuid","status":"completed","execution":{"state":"finished"},"label":"l"}],"active":[` + strings.Join(active, ",") + `],"recent":[` + strings.Join(recent, ",") + `]}`
}

// liveList lists activeIDs as running and recentIDs as done/finished.
func liveList(activeIDs []string, recentIDs []string) string {
	var a, r []string
	for _, id := range activeIDs {
		a = append(a, liveEntry(id, "running", "running"))
	}
	for _, id := range recentIDs {
		r = append(r, liveEntry(id, "done", "finished"))
	}
	return liveListRaw(a, r)
}

func historyEnvelope(text string) string {
	return `{"sessionKey":"k","messages":[{"role":"user","content":"task"},{"role":"assistant","content":[{"type":"text","text":"` + text + `"}],"stopReason":"stop"}],"truncated":true}`
}

// newTestExecutor returns an Executor polling every millisecond.
func newTestExecutor(caller *fakeCaller) *Executor {
	e := NewExecutor(caller)
	e.listPollInterval = time.Millisecond
	e.notFoundGrace = 0
	return e
}

func TestExecutor_WaitForAll_Finishes(t *testing.T) {
	caller := newFakeCaller().
		script(gatewaySubagentsTool,
			scriptedCall{result: textResult(liveList([]string{"run-2"}, []string{"run-1"}))},
			// run-2 finishes on the second poll.
			scriptedCall{result: textResult(liveList(nil, []string{"run-1", "run-2"}))}).
		scriptForSession(gatewaySessionsHistoryTool, "sess-1", scriptedCall{result: textResult(historyEnvelope("report one"))}).
		scriptForSession(gatewaySessionsHistoryTool, "sess-2", scriptedCall{result: textResult(historyEnvelope("report two"))})

	e := newTestExecutor(caller)
	e.registry.put("run-1", subagentRecord{sessionKey: "sess-1"})
	e.registry.put("run-2", subagentRecord{sessionKey: "sess-2"})

	statuses, err := e.waitForAll(context.Background(), []string{"run-1", "run-2"}, 5)
	if err != nil {
		t.Fatalf("waitForAll error: %v", err)
	}
	if got := statuses["run-1"].Result; got != "report one" {
		t.Errorf("run-1 result = %q, want %q", got, "report one")
	}
	if got := statuses["run-2"].Result; got != "report two" {
		t.Errorf("run-2 result = %q, want %q", got, "report two")
	}

	calls := caller.callsTo(gatewaySubagentsTool)
	if len(calls) != 2 {
		t.Fatalf("got %d subagents(list) calls, want 2 (run-2 needed a second poll)", len(calls))
	}
	if calls[0].arguments["action"] != "list" {
		t.Errorf("poll action = %v, want list (action=wait only accepts gateway taskIds, not runIds)", calls[0].arguments["action"])
	}
}

func TestExecutor_WaitForAll_UnknownStateIsNotTerminal(t *testing.T) {
	// "inactive"/"blocked" must neither read as finished nor as an error: the
	// wait runs to its timeout, whose message names the last state seen.
	for _, state := range []string{"inactive", "blocked", ""} {
		t.Run("state="+state, func(t *testing.T) {
			caller := newFakeCaller().script(gatewaySubagentsTool,
				scriptedCall{result: textResult(liveListRaw([]string{liveEntry("run-1", "done", state)}, nil))})
			e := newTestExecutor(caller)

			statuses, err := e.waitForAll(context.Background(), []string{"run-1"}, 1)
			if err != nil {
				t.Fatalf("waitForAll error: %v", err)
			}
			st := statuses["run-1"]
			if st.Status != "timeout" || !strings.Contains(st.Error, "did not finish") {
				t.Errorf("run-1 status = %+v, want timeout", st)
			}
			if state != "" && !strings.Contains(st.Error, state) {
				t.Errorf("error %q does not name the last state %q", st.Error, state)
			}
		})
	}
}

func TestExecutor_WaitForAll_DuplicateInActiveAndRecentPrefersFinished(t *testing.T) {
	list := liveListRaw(
		[]string{liveEntry("run-1", "running", "running")},
		[]string{liveEntry("run-1", "done", "finished")})
	caller := newFakeCaller().
		script(gatewaySubagentsTool, scriptedCall{result: textResult(list)}).
		scriptForSession(gatewaySessionsHistoryTool, "sess-1", scriptedCall{result: textResult(historyEnvelope("final"))})
	e := newTestExecutor(caller)
	e.registry.put("run-1", subagentRecord{sessionKey: "sess-1"})

	statuses, err := e.waitForAll(context.Background(), []string{"run-1"}, 5)
	if err != nil {
		t.Fatalf("waitForAll error: %v", err)
	}
	if statuses["run-1"].Result != "final" || statuses["run-1"].Error != "" {
		t.Errorf("run-1 = %+v, want finished with result", statuses["run-1"])
	}
	// Order must not matter: finished in active[], running in recent[].
	list = liveListRaw(
		[]string{liveEntry("run-1", "done", "finished")},
		[]string{liveEntry("run-1", "running", "running")})
	entries, err := parseList(list)
	if err != nil || !entries["run-1"].finished() {
		t.Errorf("parseList = %+v, %v; want run-1 finished", entries, err)
	}
}

func TestExecutor_WaitForAll_RunNeverSeenFailsEarly(t *testing.T) {
	caller := newFakeCaller().script(gatewaySubagentsTool, scriptedCall{result: textResult(liveList([]string{"run-2"}, []string{"run-3"}))}).
		scriptForSession(gatewaySessionsHistoryTool, "sess-3", scriptedCall{result: textResult(historyEnvelope("three"))})
	e := newTestExecutor(caller)
	e.registry.put("run-3", subagentRecord{sessionKey: "sess-3"})

	start := time.Now()
	statuses, err := e.waitForAll(context.Background(), []string{"run-1", "run-3"}, 60)
	if err != nil {
		t.Fatalf("waitForAll error: %v", err)
	}
	if time.Since(start) > 5*time.Second {
		t.Errorf("took %v, want an early failure instead of the 60s timeout", time.Since(start))
	}
	if st := statuses["run-1"]; st.Status != "not_found" || !strings.Contains(st.Error, "not found") {
		t.Errorf("run-1 = %+v, want a not_found error", st)
	}
	if statuses["run-3"].Result != "three" {
		t.Errorf("run-3 = %+v, want its result despite run-1 missing", statuses["run-3"])
	}
	if n := len(caller.callsTo(gatewaySubagentsTool)); n != e.notFoundPolls {
		t.Errorf("polled %d times, want %d (notFoundPolls)", n, e.notFoundPolls)
	}
}

func TestExecutor_WaitForAll_TransientListErrorRecovers(t *testing.T) {
	caller := newFakeCaller().
		script(gatewaySubagentsTool,
			scriptedCall{err: errors.New("connection reset")},
			scriptedCall{result: errorResult("gateway busy")},
			scriptedCall{result: textResult("not json")},
			scriptedCall{result: textResult(liveList(nil, []string{"run-1"}))}).
		scriptForSession(gatewaySessionsHistoryTool, "sess-1", scriptedCall{result: textResult(historyEnvelope("ok"))})
	e := newTestExecutor(caller)
	e.registry.put("run-1", subagentRecord{sessionKey: "sess-1"})

	statuses, err := e.waitForAll(context.Background(), []string{"run-1"}, 5)
	if err != nil {
		t.Fatalf("waitForAll error: %v, want recovery after maxListFailures (3) transient failures", err)
	}
	if statuses["run-1"].Result != "ok" {
		t.Errorf("run-1 = %+v, want result ok", statuses["run-1"])
	}
}

func TestExecutor_WaitForAll_PersistentListErrorReturnsPartialResults(t *testing.T) {
	caller := newFakeCaller().
		script(gatewaySubagentsTool,
			scriptedCall{result: textResult(liveList([]string{"run-2"}, []string{"run-1"}))},
			scriptedCall{err: errors.New("connection refused")}).
		scriptForSession(gatewaySessionsHistoryTool, "sess-1", scriptedCall{result: textResult(historyEnvelope("one"))})
	e := newTestExecutor(caller)
	e.registry.put("run-1", subagentRecord{sessionKey: "sess-1"})
	e.registry.put("run-2", subagentRecord{sessionKey: "sess-2"})

	statuses, err := e.waitForAll(context.Background(), []string{"run-1", "run-2"}, 5)
	if err == nil || !strings.Contains(err.Error(), "connection refused") {
		t.Fatalf("waitForAll error = %v, want the list error", err)
	}
	if statuses["run-1"].Result != "one" {
		t.Errorf("run-1 = %+v, want its finished result kept", statuses["run-1"])
	}
	if st := statuses["run-2"]; st.Status != "unknown" || st.Error == "" {
		t.Errorf("run-2 = %+v, want unknown with the error", st)
	}
	// 1 good poll + maxListFailures tolerated failures + the one that gives up.
	if n := len(caller.callsTo(gatewaySubagentsTool)); n != 1+e.maxListFailures+1 {
		t.Errorf("polled %d times, want %d", n, 1+e.maxListFailures+1)
	}

	// A fresh executor over the same script, so runWait starts at poll 1.
	caller = newFakeCaller().
		script(gatewaySubagentsTool,
			scriptedCall{result: textResult(liveList([]string{"run-2"}, []string{"run-1"}))},
			scriptedCall{err: errors.New("connection refused")}).
		scriptForSession(gatewaySessionsHistoryTool, "sess-1", scriptedCall{result: textResult(historyEnvelope("one"))})
	e = newTestExecutor(caller)
	e.registry.put("run-1", subagentRecord{sessionKey: "sess-1"})
	e.registry.put("run-2", subagentRecord{sessionKey: "sess-2"})
	out, isError := e.runWait(context.Background(), json.RawMessage(`{"ids":["run-1","run-2"],"timeout_seconds":5}`))
	if !isError || !strings.Contains(out, "partial results") || !strings.Contains(out, `"one"`) {
		t.Errorf("runWait = (%q, %v), want an error carrying the partial results", out, isError)
	}
}

func TestExecutor_WaitForAll_TimeoutLeavesUnfinishedIDsReported(t *testing.T) {
	caller := newFakeCaller().script(gatewaySubagentsTool, scriptedCall{result: textResult(liveList([]string{"run-1"}, nil))})
	e := newTestExecutor(caller)

	statuses, err := e.waitForAll(context.Background(), []string{"run-1"}, 1)
	if err != nil {
		t.Fatalf("waitForAll error: %v", err)
	}
	st := statuses["run-1"]
	if st.Error == "" || !strings.Contains(st.Error, "did not finish") {
		t.Errorf("run-1 status = %+v, want a timeout error", st)
	}
}

func TestExecutor_WaitForAll_CtxCancelStopsPromptly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	caller := newFakeCaller().script(gatewaySubagentsTool, scriptedCall{result: textResult(liveList([]string{"run-1"}, nil))})
	caller.onCall = func(name string, arguments map[string]any) {
		if name == gatewaySubagentsTool {
			cancel()
		}
	}
	e := NewExecutor(caller)

	_, err := e.waitForAll(ctx, []string{"run-1"}, 1800)
	if err == nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("waitForAll error = %v, want context.Canceled", err)
	}
	if calls := caller.callsTo(gatewaySubagentsTool); len(calls) != 1 {
		t.Fatalf("got %d subagents(wait) calls, want exactly 1 before the cancel was observed", len(calls))
	}
}

func TestExecutor_WaitForAll_MissingRegistryRecordIsAClearError(t *testing.T) {
	caller := newFakeCaller().script(gatewaySubagentsTool, scriptedCall{result: textResult(liveList(nil, []string{"run-1"}))})
	e := newTestExecutor(caller) // registry never populated: simulates a resumed process.

	statuses, err := e.waitForAll(context.Background(), []string{"run-1"}, 5)
	if err != nil {
		t.Fatalf("waitForAll error: %v", err)
	}
	if !strings.Contains(statuses["run-1"].Error, "no session-key record") {
		t.Errorf("run-1 status = %+v, want a missing-registry error", statuses["run-1"])
	}
}

func TestExecutor_FetchFinalText_TruncatesWithVisibleMarker(t *testing.T) {
	long := strings.Repeat("a", resultMaxBytes+1000)
	raw, err := json.Marshal(long)
	if err != nil {
		t.Fatal(err)
	}
	caller := newFakeCaller().script(gatewaySessionsHistoryTool, scriptedCall{result: textResult(string(raw))})
	e := NewExecutor(caller)

	text, err := e.fetchFinalText(context.Background(), "sess-1")
	if err != nil {
		t.Fatalf("fetchFinalText error: %v", err)
	}
	if !strings.Contains(text, "[truncated: showing first") {
		t.Errorf("text has no visible truncation marker: %.200q...", text)
	}
	if len(text) > resultMaxBytes+100 {
		t.Errorf("truncated text is %d bytes, want close to the %d byte cap", len(text), resultMaxBytes)
	}
}

func TestExtractLastAssistantText(t *testing.T) {
	for name, tc := range map[string]struct {
		raw  string
		want string
	}{
		"structured history picks the last assistant text": {
			raw:  `[{"role":"user","text":"hi"},{"role":"assistant","text":"first"},{"role":"user","text":"more"},{"role":"assistant","text":"second"}]`,
			want: "second",
		},
		"structured history falls back to content field": {
			raw:  `[{"role":"assistant","content":"from content"}]`,
			want: "from content",
		},
		"live envelope with content blocks": {
			raw:  `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"text","text":"PONG"}]}],"truncated":true}`,
			want: "PONG",
		},
		"non-JSON falls back to the raw text verbatim": {
			raw:  "plain transcript text, not JSON",
			want: "plain transcript text, not JSON",
		},
		"empty array falls back to raw": {
			raw:  "[]",
			want: "[]",
		},
	} {
		t.Run(name, func(t *testing.T) {
			if got := extractLastAssistantText(tc.raw); got != tc.want {
				t.Errorf("extractLastAssistantText(%q) = %q, want %q", tc.raw, got, tc.want)
			}
		})
	}
}

func TestParseList(t *testing.T) {
	entries, err := parseList(liveList([]string{"run-2"}, []string{"run-1"}))
	if err != nil {
		t.Fatalf("parseList error: %v", err)
	}
	if len(entries) != 2 || !entries["run-1"].finished() || entries["run-2"].finished() {
		t.Errorf("entries = %+v, want run-1 finished and run-2 running (tasks[] ignored)", entries)
	}
	if _, found := entries["task-uuid"]; found {
		t.Error("tasks[] taskId must not be treated as a runId")
	}
	if _, err := parseList("not json"); err == nil {
		t.Error("parseList(non-JSON) error = nil, want a decode error")
	}
}

func TestExecutor_WaitForAll_VanishedRunKeepsWaitingAndRecovers(t *testing.T) {
	caller := newFakeCaller().
		script(gatewaySubagentsTool,
			scriptedCall{result: textResult(liveList([]string{"run-1"}, nil))},
			// Dropped from both lists for more than notFoundPolls polls.
			scriptedCall{result: textResult(liveList(nil, nil))},
			scriptedCall{result: textResult(liveList(nil, nil))},
			scriptedCall{result: textResult(liveList(nil, nil))},
			scriptedCall{result: textResult(liveList(nil, []string{"run-1"}))}).
		scriptForSession(gatewaySessionsHistoryTool, "sess-1", scriptedCall{result: textResult(historyEnvelope("back"))})
	e := newTestExecutor(caller)
	e.registry.put("run-1", subagentRecord{sessionKey: "sess-1"})

	statuses, err := e.waitForAll(context.Background(), []string{"run-1"}, 5)
	if err != nil {
		t.Fatalf("waitForAll error: %v", err)
	}
	if st := statuses["run-1"]; st.Status == "not_found" || st.Result != "back" {
		t.Errorf("run-1 = %+v, want its report once it reappears (a seen run is never not_found)", st)
	}
}

func TestExecutor_WaitForAll_VanishedRunTimesOutWithHint(t *testing.T) {
	caller := newFakeCaller().script(gatewaySubagentsTool,
		scriptedCall{result: textResult(liveList([]string{"run-1"}, nil))},
		scriptedCall{result: textResult(liveList(nil, nil))})
	e := newTestExecutor(caller)

	statuses, err := e.waitForAll(context.Background(), []string{"run-1"}, 1)
	if err != nil {
		t.Fatalf("waitForAll error: %v", err)
	}
	if st := statuses["run-1"]; st.Status != "timeout" || !strings.Contains(st.Error, "no longer in the gateway's list") {
		t.Errorf("run-1 = %+v, want a timeout that says it vanished", st)
	}
}

func TestExecutor_WaitForAll_NotFoundWaitsOutGraceWindow(t *testing.T) {
	caller := newFakeCaller().script(gatewaySubagentsTool, scriptedCall{result: textResult(liveList(nil, nil))})
	e := newTestExecutor(caller)
	e.notFoundGrace = time.Hour // far longer than the 1s wait

	statuses, err := e.waitForAll(context.Background(), []string{"run-1"}, 1)
	if err != nil {
		t.Fatalf("waitForAll error: %v", err)
	}
	if st := statuses["run-1"]; st.Status != "timeout" {
		t.Errorf("run-1 = %+v, want timeout: not_found must wait for the grace window", st)
	}
}

func TestExecutor_WaitForAll_NotFoundHintsWhenIDIsNotARunID(t *testing.T) {
	caller := newFakeCaller().script(gatewaySubagentsTool, scriptedCall{result: textResult(liveList(nil, nil))})
	e := newTestExecutor(caller)
	e.registry.put("task-1", subagentRecord{sessionKey: "s", idNotRunID: true})

	statuses, err := e.waitForAll(context.Background(), []string{"task-1"}, 5)
	if err != nil {
		t.Fatalf("waitForAll error: %v", err)
	}
	if st := statuses["task-1"]; st.Status != "not_found" || !strings.Contains(st.Error, "may not match the list's runId") {
		t.Errorf("task-1 = %+v, want a not_found error with the id-source hint", st)
	}
}

func TestExecutor_WaitForAll_FinalPollAfterDeadline(t *testing.T) {
	// run-1 is running on the first poll and finished on every later one;
	// with a 1s wait and a 600ms interval, only the post-deadline poll can
	// see it finished.
	caller := newFakeCaller().
		script(gatewaySubagentsTool,
			scriptedCall{result: textResult(liveList([]string{"run-1"}, nil))},
			scriptedCall{result: textResult(liveList([]string{"run-1"}, nil))},
			scriptedCall{result: textResult(liveList(nil, []string{"run-1"}))}).
		scriptForSession(gatewaySessionsHistoryTool, "sess-1", scriptedCall{result: textResult(historyEnvelope("late"))})
	e := newTestExecutor(caller)
	e.listPollInterval = 600 * time.Millisecond
	e.registry.put("run-1", subagentRecord{sessionKey: "sess-1"})

	statuses, err := e.waitForAll(context.Background(), []string{"run-1"}, 1)
	if err != nil {
		t.Fatalf("waitForAll error: %v", err)
	}
	if statuses["run-1"].Result != "late" {
		t.Errorf("run-1 = %+v, want the result from the final poll, not a timeout", statuses["run-1"])
	}
}

func TestExecutor_RunWait_IDsDefaulting(t *testing.T) {
	tests := []struct {
		desc          string
		input         string
		registry      []string
		lists         []string
		wantOut       []string
		wantListCalls int
		fresh         []string // registry ids spawned just now, with a 1h grace window
	}{
		{
			desc:     "empty ids waits for a just-spawned child the gateway does not list yet",
			input:    `{"timeout_seconds":5}`,
			registry: []string{"run-1", "run-2"},
			fresh:    []string{"run-2"}, // run-1 is old and unlisted: pruned, excluded.
			lists: []string{
				liveList(nil, nil), // resolves the pending set: run-2 not listed yet
				liveList([]string{"run-2"}, nil),
				liveList(nil, []string{"run-2"}),
			},
			wantOut:       []string{`"runId":"run-2"`, `report run-2`},
			wantListCalls: 3,
		},
		{
			desc:     "empty ids waits for both pending children and skips finished ones",
			input:    `{"timeout_seconds":5}`,
			registry: []string{"run-1", "run-2", "run-3"},
			lists: []string{
				liveList([]string{"run-1", "run-2"}, []string{"run-3"}), // resolves the pending set
				liveList([]string{"run-2"}, []string{"run-1", "run-3"}),
				liveList(nil, []string{"run-1", "run-2", "run-3"}),
			},
			wantOut:       []string{`"runId":"run-1"`, `"runId":"run-2"`, `report run-1`, `report run-2`},
			wantListCalls: 3, // 1 to resolve the pending set + 2 polls (run-2 needs a second).
		},
		{
			desc:          "explicit empty ids list behaves like omitted",
			input:         `{"ids":[],"timeout_seconds":5}`,
			registry:      []string{"run-1"},
			lists:         []string{liveList(nil, []string{"run-1"})},
			wantOut:       []string{"[]"},
			wantListCalls: 1,
		},
		{
			desc:          "empty ids with an empty registry returns immediately without listing",
			input:         `{}`,
			wantOut:       []string{"[]"},
			wantListCalls: 0,
		},
		{
			desc:          "explicit ids are used as given, finished or not",
			input:         `{"ids":["run-1"],"timeout_seconds":5}`,
			registry:      []string{"run-1", "run-2"},
			lists:         []string{liveList([]string{"run-2"}, []string{"run-1"})},
			wantOut:       []string{`"runId":"run-1"`, `report run-1`},
			wantListCalls: 1,
		},
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			caller := newFakeCaller()
			for _, l := range test.lists {
				caller.script(gatewaySubagentsTool, scriptedCall{result: textResult(l)})
			}
			e := newTestExecutor(caller)
			e.notFoundGrace = time.Hour
			for _, id := range test.registry {
				sk := "sess-" + id
				caller.scriptForSession(gatewaySessionsHistoryTool, sk, scriptedCall{result: textResult(historyEnvelope("report " + id))})
				rec := subagentRecord{sessionKey: sk}
				for _, f := range test.fresh {
					if f == id {
						rec.spawnedAt = time.Now()
					}
				}
				e.registry.put(id, rec)
			}

			out, isError := e.runWait(context.Background(), json.RawMessage(test.input))
			if isError {
				t.Fatalf("runWait error: %s", out)
			}
			for _, want := range test.wantOut {
				if !strings.Contains(out, want) {
					t.Errorf("output %q missing %q", out, want)
				}
			}
			if len(test.fresh) == 0 && strings.Contains(out, "run-3") {
				t.Errorf("output %q includes already-finished run-3", out)
			}
			if len(test.fresh) > 0 && strings.Contains(out, `"run-1"`) {
				t.Errorf("output %q includes pruned run-1", out)
			}
			if n := len(caller.callsTo(gatewaySubagentsTool)); n != test.wantListCalls {
				t.Errorf("list calls = %d, want %d", n, test.wantListCalls)
			}
		})
	}
}
