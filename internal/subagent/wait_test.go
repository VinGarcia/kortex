package subagent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

// liveList builds a subagents action="list" response in the shape observed
// live (2026-11-14): entries keyed by runId under active[] (status "running")
// and recent[] (status "done"), plus a tasks[] array keyed by an unrelated
// taskId that must be ignored.
func liveList(active []string, recent []string) string {
	entry := func(id string, status string) string {
		return `{"index":1,"runId":"` + id + `","sessionKey":"s-` + id + `","label":"l","status":"` + status + `","execution":{"state":"x"}}`
	}
	var a, r []string
	for _, id := range active {
		a = append(a, entry(id, "running"))
	}
	for _, id := range recent {
		r = append(r, entry(id, "done"))
	}
	return `{"status":"ok","action":"list","tasks":[{"taskId":"task-uuid","status":"running","label":"l"}],"active":[` + strings.Join(a, ",") + `],"recent":[` + strings.Join(r, ",") + `]}`
}

func historyEnvelope(text string) string {
	return `{"sessionKey":"k","messages":[{"role":"user","content":"task"},{"role":"assistant","content":[{"type":"text","text":"` + text + `"}],"stopReason":"stop"}],"truncated":true}`
}

func init() { listPollInterval = time.Millisecond }

func TestExecutor_WaitForAll_Finishes(t *testing.T) {
	caller := newFakeCaller().
		script(gatewaySubagentsTool, scriptedCall{result: textResult(liveList([]string{"run-2"}, []string{"run-1"}))}).
		script(gatewaySessionsHistoryTool,
			scriptedCall{result: textResult(historyEnvelope("report one"))},
			scriptedCall{result: textResult(historyEnvelope("report two"))})
	// run-2 finishes on the second poll.
	caller.scripts[gatewaySubagentsTool] = append(caller.scripts[gatewaySubagentsTool],
		scriptedCall{result: textResult(liveList(nil, []string{"run-1", "run-2"}))})

	e := NewExecutor(caller)
	e.registry.put("run-1", subagentRecord{sessionKey: "sess-1"})
	e.registry.put("run-2", subagentRecord{sessionKey: "sess-2"})

	statuses, err := e.waitForAll(context.Background(), []string{"run-1", "run-2"}, 5)
	if err != nil {
		t.Fatalf("waitForAll error: %v", err)
	}
	got := []string{statuses["run-1"].Result, statuses["run-2"].Result}
	if !(got[0] == "report one" && got[1] == "report two") && !(got[0] == "report two" && got[1] == "report one") {
		t.Errorf("results = %q, want both reports fetched", got)
	}

	calls := caller.callsTo(gatewaySubagentsTool)
	if len(calls) != 2 {
		t.Fatalf("got %d subagents(list) calls, want 2 (run-2 needed a second poll)", len(calls))
	}
	if calls[0].arguments["action"] != "list" {
		t.Errorf("poll action = %v, want list (action=wait only accepts gateway taskIds, not runIds)", calls[0].arguments["action"])
	}
}

func TestExecutor_WaitForAll_TimeoutLeavesUnfinishedIDsReported(t *testing.T) {
	caller := newFakeCaller().script(gatewaySubagentsTool, scriptedCall{result: textResult(liveList([]string{"run-1"}, nil))})
	e := NewExecutor(caller)

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
	e := NewExecutor(caller) // registry never populated: simulates a resumed process.

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

func TestParseSubagentsStatuses(t *testing.T) {
	for name, tc := range map[string]struct {
		raw       string
		queried   []string
		wantFound map[string]subagentStatus
	}{
		"bare array of entries": {
			raw:     `[{"taskId":"run-1","status":"completed"},{"taskId":"run-2","status":"running"}]`,
			queried: []string{"run-1", "run-2"},
			wantFound: map[string]subagentStatus{
				"run-1": {Status: "completed", Finished: true},
				"run-2": {Status: "running", Finished: false},
			},
		},
		"wrapped under a tasks key": {
			raw:     `{"tasks":[{"id":"run-1","status":"failed","error":"boom"}]}`,
			queried: []string{"run-1"},
			wantFound: map[string]subagentStatus{
				"run-1": {Status: "failed", Finished: true, Error: "boom"},
			},
		},
		"live list shape: runId in active/recent, taskId in tasks ignored": {
			raw:     liveList([]string{"run-2"}, []string{"run-1"}),
			queried: []string{"run-1", "run-2", "task-uuid"},
			wantFound: map[string]subagentStatus{
				"run-1":     {Status: "done", Finished: true},
				"run-2":     {Status: "running", Finished: false},
				"task-uuid": {Status: "running", Finished: false},
			},
		},
		"unqueried entries are ignored": {
			raw:       `[{"taskId":"other","status":"completed"}]`,
			queried:   []string{"run-1"},
			wantFound: map[string]subagentStatus{},
		},
		"an error field marks it finished even with an unrecognized status word": {
			raw:     `[{"taskId":"run-1","status":"blocked","errorMessage":"needs input"}]`,
			queried: []string{"run-1"},
			wantFound: map[string]subagentStatus{
				"run-1": {Status: "blocked", Finished: true, Error: "needs input"},
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			got := parseSubagentsStatuses(tc.raw, tc.queried)
			if len(got) != len(tc.wantFound) {
				t.Fatalf("parseSubagentsStatuses = %+v, want %+v", got, tc.wantFound)
			}
			for id, want := range tc.wantFound {
				if got[id] != want {
					t.Errorf("status[%q] = %+v, want %+v", id, got[id], want)
				}
			}
		})
	}
}

func TestParseSubagentsStatuses_NonJSONReturnsNil(t *testing.T) {
	if got := parseSubagentsStatuses("not json", []string{"run-1"}); got != nil {
		t.Errorf("parseSubagentsStatuses(non-JSON) = %+v, want nil", got)
	}
}
