package subagent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestExecutor_WaitForAll_Finishes(t *testing.T) {
	caller := newFakeCaller().
		script(gatewaySubagentsTool, scriptedCall{result: textResult(`[{"taskId":"run-1","status":"completed"},{"taskId":"run-2","status":"pending"}]`)}).
		script(gatewaySessionsHistoryTool, scriptedCall{result: textResult(`[{"role":"assistant","text":"report one"}]`)})
	// run-2 finishes on the second poll.
	caller.scripts[gatewaySubagentsTool] = append(caller.scripts[gatewaySubagentsTool],
		scriptedCall{result: textResult(`[{"taskId":"run-2","status":"completed"}]`)})
	caller.scripts[gatewaySessionsHistoryTool] = append(caller.scripts[gatewaySessionsHistoryTool],
		scriptedCall{result: textResult(`[{"role":"assistant","text":"report two"}]`)})

	e := NewExecutor(caller)
	e.registry.put("run-1", subagentRecord{sessionKey: "sess-1"})
	e.registry.put("run-2", subagentRecord{sessionKey: "sess-2"})

	statuses, err := e.waitForAll(context.Background(), []string{"run-1", "run-2"}, 5)
	if err != nil {
		t.Fatalf("waitForAll error: %v", err)
	}
	if statuses["run-1"].Result != "report one" {
		t.Errorf("run-1 result = %q, want %q", statuses["run-1"].Result, "report one")
	}
	if statuses["run-2"].Result != "report two" {
		t.Errorf("run-2 result = %q, want %q", statuses["run-2"].Result, "report two")
	}

	calls := caller.callsTo(gatewaySubagentsTool)
	if len(calls) != 2 {
		t.Fatalf("got %d subagents(wait) calls, want 2 (run-2 needed a second poll)", len(calls))
	}
	if ids, ok := calls[1].arguments["taskIds"].([]string); !ok || len(ids) != 1 || ids[0] != "run-2" {
		t.Errorf("second poll taskIds = %v, want only the still-pending run-2", calls[1].arguments["taskIds"])
	}
}

func TestExecutor_WaitForAll_TimeoutLeavesUnfinishedIDsReported(t *testing.T) {
	caller := newFakeCaller().script(gatewaySubagentsTool, scriptedCall{result: textResult(`[{"taskId":"run-1","status":"running"}]`)})
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
	caller := newFakeCaller().script(gatewaySubagentsTool, scriptedCall{result: textResult(`[{"taskId":"run-1","status":"running"}]`)})
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
	caller := newFakeCaller().script(gatewaySubagentsTool, scriptedCall{result: textResult(`[{"taskId":"run-1","status":"completed"}]`)})
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
