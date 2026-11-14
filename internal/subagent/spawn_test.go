package subagent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestExecutor_RunSpawn(t *testing.T) {
	for name, tc := range map[string]struct {
		input        string
		spawnResult  scriptedCall
		waitScripts  map[string][]scriptedCall // only used by the wait=true cases
		wantErr      bool
		wantContains string
		// wantArgs, when set, checks the exact sessions_spawn arguments sent.
		wantArgs map[string]any
	}{
		"missing task is an error": {
			input:   `{"wait":false}`,
			wantErr: true,
		},
		"fable model is rejected before calling the gateway": {
			input:   `{"task":"do it","model":"fable","wait":false}`,
			wantErr: true,
		},
		"wait=false returns runId and childSessionKey without blocking": {
			input:        `{"task":"do it","wait":false}`,
			spawnResult:  scriptedCall{result: textResult(`{"taskId":"run-1","sessionKey":"sess-1"}`)},
			wantContains: `"runId":"run-1"`,
		},
		"task is prefixed with the no-yield preamble": {
			input:       `{"task":"do it","wait":false}`,
			spawnResult: scriptedCall{result: textResult(`{"taskId":"run-1","sessionKey":"sess-1"}`)},
			wantArgs: map[string]any{
				"task":             subagentPreamble + "\n\ndo it",
				"context":          "isolated",
				"completionTarget": "parent",
				"cleanup":          "keep",
				"runtime":          "subagent",
				"model":            "anthropic/claude-sonnet-5",
			},
		},
		"label is forwarded when given": {
			input:       `{"task":"do it","label":"my-run","wait":false}`,
			spawnResult: scriptedCall{result: textResult(`{"taskId":"run-1","sessionKey":"sess-1"}`)},
			wantArgs: map[string]any{
				"task":             subagentPreamble + "\n\ndo it",
				"context":          "isolated",
				"completionTarget": "parent",
				"cleanup":          "keep",
				"runtime":          "subagent",
				"model":            "anthropic/claude-sonnet-5",
				"label":            "my-run",
			},
		},
		"gateway tool-level error is forwarded": {
			input:        `{"task":"do it","wait":false}`,
			spawnResult:  scriptedCall{result: errorResult("boom")},
			wantErr:      true,
			wantContains: "boom",
		},
		"unparseable spawn result is a clear error, not a panic": {
			input:        `{"task":"do it","wait":false}`,
			spawnResult:  scriptedCall{result: textResult(`not json`)},
			wantErr:      true,
			wantContains: "cannot parse",
		},
		"wait=true (default) blocks and returns the child's final report": {
			input:       `{"task":"do it"}`,
			spawnResult: scriptedCall{result: textResult(`{"taskId":"run-1","sessionKey":"sess-1"}`)},
			waitScripts: map[string][]scriptedCall{
				gatewaySubagentsTool:       {{result: textResult(`{"taskId":"run-1","status":"completed"}`)}},
				gatewaySessionsHistoryTool: {{result: textResult(`[{"role":"assistant","text":"the final report"}]`)}},
			},
			wantContains: "the final report",
		},
	} {
		t.Run(name, func(t *testing.T) {
			caller := newFakeCaller().script(gatewaySpawnTool, tc.spawnResult)
			for toolName, calls := range tc.waitScripts {
				caller.script(toolName, calls...)
			}
			e := NewExecutor(caller)

			output, isError := e.runSpawn(context.Background(), json.RawMessage(tc.input))
			if isError != tc.wantErr {
				t.Fatalf("isError = %v, want %v (output=%q)", isError, tc.wantErr, output)
			}
			if tc.wantContains != "" && !strings.Contains(output, tc.wantContains) {
				t.Errorf("output = %q, want it to contain %q", output, tc.wantContains)
			}
			if tc.wantArgs != nil {
				calls := caller.callsTo(gatewaySpawnTool)
				if len(calls) != 1 {
					t.Fatalf("got %d calls to %s, want 1", len(calls), gatewaySpawnTool)
				}
				got := calls[0].arguments
				if len(got) != len(tc.wantArgs) {
					t.Fatalf("sessions_spawn args = %+v, want %+v", got, tc.wantArgs)
				}
				for k, want := range tc.wantArgs {
					if got[k] != want {
						t.Errorf("sessions_spawn arg %q = %v, want %v", k, got[k], want)
					}
				}
			}
		})
	}
}

func TestExecutor_RunSpawnPopulatesRegistry(t *testing.T) {
	caller := newFakeCaller().script(gatewaySpawnTool, scriptedCall{result: textResult(`{"taskId":"run-1","sessionKey":"sess-1"}`)})
	e := NewExecutor(caller)

	if _, isError := e.runSpawn(context.Background(), json.RawMessage(`{"task":"do it","label":"my-label","wait":false}`)); isError {
		t.Fatal("runSpawn reported an error")
	}
	rec, ok := e.registry.get("run-1")
	if !ok {
		t.Fatal("registry has no record for run-1 after a successful spawn")
	}
	if rec.sessionKey != "sess-1" || rec.label != "my-label" {
		t.Errorf("registry record = %+v, want sessionKey=sess-1 label=my-label", rec)
	}
}

func TestParseSpawnResult(t *testing.T) {
	for name, tc := range map[string]struct {
		raw            string
		wantRunID      string
		wantSessionKey string
		wantErr        bool
	}{
		"live gateway response (smoke 2026-11-14)": {
			raw:            `{"status":"accepted","childSessionKey":"agent:x:subagent:fac5","runId":"8f9c","mode":"run","completionTarget":"parent","modelApplied":true}`,
			wantRunID:      "8f9c",
			wantSessionKey: "agent:x:subagent:fac5",
		},
		"taskId and sessionKey": {
			raw:            `{"taskId":"t1","sessionKey":"s1"}`,
			wantRunID:      "t1",
			wantSessionKey: "s1",
		},
		"falls back to id and childSessionKey": {
			raw:            `{"id":"t2","childSessionKey":"s2"}`,
			wantRunID:      "t2",
			wantSessionKey: "s2",
		},
		"falls back to runId and sessionId": {
			raw:            `{"runId":"t3","sessionId":"s3"}`,
			wantRunID:      "t3",
			wantSessionKey: "s3",
		},
		"missing id fields is an error": {
			raw:     `{"sessionKey":"s1"}`,
			wantErr: true,
		},
		"missing session key fields is an error": {
			raw:     `{"taskId":"t1"}`,
			wantErr: true,
		},
		"non-JSON is an error": {
			raw:     `not json`,
			wantErr: true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			runID, sessionKey, err := parseSpawnResult(tc.raw)
			if (err != nil) != tc.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tc.wantErr)
			}
			if !tc.wantErr && (runID != tc.wantRunID || sessionKey != tc.wantSessionKey) {
				t.Errorf("parseSpawnResult(%q) = (%q, %q), want (%q, %q)", tc.raw, runID, sessionKey, tc.wantRunID, tc.wantSessionKey)
			}
		})
	}
}
