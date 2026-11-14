package subagent

import (
	"context"
	"encoding/json"
	"testing"
)

func TestExecutor_RunList(t *testing.T) {
	t.Run("no spawned runs skips calling the gateway", func(t *testing.T) {
		caller := newFakeCaller()
		e := NewExecutor(caller)

		output, isError := e.runList(context.Background(), nil)
		if isError {
			t.Fatalf("isError = true, output=%q", output)
		}
		if output != "[]" {
			t.Errorf("output = %q, want an empty array", output)
		}
		if calls := caller.callsTo(gatewaySubagentsTool); len(calls) != 0 {
			t.Errorf("got %d calls to %s with no registered runs, want 0", len(calls), gatewaySubagentsTool)
		}
	})

	t.Run("enriches the gateway's statuses with the registry's labels", func(t *testing.T) {
		caller := newFakeCaller().script(gatewaySubagentsTool, scriptedCall{
			result: textResult(`[{"taskId":"run-1","status":"completed"},{"taskId":"run-2","status":"running"}]`),
		})
		e := NewExecutor(caller)
		e.registry.put("run-1", subagentRecord{sessionKey: "sess-1", label: "first"})
		e.registry.put("run-2", subagentRecord{sessionKey: "sess-2", label: "second"})

		output, isError := e.runList(context.Background(), nil)
		if isError {
			t.Fatalf("isError = true, output=%q", output)
		}

		var entries []listResultEntry
		if err := json.Unmarshal([]byte(output), &entries); err != nil {
			t.Fatalf("output is not valid JSON: %v (output=%s)", err, output)
		}
		if len(entries) != 2 {
			t.Fatalf("got %d entries, want 2: %+v", len(entries), entries)
		}
		want := map[string]listResultEntry{
			"run-1": {RunID: "run-1", Label: "first", Status: "completed"},
			"run-2": {RunID: "run-2", Label: "second", Status: "running"},
		}
		for _, e := range entries {
			if e != want[e.RunID] {
				t.Errorf("entry %+v, want %+v", e, want[e.RunID])
			}
		}
	})

	t.Run("gateway tool-level error is forwarded", func(t *testing.T) {
		caller := newFakeCaller().script(gatewaySubagentsTool, scriptedCall{result: errorResult("gateway down")})
		e := NewExecutor(caller)
		e.registry.put("run-1", subagentRecord{sessionKey: "sess-1"})

		output, isError := e.runList(context.Background(), nil)
		if !isError {
			t.Fatalf("isError = false, want true (output=%q)", output)
		}
	})
}
