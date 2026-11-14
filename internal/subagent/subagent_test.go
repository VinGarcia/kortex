package subagent

import (
	"context"
	"encoding/json"
	"testing"
)

func TestExecutor_ToolDefs(t *testing.T) {
	e := NewExecutor(newFakeCaller())
	got := e.ToolDefs()
	if len(got) != 3 {
		t.Fatalf("ToolDefs returned %d tools, want 3", len(got))
	}
	wantNames := []string{SpawnToolName, WaitToolName, ListToolName}
	for i, def := range got {
		if def.Name != wantNames[i] {
			t.Errorf("ToolDefs[%d].Name = %q, want %q", i, def.Name, wantNames[i])
		}
	}
}

func TestExecutor_Execute_UnknownNameIsAClearError(t *testing.T) {
	e := NewExecutor(newFakeCaller())
	output, isError := e.Execute(context.Background(), "not_a_real_tool", json.RawMessage(`{}`))
	if !isError {
		t.Fatalf("isError = false, want true (output=%q)", output)
	}
}

func TestHiddenGatewayTools(t *testing.T) {
	got := HiddenGatewayTools()
	want := map[string]bool{"sessions_spawn": true, "agents_wait": true, "subagents": true}
	if len(got) != len(want) {
		t.Fatalf("HiddenGatewayTools = %v, want exactly %v", got, want)
	}
	for _, name := range got {
		if !want[name] {
			t.Errorf("HiddenGatewayTools contains unexpected %q", name)
		}
	}
}
