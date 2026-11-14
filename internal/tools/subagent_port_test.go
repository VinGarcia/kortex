package tools

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/mcp"
)

// fakeSubagentPort is a scripted SubagentPort standing in for
// internal/subagent.Executor, so Dispatcher's generic routing is tested
// without depending on that package (which would also violate the
// one-way subagent -> tools dependency direction: subagent imports tools,
// so tools importing subagent would be a cycle).
type fakeSubagentPort struct {
	defs      []anthropic.ToolDef
	output    string
	isError   bool
	gotName   string
	gotInput  json.RawMessage
	executedN int
}

func (f *fakeSubagentPort) ToolDefs() []anthropic.ToolDef { return f.defs }

func (f *fakeSubagentPort) Execute(ctx context.Context, name string, input json.RawMessage) (string, bool) {
	f.executedN++
	f.gotName = name
	f.gotInput = input
	return f.output, f.isError
}

func TestDispatcher_SubagentToolsOnlyWhenWired(t *testing.T) {
	has := func(d *Dispatcher) bool {
		for _, def := range d.ToolDefs() {
			if def.Name == "spawn_subagent" {
				return true
			}
		}
		return false
	}
	if has(NewDispatcher()) {
		t.Error("spawn_subagent declared with no port wired")
	}
	port := &fakeSubagentPort{defs: []anthropic.ToolDef{{Name: "spawn_subagent"}}}
	d := NewDispatcher().WithSubagentTools(port)
	if !has(d) {
		t.Error("spawn_subagent not declared once WithSubagentTools is wired")
	}
}

func TestDispatcher_ExecuteRoutesToSubagentPortByToolDefsMembership(t *testing.T) {
	port := &fakeSubagentPort{
		defs:   []anthropic.ToolDef{{Name: "spawn_subagent"}, {Name: "wait_subagents"}},
		output: `{"runId":"abc"}`,
	}
	d := NewDispatcher().WithSubagentTools(port)

	got := d.Execute(context.Background(), toolUseBlock("toolu_1", "spawn_subagent", `{"task":"do it"}`))
	if port.executedN != 1 {
		t.Fatalf("port.Execute called %d times, want 1", port.executedN)
	}
	if port.gotName != "spawn_subagent" {
		t.Errorf("port got name %q, want spawn_subagent", port.gotName)
	}
	if string(port.gotInput) != `{"task":"do it"}` {
		t.Errorf("port got input %s, want the raw tool_use input unchanged", port.gotInput)
	}
	if got.Content != `{"runId":"abc"}` || got.IsError {
		t.Errorf("result = %+v, want the port's scripted output forwarded as-is", got)
	}
}

func TestDispatcher_ExecuteSubagentToolNotWiredIsError(t *testing.T) {
	// A bare NewDispatcher() never declares spawn_subagent (see
	// TestDispatcher_SubagentToolsOnlyWhenWired), so reaching Execute with
	// that name here is the "model called something it was never told about"
	// case covered by Execute's existing unknown-tool fallthrough, not a
	// SubagentPort code path.
	got := NewDispatcher().Execute(context.Background(), toolUseBlock("toolu_1", "spawn_subagent", `{}`))
	if !got.IsError {
		t.Errorf("want is_error for spawn_subagent with no port wired, got %+v", got)
	}
}

func TestDispatcher_HideMCPTools(t *testing.T) {
	d := &Dispatcher{
		mcp:      &fakeMCPCaller{},
		mcpTools: []mcp.Tool{{Name: "sessions_spawn"}, {Name: "sessions_send"}, {Name: "subagents"}},
	}
	d.HideMCPTools("sessions_spawn", "subagents")

	got := d.ToolDefs()
	want := append(Definitions(), anthropic.ToolDef{Name: "mcp__openclaw__sessions_send"})
	if len(got) != len(want) {
		t.Fatalf("ToolDefs = %+v, want %+v", got, want)
	}
	for _, def := range got {
		if def.Name == "mcp__openclaw__sessions_spawn" || def.Name == "mcp__openclaw__subagents" {
			t.Errorf("hidden tool %q still declared in ToolDefs()", def.Name)
		}
	}
	found := false
	for _, def := range got {
		if def.Name == "mcp__openclaw__sessions_send" {
			found = true
		}
	}
	if !found {
		t.Error("non-hidden MCP tool sessions_send missing from ToolDefs()")
	}
}

func TestDispatcher_HideMCPToolsLeavesExecuteUnaffected(t *testing.T) {
	// Hiding is a declaration-time filter only (see HideMCPTools's doc
	// comment); Execute must still forward a hidden name exactly like any
	// other mcp__openclaw__* call, matching executeMCP's existing fail-open
	// behavior for a name outside mcpTools entirely.
	fake := &fakeMCPCaller{result: mcp.CallToolResult{Content: []mcp.Content{{Type: "text", Text: "ran anyway"}}}}
	d := (&Dispatcher{mcp: fake, mcpTools: []mcp.Tool{{Name: "sessions_spawn"}}}).HideMCPTools("sessions_spawn")

	got := d.Execute(context.Background(), toolUseBlock("toolu_1", "mcp__openclaw__sessions_spawn", `{}`))
	if got.Content != "ran anyway" || got.IsError {
		t.Errorf("Execute of a hidden-but-called MCP tool = %+v, want it to still run", got)
	}
}
