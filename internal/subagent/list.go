package subagent

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
)

var listDef = mustAnthropicToolDef(
	ListToolName,
	"List every subagent run spawned by this session, with its status.",
	map[string]any{"type": "object", "properties": map[string]any{}},
)

// listResultEntry is one element of list_subagents' returned JSON array.
type listResultEntry struct {
	RunID  string `json:"runId"`
	Label  string `json:"label,omitempty"`
	Status string `json:"status"`
	Error  string `json:"error,omitempty"`
}

func (e *Executor) runList(ctx context.Context, _ json.RawMessage) (output string, isError bool) {
	records := e.registry.all()

	entries := make([]listResultEntry, 0, len(records))
	if len(records) > 0 {
		ids := make([]string, 0, len(records))
		for id := range records {
			ids = append(ids, id)
		}

		result, err := e.mcp.CallTool(ctx, gatewaySubagentsTool, map[string]any{"action": "list"})
		if err != nil {
			return fmt.Sprintf("error: calling %s: %v", gatewaySubagentsTool, err), true
		}
		if result.IsError {
			return fmt.Sprintf("error: %s: %s", gatewaySubagentsTool, joinContent(result)), true
		}
		statuses := parseSubagentsStatuses(joinContent(result), ids)

		for id, rec := range records {
			st := statuses[id]
			entries = append(entries, listResultEntry{RunID: id, Label: rec.label, Status: st.Status, Error: st.Error})
		}
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].RunID < entries[j].RunID })

	out, err := json.Marshal(entries)
	if err != nil {
		return fmt.Sprintf("error: encoding %s result: %v", ListToolName, err), true
	}
	return string(out), false
}
