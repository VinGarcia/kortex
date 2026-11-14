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
		listed, err := e.pollList(ctx)
		if err != nil {
			return fmt.Sprintf("error: %v", err), true
		}

		for id, rec := range records {
			entry := listResultEntry{RunID: id, Label: rec.label}
			if l, ok := listed[id]; ok {
				entry.Status = l.Status
			} else {
				entry.Status = "not_found"
				entry.Error = "not present in the gateway's run list"
			}
			entries = append(entries, entry)
		}
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].RunID < entries[j].RunID })

	out, err := json.Marshal(entries)
	if err != nil {
		return fmt.Sprintf("error: encoding %s result: %v", ListToolName, err), true
	}
	return string(out), false
}
