package subagent

import "sync"

// subagentRecord is what Executor remembers about one spawned child, keyed by
// runID in registry.
type subagentRecord struct {
	sessionKey string
	label      string
}

// registry maps a spawn_subagent call's runID to the child's gateway session
// key and label, scoped to this kortex process. The gateway — not this map —
// is the system of record for whether a child is still running and what it
// returned: wait_subagents and list_subagents always re-ask it via the
// subagents/sessions_history MCP tools. This map only caches the one field
// (sessionKey) that spawn time learns and wait/list time need to look the
// child up again, with an explicit degrade-and-report path (see wait.go's
// handling of a registry miss) for the case a resumed process starts with
// this map empty — session.Run's doc comment notes one kortex process
// "serves every turn of a session until stdin closes", so the map is valid
// for exactly that lifetime, same scope as the rest of a native session's
// in-memory state (e.g. session.Run's own `turns`/`priorUserTexts`
// accumulators).
type registry struct {
	mu      sync.Mutex
	records map[string]subagentRecord
}

func newRegistry() *registry {
	return &registry{records: make(map[string]subagentRecord)}
}

func (r *registry) put(runID string, rec subagentRecord) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.records[runID] = rec
}

func (r *registry) get(runID string) (subagentRecord, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	rec, ok := r.records[runID]
	return rec, ok
}

// all returns a snapshot of every known record, keyed by runID, for
// list_subagents to enrich the gateway's own listing with the label this
// process assigned at spawn time.
func (r *registry) all() map[string]subagentRecord {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make(map[string]subagentRecord, len(r.records))
	for k, v := range r.records {
		out[k] = v
	}
	return out
}
