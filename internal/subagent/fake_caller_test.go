package subagent

import (
	"context"
	"sync"

	"github.com/vingarcia/kortex/internal/mcp"
)

// scriptedCall is one canned response fakeCaller returns for a given gateway
// tool name, in call order (each name's calls are consumed in order; the
// last entry repeats once exhausted so a test need not script every poll of
// a long wait loop).
type scriptedCall struct {
	result mcp.CallToolResult
	err    error
}

// fakeCaller stands in for *mcp.Client across every subagent test: it
// records every call (name + arguments, in order) and answers from a
// per-tool-name script, so tests can assert both what Executor sent the
// gateway and how it reacted to a scripted gateway reply, without a live
// HTTP server.
type fakeCaller struct {
	mu       sync.Mutex
	scripts  map[string][]scriptedCall
	nextCall map[string]int
	calls    []fakeCallRecord
	// onCall, when set, runs synchronously inside CallTool before the
	// scripted reply is returned — tests use it to cancel a context after N
	// calls, proving the wait loop is ctx-aware.
	onCall func(name string, arguments map[string]any)
}

type fakeCallRecord struct {
	name      string
	arguments map[string]any
}

func newFakeCaller() *fakeCaller {
	return &fakeCaller{scripts: make(map[string][]scriptedCall), nextCall: make(map[string]int)}
}

// script queues calls for name, consumed in order as CallTool(name, ...) is
// invoked; the last queued entry repeats for every call past the end.
func (f *fakeCaller) script(name string, calls ...scriptedCall) *fakeCaller {
	f.scripts[name] = append(f.scripts[name], calls...)
	return f
}

func (f *fakeCaller) CallTool(ctx context.Context, name string, arguments map[string]any) (mcp.CallToolResult, error) {
	f.mu.Lock()
	f.calls = append(f.calls, fakeCallRecord{name: name, arguments: arguments})
	if f.onCall != nil {
		f.onCall(name, arguments)
	}
	queue := f.scripts[name]
	idx := f.nextCall[name]
	if idx >= len(queue) {
		idx = len(queue) - 1
	}
	f.nextCall[name]++
	f.mu.Unlock()

	if idx < 0 || idx >= len(queue) {
		return mcp.CallToolResult{}, nil
	}
	return queue[idx].result, queue[idx].err
}

func (f *fakeCaller) callsTo(name string) []fakeCallRecord {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []fakeCallRecord
	for _, c := range f.calls {
		if c.name == name {
			out = append(out, c)
		}
	}
	return out
}

func textResult(text string) mcp.CallToolResult {
	return mcp.CallToolResult{Content: []mcp.Content{{Type: "text", Text: text}}}
}

func errorResult(text string) mcp.CallToolResult {
	return mcp.CallToolResult{Content: []mcp.Content{{Type: "text", Text: text}}, IsError: true}
}
