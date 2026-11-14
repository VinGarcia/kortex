package subagent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/vingarcia/kortex/internal/tools"
)

// defaultWaitTimeoutSeconds bounds wait_subagents (and spawn_subagent's own
// wait=true path, which delegates to the same loop) when the tool input
// omits timeout_seconds.
const defaultWaitTimeoutSeconds = 1800

// gatewayWaitPollSeconds is the "subagents" action="wait" tool's own
// timeoutSeconds cap, per the gateway's published schema (maximum: 60): one
// wait call blocks server-side for at most this long, so waitForAll polls in
// slices of this size until its own (longer) deadline.
const gatewayWaitPollSeconds = 60

// resultMaxBytes caps one subagent's fetched report, so a child that ignored
// the preamble's size guidance cannot balloon the next request's token cost.
const resultMaxBytes = 8 << 10

// historyFetchLimit bounds the sessions_history call waitForAll makes to
// read a finished child's final report: enough to find the last assistant
// message without pulling the child's whole transcript.
const historyFetchLimit = 5

var waitDef = mustAnthropicToolDef(
	WaitToolName,
	"Block until every listed subagent run finishes (or the timeout elapses), then return each run's final report.",
	map[string]any{
		"type": "object",
		"properties": map[string]any{
			"ids":             map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "The run ids returned by spawn_subagent with wait=false."},
			"timeout_seconds": map[string]any{"type": "integer", "description": "How long to block before giving up on whatever hasn't finished; defaults to 1800 (30 minutes)."},
		},
		"required": []string{"ids"},
	},
)

type waitInput struct {
	IDs            []string `json:"ids"`
	TimeoutSeconds int      `json:"timeout_seconds,omitempty"`
}

// waitResultEntry is one element of wait_subagents' returned JSON array.
type waitResultEntry struct {
	RunID  string `json:"runId"`
	Status string `json:"status"`
	Result string `json:"result,omitempty"`
	Error  string `json:"error,omitempty"`
}

func (e *Executor) runWait(ctx context.Context, rawInput json.RawMessage) (output string, isError bool) {
	var input waitInput
	if err := json.Unmarshal(rawInput, &input); err != nil {
		return fmt.Sprintf("error: decoding %s input: %v", WaitToolName, err), true
	}
	if len(input.IDs) == 0 {
		return "error: ids is required and must be non-empty", true
	}
	timeout := input.TimeoutSeconds
	if timeout <= 0 {
		timeout = defaultWaitTimeoutSeconds
	}

	statuses, err := e.waitForAll(ctx, input.IDs, timeout)
	if err != nil {
		return fmt.Sprintf("error: %v", err), true
	}

	entries := make([]waitResultEntry, 0, len(input.IDs))
	for _, id := range input.IDs {
		st := statuses[id]
		entries = append(entries, waitResultEntry{RunID: id, Status: st.Status, Result: st.Result, Error: st.Error})
	}
	out, err := json.Marshal(entries)
	if err != nil {
		return fmt.Sprintf("error: encoding %s result: %v", WaitToolName, err), true
	}
	return string(out), false
}

// subagentStatus is what waitForAll learns about one run id: the gateway's
// own status label, whether that label is terminal, and — once resolved via
// sessions_history — the child's final report or a reason it could not be
// fetched.
type subagentStatus struct {
	Status   string
	Finished bool
	Error    string
	Result   string
}

// waitForAll polls the gateway's "subagents" action="wait" tool for ids
// until every one is reported finished or timeoutSeconds elapses, then
// fetches each finished (non-error) id's final report via sessions_history.
// It is ctx-aware: a cancelled ctx stops the loop before the next poll
// rather than running it to the timeout. Shared by wait_subagents and
// spawn_subagent's wait=true path.
func (e *Executor) waitForAll(ctx context.Context, ids []string, timeoutSeconds int) (map[string]subagentStatus, error) {
	deadline := time.Now().Add(time.Duration(timeoutSeconds) * time.Second)
	remaining := append([]string(nil), ids...)
	results := make(map[string]subagentStatus, len(ids))

	for len(remaining) > 0 {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("%s: %w", WaitToolName, err)
		}
		remainingTime := time.Until(deadline)
		if remainingTime <= 0 {
			break
		}
		pollSeconds := gatewayWaitPollSeconds
		if secs := int(remainingTime.Seconds()); secs < pollSeconds {
			pollSeconds = secs
		}
		if pollSeconds <= 0 {
			// A sub-second remainder still deserves one last poll rather than
			// being declared a timeout without ever asking the gateway.
			pollSeconds = 1
		}

		result, err := e.mcp.CallTool(ctx, gatewaySubagentsTool, map[string]any{
			"action":         "wait",
			"taskIds":        remaining,
			"timeoutSeconds": pollSeconds,
		})
		if err != nil {
			return nil, fmt.Errorf("%s: calling %s: %w", WaitToolName, gatewaySubagentsTool, err)
		}
		if result.IsError {
			return nil, fmt.Errorf("%s: %s: %s", WaitToolName, gatewaySubagentsTool, joinContent(result))
		}

		statuses := parseSubagentsStatuses(joinContent(result), remaining)
		var stillPending []string
		for _, id := range remaining {
			st, ok := statuses[id]
			if !ok || !st.Finished {
				stillPending = append(stillPending, id)
				continue
			}
			results[id] = st
		}
		remaining = stillPending
	}

	for _, id := range remaining {
		results[id] = subagentStatus{Status: "timeout", Error: fmt.Sprintf("did not finish within %ds", timeoutSeconds)}
	}

	for id, st := range results {
		if st.Error != "" {
			continue
		}
		rec, ok := e.registry.get(id)
		if !ok {
			st.Error = fmt.Sprintf("run %q finished but this process has no session-key record for it (likely spawned by an earlier, now-restarted process); cannot fetch its result via sessions_history", id)
			results[id] = st
			continue
		}
		text, err := e.fetchFinalText(ctx, rec.sessionKey)
		if err != nil {
			st.Error = err.Error()
		} else {
			st.Result = text
		}
		results[id] = st
	}
	return results, nil
}

// fetchFinalText reads the finished child's history and returns its last
// assistant message, truncated to resultMaxBytes with a visible marker.
func (e *Executor) fetchFinalText(ctx context.Context, sessionKey string) (string, error) {
	result, err := e.mcp.CallTool(ctx, gatewaySessionsHistoryTool, map[string]any{
		"sessionKey": sessionKey,
		"limit":      historyFetchLimit,
	})
	if err != nil {
		return "", fmt.Errorf("fetching result via %s: %w", gatewaySessionsHistoryTool, err)
	}
	if result.IsError {
		return "", fmt.Errorf("%s: %s", gatewaySessionsHistoryTool, joinContent(result))
	}
	return truncateResult(extractLastAssistantText(joinContent(result)), resultMaxBytes), nil
}

// extractLastAssistantText pulls the last assistant message's text out of a
// sessions_history result. The gateway's exact history-entry shape is not
// documented to this package beyond "sanitized visible-session history"
// prose, so this tries a structured read first (a JSON array of entries
// carrying role/text/content fields) and falls back to the raw text verbatim
// when that doesn't decode — a transcript-formatted response is still useful
// to the model even unparsed, and this way a child's report is never
// silently dropped. Needs a live smoke to confirm whether the structured path
// ever actually fires (see package doc).
func extractLastAssistantText(raw string) string {
	var entries []struct {
		Role    string `json:"role"`
		Text    string `json:"text"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal([]byte(raw), &entries); err != nil || len(entries) == 0 {
		return raw
	}
	for i := len(entries) - 1; i >= 0; i-- {
		if entries[i].Role != "assistant" {
			continue
		}
		if entries[i].Text != "" {
			return entries[i].Text
		}
		if entries[i].Content != "" {
			return entries[i].Content
		}
	}
	return raw
}

// truncateResult mirrors internal/tools.RunLoadSuperegoArchive's truncation
// wording, so every "this was cut" marker in kortex's tool_results reads the
// same way.
func truncateResult(s string, max int) string {
	cut, truncated := tools.TruncateUTF8([]byte(s), max)
	if !truncated {
		return s
	}
	return fmt.Sprintf("%s\n[truncated: showing first %d of %d bytes]", cut, len(cut), len(s))
}

// parseSubagentsStatuses defensively extracts a per-id status from one
// "subagents" action="wait"/"list" call's result text. See the package doc
// comment: the gateway's exact response shape for this action is not
// documented beyond its request schema, so this walks the decoded JSON
// looking for objects that carry a recognizable id, rather than assuming one
// fixed envelope.
func parseSubagentsStatuses(raw string, queriedIDs []string) map[string]subagentStatus {
	queried := make(map[string]bool, len(queriedIDs))
	for _, id := range queriedIDs {
		queried[id] = true
	}

	var decoded any
	if err := json.Unmarshal([]byte(raw), &decoded); err != nil {
		// Not JSON: nothing structured to extract per id. The caller's own
		// deadline/timeout handling covers eventually declaring these ids
		// unresolved rather than hanging on an unparseable response forever.
		return nil
	}

	statuses := make(map[string]subagentStatus)
	scanSubagentEntries(decoded, queried, statuses)
	return statuses
}

// scanSubagentEntries walks a decoded JSON value looking for objects that
// carry both an id-like field (taskId/id/runId/name) matching one of
// queried, and status-like fields. It does not assume a fixed envelope
// (a bare array vs. {"tasks":[...]} vs. {"items":[...]}) since that envelope
// is one of the unconfirmed parts of the response shape (see package doc);
// walking finds matching entries wherever they are nested.
func scanSubagentEntries(node any, queried map[string]bool, out map[string]subagentStatus) {
	switch v := node.(type) {
	case map[string]any:
		if id := firstStringField(v, "taskId", "id", "runId", "name"); id != "" && queried[id] {
			out[id] = subagentStatusFromFields(v)
		}
		for _, child := range v {
			scanSubagentEntries(child, queried, out)
		}
	case []any:
		for _, child := range v {
			scanSubagentEntries(child, queried, out)
		}
	}
}

func firstStringField(m map[string]any, keys ...string) string {
	for _, k := range keys {
		if s, ok := m[k].(string); ok && s != "" {
			return s
		}
	}
	return ""
}

// terminalStatusWords and nonTerminalStatusWords classify a status string by
// case-insensitive substring. The exact enum the gateway uses for this field
// is one of the unconfirmed parts of the response shape (see package doc);
// nonTerminalStatusWords is checked first so an ambiguous match (e.g. a
// status containing both "queued" and, coincidentally, "done") resolves to
// "not finished yet" rather than ending the wait early.
var terminalStatusWords = []string{"complet", "done", "finish", "fail", "error", "cancel", "timeout"}
var nonTerminalStatusWords = []string{"pending", "running", "queued", "active", "in_progress", "waiting"}

func subagentStatusFromFields(m map[string]any) subagentStatus {
	status := firstStringField(m, "status", "state")
	errMsg := firstStringField(m, "error", "errorMessage", "message")
	lower := strings.ToLower(status)

	for _, w := range nonTerminalStatusWords {
		if strings.Contains(lower, w) {
			return subagentStatus{Status: status}
		}
	}
	finished := false
	for _, w := range terminalStatusWords {
		if strings.Contains(lower, w) {
			finished = true
			break
		}
	}
	if errMsg != "" {
		// An explicit error field means the run is done, even when its status
		// word isn't one this package recognizes as terminal.
		finished = true
	}
	return subagentStatus{Status: status, Finished: finished, Error: errMsg}
}
