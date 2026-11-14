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

// Executor.listPollInterval (default defaultListPollInterval) is how long
// waitForAll sleeps between "subagents" action="list" polls.
//
// Why list-polling instead of the gateway's action="wait": LIVE SMOKE
// (2026-11-14) showed action="wait" only accepts the gateway's internal
// *taskId*, which sessions_spawn does NOT return (it returns runId +
// childSessionKey); waiting by runId or childSessionKey answers
// reason:"unavailable". action="list" does carry runId in its active[]/recent[]
// entries, so polling it keyed by runId needs no runId->taskId mapping.
const defaultListPollInterval = 3 * time.Second

// defaultNotFoundPolls is how many consecutive polls may omit a run id from
// both active[] and recent[] before waitForAll gives up on that id as "not
// found" instead of waiting out the full timeout (e.g. a mistyped id).
const defaultNotFoundPolls = 2

// defaultMaxListFailures is how many consecutive failed list polls (call
// error, tool error, or undecodable response) waitForAll tolerates before
// giving up and returning whatever it already gathered.
const defaultMaxListFailures = 3

// stateFinished is the gateway's execution.state value for a run that has
// reached a terminal state.
const stateFinished = "finished"

// listRecentMinutes is the recentMinutes window passed to action="list" so a
// run that finished long before wait_subagents was called is still reported
// (the gateway default window is short; 1440 was accepted live).
const listRecentMinutes = 1440

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

	statuses, waitErr := e.waitForAll(ctx, input.IDs, timeout)
	if waitErr != nil && statuses == nil {
		return fmt.Sprintf("error: %v", waitErr), true
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
	if waitErr != nil {
		// Partial results: the entries gathered before waitForAll gave up,
		// returned alongside the error so finished reports are not lost.
		return fmt.Sprintf("error: %v\npartial results: %s", waitErr, out), true
	}
	return string(out), false
}

// subagentStatus is what waitForAll learns about one run id: the gateway's
// own status label and — once resolved via sessions_history — the child's
// final report or a reason it could not be fetched.
type subagentStatus struct {
	Status string
	Error  string
	Result string
}

// listEntry is one element of the active[] or recent[] arrays of a
// subagents action="list" response, as observed live (2026-11-14). The
// response's tasks[] array is keyed by an unrelated gateway taskId and is
// deliberately not decoded.
type listEntry struct {
	RunID     string `json:"runId"`
	Status    string `json:"status"`
	Execution struct {
		State string `json:"state"`
	} `json:"execution"`
}

// finished reports whether the run reached a terminal state. Only
// execution.state == "finished" is terminal. Any other value — "running",
// any state this package has never seen, or a missing field — is treated as
// NOT finished: waiting on a run that is really done costs at most the
// caller's timeout (whose message names the last state seen), while
// reporting an unfinished run as done would fetch a partial transcript as
// its final report. Exact equality, never substring matching ("inactive"
// must not read as "active").
func (l listEntry) finished() bool {
	return l.Execution.State == stateFinished
}

type listResponse struct {
	Active []listEntry `json:"active"`
	Recent []listEntry `json:"recent"`
}

// parseList decodes one list response into entries keyed by runId. When a
// run appears in both active[] and recent[], the finished entry wins, so a
// run caught mid-transition is never reported as still running.
func parseList(raw string) (map[string]listEntry, error) {
	var resp listResponse
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		return nil, fmt.Errorf("decoding %s list response: %w", gatewaySubagentsTool, err)
	}
	entries := make(map[string]listEntry, len(resp.Active)+len(resp.Recent))
	for _, group := range [][]listEntry{resp.Active, resp.Recent} {
		for _, entry := range group {
			if entry.RunID == "" {
				continue
			}
			if prev, seen := entries[entry.RunID]; seen && prev.finished() && !entry.finished() {
				continue
			}
			entries[entry.RunID] = entry
		}
	}
	return entries, nil
}

// pollList makes one list call and decodes it.
func (e *Executor) pollList(ctx context.Context) (map[string]listEntry, error) {
	result, err := e.mcp.CallTool(ctx, gatewaySubagentsTool, map[string]any{
		"action":        "list",
		"recentMinutes": listRecentMinutes,
	})
	if err != nil {
		return nil, fmt.Errorf("calling %s: %w", gatewaySubagentsTool, err)
	}
	if result.IsError {
		return nil, fmt.Errorf("%s: %s", gatewaySubagentsTool, joinContent(result))
	}
	return parseList(joinContent(result))
}

// waitForAll polls the gateway's "subagents" action="list" tool for ids
// until every one is finished, found missing, or timeoutSeconds elapse, then
// fetches each finished id's final report via sessions_history. It is
// ctx-aware: a cancelled ctx stops the loop before the next poll and returns
// (nil, err).
//
// An id absent from both lists for e.notFoundPolls consecutive polls is
// reported with Status "not_found" rather than waited out. Up to
// e.maxListFailures consecutive failed polls are tolerated; past that,
// waitForAll returns the results gathered so far (finished ids with their
// reports, the rest marked unknown) TOGETHER with the error. Shared by
// wait_subagents and spawn_subagent's wait=true path.
func (e *Executor) waitForAll(ctx context.Context, ids []string, timeoutSeconds int) (map[string]subagentStatus, error) {
	deadline := time.Now().Add(time.Duration(timeoutSeconds) * time.Second)
	remaining := append([]string(nil), ids...)
	results := make(map[string]subagentStatus, len(ids))
	absent := make(map[string]int, len(ids))
	lastState := make(map[string]string, len(ids))
	failures := 0
	var giveUpErr error

	for len(remaining) > 0 {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("%s: %w", WaitToolName, err)
		}
		if time.Until(deadline) <= 0 {
			break
		}

		entries, err := e.pollList(ctx)
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return nil, fmt.Errorf("%s: %w", WaitToolName, ctxErr)
			}
			failures++
			if failures >= e.maxListFailures {
				giveUpErr = fmt.Errorf("%s: giving up after %d consecutive list failures: %w", WaitToolName, failures, err)
				break
			}
		} else {
			failures = 0
			var stillPending []string
			for _, id := range remaining {
				entry, ok := entries[id]
				switch {
				case !ok:
					absent[id]++
					if absent[id] >= e.notFoundPolls {
						results[id] = subagentStatus{Status: "not_found", Error: fmt.Sprintf("run %q not found in the %s list (absent from %d consecutive polls)", id, gatewaySubagentsTool, absent[id])}
						continue
					}
				case entry.finished():
					results[id] = subagentStatus{Status: entry.Status}
					continue
				default:
					absent[id] = 0
					lastState[id] = entry.Execution.State
				}
				stillPending = append(stillPending, id)
			}
			remaining = stillPending
			if len(remaining) == 0 {
				break
			}
		}

		sleep := e.listPollInterval
		if left := time.Until(deadline); left < sleep {
			sleep = left
		}
		if sleep > 0 {
			select {
			case <-ctx.Done():
				return nil, fmt.Errorf("%s: %w", WaitToolName, ctx.Err())
			case <-time.After(sleep):
			}
		}
	}

	for _, id := range remaining {
		if giveUpErr != nil {
			results[id] = subagentStatus{Status: "unknown", Error: giveUpErr.Error()}
			continue
		}
		msg := fmt.Sprintf("did not finish within %ds", timeoutSeconds)
		if state := lastState[id]; state != "" {
			msg += fmt.Sprintf(" (last execution state: %q)", state)
		}
		results[id] = subagentStatus{Status: "timeout", Error: msg}
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
	return results, giveUpErr
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
	// LIVE SMOKE (2026-11-14): sessions_history returns an envelope
	// {"messages":[{"role":"assistant","content":[{"type":"text","text":...}]}]},
	// where content is a block array (a plain string is also tolerated, as are
	// bare arrays and a top-level "text" field).
	type message struct {
		Role    string          `json:"role"`
		Text    string          `json:"text"`
		Content json.RawMessage `json:"content"`
	}
	var entries []message
	var envelope struct {
		Messages []message `json:"messages"`
	}
	if err := json.Unmarshal([]byte(raw), &envelope); err == nil && len(envelope.Messages) > 0 {
		entries = envelope.Messages
	} else if err := json.Unmarshal([]byte(raw), &entries); err != nil || len(entries) == 0 {
		return raw
	}
	for i := len(entries) - 1; i >= 0; i-- {
		if entries[i].Role != "assistant" {
			continue
		}
		if entries[i].Text != "" {
			return entries[i].Text
		}
		var asString string
		if json.Unmarshal(entries[i].Content, &asString) == nil && asString != "" {
			return asString
		}
		var blocks []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if json.Unmarshal(entries[i].Content, &blocks) == nil {
			var texts []string
			for _, b := range blocks {
				if b.Type == "text" && b.Text != "" {
					texts = append(texts, b.Text)
				}
			}
			if len(texts) > 0 {
				return strings.Join(texts, "\n")
			}
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
