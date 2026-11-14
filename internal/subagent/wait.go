package subagent

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
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

// defaultNotFoundPolls is how many consecutive polls may omit a NEVER-SEEN
// run id from both active[] and recent[] before waitForAll gives up on that
// id as "not found" instead of waiting out the full timeout (e.g. a mistyped
// id). It applies together with defaultNotFoundGrace: both must be exceeded.
const defaultNotFoundPolls = 2

// defaultNotFoundGrace is the minimum time since the wait started before a
// never-seen id may be declared not found. Right after sessions_spawn the
// child may not be listed yet, and a couple of 3s polls are too short a
// window for that.
const defaultNotFoundGrace = 15 * time.Second

// defaultMaxListFailures is how many consecutive failed list polls (call
// error, tool error, or undecodable response) waitForAll tolerates: the next
// failure after that (the N+1th in a row) makes it give up and return
// whatever it already gathered.
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
	"Block until every listed subagent run finishes (or the timeout elapses), then return each run's final report. If ids is omitted or empty, waits for ALL of this session's subagents that are not yet finished (returns an empty list immediately if none are pending).",
	map[string]any{
		"type": "object",
		"properties": map[string]any{
			"ids":             map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "The run ids returned by spawn_subagent with wait=false. Omit or pass [] to wait for every not-yet-finished subagent of this session."},
			"timeout_seconds": map[string]any{"type": "integer", "description": "How long to block before giving up on whatever hasn't finished; defaults to 1800 (30 minutes)."},
		},
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
	timeout := input.TimeoutSeconds
	if timeout <= 0 {
		timeout = defaultWaitTimeoutSeconds
	}

	ids := input.IDs
	if len(ids) == 0 {
		pending, err := e.pendingIDs(ctx)
		if err != nil {
			return fmt.Sprintf("error: %v", err), true
		}
		if len(pending) == 0 {
			return "[]", false
		}
		ids = pending
	}

	statuses, waitErr := e.waitForAll(ctx, ids, timeout)
	if waitErr != nil && statuses == nil {
		return fmt.Sprintf("error: %v", waitErr), true
	}

	entries := make([]waitResultEntry, 0, len(ids))
	for _, id := range ids {
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

// pendingIDs returns, sorted, the registry's run ids that the gateway's list
// reports as present but not finished, plus those absent from the list that
// were spawned within e.notFoundGrace (the gateway may not list a fresh child
// yet; this is the "spawn wait=false, then wait with no ids" flow): the "all
// pending children" default for wait_subagents with no ids. Older registry ids
// absent from the list are skipped (finished long ago and pruned), so they
// never turn a no-op wait into a not_found round trip.
func (e *Executor) pendingIDs(ctx context.Context) ([]string, error) {
	known := e.registry.all()
	if len(known) == 0 {
		return nil, nil
	}
	entries, err := e.pollList(ctx)
	if err != nil {
		return nil, fmt.Errorf("%s: listing pending subagents: %w", WaitToolName, err)
	}
	var pending []string
	for id, rec := range known {
		entry, listed := entries[id]
		if listed && !entry.finished() || !listed && time.Since(rec.spawnedAt) < e.notFoundGrace {
			pending = append(pending, id)
		}
	}
	sort.Strings(pending)
	return pending, nil
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
// (nil, err). One last poll runs after the deadline, so a run that finished
// during the final sleep is not misreported as a timeout.
//
// Missing ids: an id NEVER seen in a list is reported "not_found" once it
// has been absent for e.notFoundPolls consecutive polls AND e.notFoundGrace
// has passed since the wait started (the child may not be listed right after
// spawn). An id that WAS seen and later drops out of the lists (gateway
// restart, list pruning) is NOT declared not found: waitForAll keeps waiting
// for it until the timeout (whose message says it vanished), because it may
// reappear and its report is only fetchable once it is seen finished.
//
// Failures: e.maxListFailures consecutive failed polls are tolerated; one
// more makes waitForAll return the results gathered so far (finished ids
// with their reports, the rest marked unknown) TOGETHER with the error.
// Shared by wait_subagents and spawn_subagent's wait=true path.
func (e *Executor) waitForAll(ctx context.Context, ids []string, timeoutSeconds int) (map[string]subagentStatus, error) {
	start := time.Now()
	deadline := start.Add(time.Duration(timeoutSeconds) * time.Second)
	remaining := append([]string(nil), ids...)
	results := make(map[string]subagentStatus, len(ids))
	seen := make(map[string]bool, len(ids))
	absent := make(map[string]int, len(ids))
	vanished := make(map[string]bool, len(ids))
	lastState := make(map[string]string, len(ids))
	failures := 0
	var giveUpErr error

	for len(remaining) > 0 {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("%s: %w", WaitToolName, err)
		}

		entries, err := e.pollList(ctx)
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return nil, fmt.Errorf("%s: %w", WaitToolName, ctxErr)
			}
			failures++
			if failures > e.maxListFailures {
				giveUpErr = fmt.Errorf("%s: giving up after %d consecutive list failures: %w", WaitToolName, failures, err)
				break
			}
		} else {
			failures = 0
			var stillPending []string
			for _, id := range remaining {
				entry, ok := entries[id]
				switch {
				case !ok && seen[id]:
					// Vanished after being listed: keep waiting (see doc).
					vanished[id] = true
				case !ok:
					absent[id]++
					if absent[id] >= e.notFoundPolls && time.Since(start) >= e.notFoundGrace {
						results[id] = subagentStatus{Status: "not_found", Error: e.notFoundMessage(id, absent[id])}
						continue
					}
				case entry.finished():
					seen[id] = true
					vanished[id] = false
					results[id] = subagentStatus{Status: entry.Status}
					continue
				default:
					seen[id] = true
					vanished[id] = false
					lastState[id] = entry.Execution.State
				}
				stillPending = append(stillPending, id)
			}
			remaining = stillPending
			if len(remaining) == 0 {
				break
			}
		}

		left := time.Until(deadline)
		if left <= 0 {
			break
		}
		sleep := e.listPollInterval
		if left < sleep {
			sleep = left
		}
		timer := time.NewTimer(sleep)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, fmt.Errorf("%s: %w", WaitToolName, ctx.Err())
		case <-timer.C:
		}
	}

	for _, id := range remaining {
		if giveUpErr != nil {
			results[id] = subagentStatus{Status: "unknown", Error: giveUpErr.Error()}
			continue
		}
		msg := fmt.Sprintf("did not finish within %ds", timeoutSeconds)
		switch {
		case seen[id] && lastState[id] != "":
			msg += fmt.Sprintf(" (last execution state: %q)", lastState[id])
		case seen[id]:
			msg += " (last seen with no execution state)"
		}
		if vanished[id] {
			msg += "; it was listed earlier but is no longer in the gateway's list"
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

// notFoundMessage words the error for a never-seen id, adding a hint when
// spawn_subagent took the id from a field other than runId (the list is keyed
// by runId, so such an id may simply not match).
func (e *Executor) notFoundMessage(id string, absentPolls int) string {
	msg := fmt.Sprintf("run %q not found in the %s list (never listed; absent from %d consecutive polls over %s)", id, gatewaySubagentsTool, absentPolls, e.notFoundGrace)
	if rec, ok := e.registry.get(id); ok && rec.idNotRunID {
		msg += "; this id came from the spawn result's taskId/id field rather than runId, so it may not match the list's runId"
	}
	return msg
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
