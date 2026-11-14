package subagent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// subagentPreamble is prefixed onto every spawned subagent's task, ahead of
// the caller's own text. It exists for two reasons specific to how this
// Executor manages a child: (1) sessions_yield is OpenClaw's native
// await-events primitive, meaningless for a one-shot completionTarget
// "parent" child that reports once and exits, so calling it would just hang
// the run; (2) a child's report lands in the tool_result Executor hands back
// to the core model, which is itself a prompt — a report containing a large
// pasted file or log can balloon that prompt, so the child is asked to write
// large output to the repo and report the path instead.
const subagentPreamble = "Never call sessions_yield. Keep your final report compact (<= 8 KiB); write large artifacts to files in the repo and return paths."

var spawnDef = mustAnthropicToolDef(
	SpawnToolName,
	"Spawn a background subagent to work a task independently, then either wait for its result or return immediately with a run id to check on later with wait_subagents/list_subagents.",
	map[string]any{
		"type": "object",
		"properties": map[string]any{
			"task":            map[string]any{"type": "string", "description": "The task to hand the subagent, in full — it starts with no context beyond this."},
			"model":           map[string]any{"type": "string", "description": "haiku, sonnet, or opus (short name, or a full model id containing one of them); omit for sonnet. fable is never allowed."},
			"label":           map[string]any{"type": "string", "description": "Short human-readable label for this run, shown in list_subagents."},
			"wait":            map[string]any{"type": "boolean", "description": "true (default): block until the subagent finishes and return its report directly. false: return immediately with a run id."},
			"timeout_seconds": map[string]any{"type": "integer", "description": "When wait is true, how long to block before giving up; defaults to 1800 (30 minutes)."},
		},
		"required": []string{"task"},
	},
)

type spawnInput struct {
	Task           string `json:"task"`
	Model          string `json:"model,omitempty"`
	Label          string `json:"label,omitempty"`
	Wait           *bool  `json:"wait,omitempty"`
	TimeoutSeconds int    `json:"timeout_seconds,omitempty"`
}

// spawnResult is the {runId, childSessionKey} JSON spawn_subagent returns
// when wait is false.
type spawnResult struct {
	RunID           string `json:"runId"`
	ChildSessionKey string `json:"childSessionKey"`
}

func (e *Executor) runSpawn(ctx context.Context, rawInput json.RawMessage) (output string, isError bool) {
	var input spawnInput
	if err := json.Unmarshal(rawInput, &input); err != nil {
		return fmt.Sprintf("error: decoding %s input: %v", SpawnToolName, err), true
	}
	if strings.TrimSpace(input.Task) == "" {
		return "error: task is required", true
	}
	model, err := resolveModel(input.Model)
	if err != nil {
		return fmt.Sprintf("error: %v", err), true
	}

	args := map[string]any{
		"task":             subagentPreamble + "\n\n" + input.Task,
		"context":          "isolated",
		"completionTarget": "parent",
		"cleanup":          "keep",
		"runtime":          "subagent",
		"model":            model,
	}
	if input.Label != "" {
		args["label"] = input.Label
	}

	result, err := e.mcp.CallTool(ctx, gatewaySpawnTool, args)
	if err != nil {
		return fmt.Sprintf("error: calling %s: %v", gatewaySpawnTool, err), true
	}
	raw := joinContent(result)
	if result.IsError {
		return fmt.Sprintf("error: %s: %s", gatewaySpawnTool, raw), true
	}

	runID, sessionKey, err := parseSpawnResult(raw)
	if err != nil {
		return fmt.Sprintf("error: %v", err), true
	}
	e.registry.put(runID, subagentRecord{sessionKey: sessionKey, label: input.Label})

	wait := true
	if input.Wait != nil {
		wait = *input.Wait
	}
	if !wait {
		out, err := json.Marshal(spawnResult{RunID: runID, ChildSessionKey: sessionKey})
		if err != nil {
			return fmt.Sprintf("error: encoding spawn result: %v", err), true
		}
		return string(out), false
	}

	timeout := input.TimeoutSeconds
	if timeout <= 0 {
		timeout = defaultWaitTimeoutSeconds
	}
	statuses, err := e.waitForAll(ctx, []string{runID}, timeout)
	if err != nil && statuses == nil {
		return fmt.Sprintf("error: %v", err), true
	}
	st := statuses[runID]
	if st.Error != "" {
		return fmt.Sprintf("error: subagent %s: %s", runID, st.Error), true
	}
	return st.Result, false
}

// parseSpawnResult defensively extracts the run id and session key from one
// sessions_spawn call's result text. See the package doc comment: the
// request shape is confirmed from the gateway's published schema, but this
// response shape is not, so several plausible field names are tried rather
// than one assumed.
func parseSpawnResult(raw string) (runID string, sessionKey string, err error) {
	var fields struct {
		TaskID          string `json:"taskId"`
		ID              string `json:"id"`
		RunID           string `json:"runId"`
		SessionKey      string `json:"sessionKey"`
		ChildSessionKey string `json:"childSessionKey"`
		SessionID       string `json:"sessionId"`
	}
	if err := json.Unmarshal([]byte(raw), &fields); err != nil {
		return "", "", fmt.Errorf("sessions_spawn returned a result spawn_subagent cannot parse (wanted a JSON object with a run/task id and a session key, got %.200q): %w — this needs a live smoke to confirm the real response shape (see package doc)", raw, err)
	}
	runID = firstNonEmpty(fields.TaskID, fields.RunID, fields.ID)
	sessionKey = firstNonEmpty(fields.ChildSessionKey, fields.SessionKey, fields.SessionID)
	if runID == "" {
		return "", "", fmt.Errorf("sessions_spawn result had no recognizable run/task id field (taskId/runId/id); raw=%.200q — this needs a live smoke to confirm the real response shape (see package doc)", raw)
	}
	if sessionKey == "" {
		return "", "", fmt.Errorf("sessions_spawn result had no recognizable session-key field (sessionKey/childSessionKey/sessionId); raw=%.200q — this needs a live smoke to confirm the real response shape (see package doc)", raw)
	}
	return runID, sessionKey, nil
}
