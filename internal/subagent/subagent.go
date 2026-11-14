// Package subagent is the composition-root seam for kortex's native
// spawn/wait/list-subagent tools: it is the ONLY place in kortex that knows
// the OpenClaw gateway's sessions_spawn/subagents/sessions_history MCP tool
// names, their argument shapes, and the model-id allowlist policy, exactly
// like main's openclaw_terminal_tool.go is the only place that knows the
// gateway's sessions_yield semantics. internal/tools stays agnostic of all of
// it behind the generic tools.SubagentPort interface (ToolDefs/Execute) that
// Executor satisfies.
//
// LIVE-SMOKE STATUS (2026-11-14, real gateway): sessions_spawn returns
// {status:"accepted", childSessionKey, runId, mode, ...} (no taskId), which
// parseSpawnResult handles. subagents action="wait" only accepts the gateway's
// internal taskId, which spawn does not return, so waiting is done by polling
// action="list" keyed by runId (see wait.go). sessions_history returns a
// {messages:[{role, content:[{type:"text",text}]}]} envelope.
package subagent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/mcp"
)

// mcpCaller is the slice of the MCP client Executor needs: invoking a gateway
// tool by its bare (unprefixed) name. The concrete *mcp.Client satisfies it;
// the interface keeps Executor testable without a live HTTP server.
type mcpCaller interface {
	CallTool(ctx context.Context, name string, arguments map[string]any) (mcp.CallToolResult, error)
}

// Gateway MCP tool names Executor calls through mcpCaller — the real
// OpenClaw gateway tools (unprefixed, matching mcp.Tool.Name). Executor talks
// to the gateway directly, so tools.MCPToolPrefix (which only applies to the
// model-facing declaration one layer up) never enters here.
const (
	gatewaySpawnTool           = "sessions_spawn"
	gatewaySubagentsTool       = "subagents"
	gatewaySessionsHistoryTool = "sessions_history"
)

// HiddenGatewayTools lists the raw gateway MCP tool names the core model must
// never see directly: Executor's three native tools (SpawnToolName,
// WaitToolName, ListToolName) replace them. agents_wait is hidden too even
// though Executor never calls it: it is the wait tool for collect=true
// children, a mode spawn_subagent never requests, and leaving it declared
// would let the model bypass spawn_subagent's preamble injection and model
// allowlist by spawning through some other still-visible path that also
// surfaces run ids agents_wait expects. The composition root passes this
// slice to tools.Dispatcher.HideMCPTools.
func HiddenGatewayTools() []string {
	return []string{gatewaySpawnTool, "agents_wait", gatewaySubagentsTool}
}

// Kortex's own native tool names, declared in ToolDefs() and matched by
// tools.Dispatcher.Execute to route here.
const (
	SpawnToolName = "spawn_subagent"
	WaitToolName  = "wait_subagents"
	ListToolName  = "list_subagents"
)

// Executor implements tools.SubagentPort: it declares and executes
// spawn_subagent, wait_subagents, and list_subagents by wrapping the
// gateway's sessions_spawn/subagents/sessions_history MCP tools.
type Executor struct {
	mcp      mcpCaller
	registry *registry

	// listPollInterval, notFoundPolls and maxListFailures tune waitForAll
	// (see the default* constants in wait.go); tests shrink listPollInterval.
	listPollInterval time.Duration
	notFoundPolls    int
	maxListFailures  int
}

// NewExecutor builds an Executor that calls the gateway through mcpClient.
func NewExecutor(mcpClient mcpCaller) *Executor {
	return &Executor{
		mcp:              mcpClient,
		registry:         newRegistry(),
		listPollInterval: defaultListPollInterval,
		notFoundPolls:    defaultNotFoundPolls,
		maxListFailures:  defaultMaxListFailures,
	}
}

// ToolDefs returns the three native tool declarations for the Messages API
// "tools" field, in the order the model should see them: spawn, wait, list.
func (e *Executor) ToolDefs() []anthropic.ToolDef {
	return []anthropic.ToolDef{spawnDef, waitDef, listDef}
}

// Execute routes a tool_use by name to the matching implementation. The
// caller (tools.Dispatcher) only reaches here after matching block.Name
// against ToolDefs(), so any other name reaching here is a dispatcher bug,
// reported as a clear error rather than a panic.
func (e *Executor) Execute(ctx context.Context, name string, input json.RawMessage) (output string, isError bool) {
	switch name {
	case SpawnToolName:
		return e.runSpawn(ctx, input)
	case WaitToolName:
		return e.runWait(ctx, input)
	case ListToolName:
		return e.runList(ctx, input)
	default:
		return fmt.Sprintf("error: subagent executor given unknown tool %q", name), true
	}
}

// mustAnthropicToolDef builds an anthropic.ToolDef from a Go value input
// schema, panicking on a marshal failure — schemas here are small static
// literals, so a failure can only be a programmer error caught immediately at
// package init. Mirrors internal/tools.mustToolDef; duplicated rather than
// imported because that helper is unexported and this is the only other
// caller (see u-keep-incidental-duplication-inline).
func mustAnthropicToolDef(name string, description string, inputSchema map[string]any) anthropic.ToolDef {
	raw, err := json.Marshal(inputSchema)
	if err != nil {
		panic(fmt.Sprintf("subagent: building schema for %s: %v", name, err))
	}
	return anthropic.ToolDef{Name: name, Description: description, InputSchema: raw}
}

// joinContent flattens an mcp.CallToolResult's content blocks into one
// string, mirroring internal/tools.MCPToolResultBlock's joining rule.
func joinContent(result mcp.CallToolResult) string {
	texts := make([]string, len(result.Content))
	for i, block := range result.Content {
		texts[i] = block.Text
	}
	return strings.Join(texts, "\n")
}

// firstNonEmpty returns the first non-empty string in values, or "".
func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}
