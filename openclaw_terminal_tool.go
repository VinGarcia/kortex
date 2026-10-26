package main

import (
	"encoding/json"
	"strings"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/tools"
)

// This file is the composition root's OpenClaw-semantics seam: it is the ONLY
// place in kortex that knows the gateway's sessions_yield tool name and its
// yielded-result shape. internal/tools.RunLoop is a generic tool-use loop (see
// internal/tools's README) that has no business knowing any specific tool's
// name or result semantics; it instead accepts an injected
// tools.LoopRequest.IsTerminalTool predicate, and isTerminalTool below is the
// concrete function main wires into every session.Config it builds (see
// runNative) to supply that predicate. Moving this logic out of internal/tools
// and into main closes architecture-review finding #2 of issue #5431 (a
// layer-purity violation: the generic loop hardcoded an OpenClaw-specific tool
// name).

// sessionsYieldTool is the bare (unprefixed) name of the gateway's yield tool.
// The model calls it as the MCP tool mcp__openclaw__sessions_yield, but the
// detection strips tools.MCPToolPrefix first so it matches whether or not the
// prefix is present (see isSessionsYield).
const sessionsYieldTool = "sessions_yield"

// isSessionsYield reports whether a tool_use name is the gateway's
// sessions_yield tool, matching on the bare name with tools.MCPToolPrefix
// stripped so it holds for the prefixed MCP form the model actually calls
// (mcp__openclaw__sessions_yield) and for a bare name alike.
func isSessionsYield(name string) bool {
	return strings.TrimPrefix(name, tools.MCPToolPrefix) == sessionsYieldTool
}

// isYieldedResult reports whether a sessions_yield tool_result indicates the
// turn actually yielded. The gateway answers a yield with a JSON object
// carrying {"status":"yielded"} (optionally alongside an "acknowledgment"
// field), so the check parses that object and matches status. A tool-level
// error, or any payload that does not decode to status "yielded", returns
// false so the caller falls back to ordinary tool-loop handling rather than
// ending the turn on a failure.
func isYieldedResult(result anthropic.ToolResultBlock) bool {
	if result.IsError {
		return false
	}
	var payload struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal([]byte(result.Content), &payload); err != nil {
		return false
	}
	return payload.Status == "yielded"
}

// isTerminalTool is the concrete tools.LoopRequest.IsTerminalTool /
// session.Config.IsTerminalTool predicate the composition root injects into
// every native session (see runNative). It recognizes exactly one case today —
// a sessions_yield call whose result reports {"status":"yielded"} — matching
// the behavior internal/tools.RunLoop hardcoded before this seam existed.
func isTerminalTool(name string, result anthropic.ToolResultBlock) bool {
	return isSessionsYield(name) && isYieldedResult(result)
}
