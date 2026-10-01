package tools

import (
	"strings"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/mcp"
)

// MCPToolPrefix is prepended to each gateway tool name: the gateway's
// tools/list catalog omits this prefix (e.g. "sessions_send"), so we add it
// when exposing the tool to the Messages API.
const MCPToolPrefix = "mcp__openclaw__"

// MCPToolDef maps one mcp.Tool to the anthropic.ToolDef the Messages API
// "tools" field expects, applying MCPToolPrefix to the name. InputSchema is
// copied as-is — only the JSON envelope key differs ("inputSchema" vs
// "input_schema"), not the schema document.
func MCPToolDef(tool mcp.Tool) anthropic.ToolDef {
	return anthropic.ToolDef{
		Name:        MCPToolPrefix + tool.Name,
		Description: tool.Description,
		InputSchema: tool.InputSchema,
	}
}

// MCPToolDefs maps a whole tools/list catalog, preserving order, for
// declaring alongside Definitions() in a request's "tools" field.
func MCPToolDefs(tools []mcp.Tool) []anthropic.ToolDef {
	defs := make([]anthropic.ToolDef, len(tools))
	for i, tool := range tools {
		defs[i] = MCPToolDef(tool)
	}
	return defs
}

// MCPToolResultBlock maps an mcp.CallToolResult to the
// anthropic.ToolResultBlock keyed to toolUseID. Multiple content blocks are
// joined with newlines into the single string ToolResultBlock.Content
// expects. IsError is forwarded as-is (a tool-level failure, distinct from a
// transport error, which CallTool returns as a Go error instead).
func MCPToolResultBlock(toolUseID string, result mcp.CallToolResult) anthropic.ToolResultBlock {
	texts := make([]string, len(result.Content))
	for i, block := range result.Content {
		texts[i] = block.Text
	}
	return anthropic.NewToolResultBlock(toolUseID, strings.Join(texts, "\n"), result.IsError)
}
