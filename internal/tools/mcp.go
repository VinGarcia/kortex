package tools

import (
	"strings"

	"github.com/vingarcia/kortex/internal/anthropic"
	"github.com/vingarcia/kortex/internal/mcp"
)

// MCPToolPrefix is the model-side naming convention for a tool reached
// through the MCP bridge (F3c): the gateway's tools/list catalog returns
// names WITHOUT this prefix (e.g. "sessions_send"), and prefixing is this
// package's job when exposing the tool to the Messages API — see
// sylphie/memory/design-prefrontal-redesign.md §F3c(b).3, confirmed by the
// LIVE HANDSHAKE DUMP.
const MCPToolPrefix = "mcp__openclaw__"

// MCPToolDef maps one mcp.Tool from the gateway's tools/list catalog to the
// anthropic.ToolDef shape the Messages API "tools" field expects, applying
// MCPToolPrefix to the name. InputSchema is copied as-is: mcp.Tool decodes
// it from the wire's "inputSchema" key and anthropic.ToolDef re-encodes the
// identical JSON Schema value under "input_schema" — only the envelope key
// differs, not the schema document itself.
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

// MCPToolResultBlock maps the result of an mcp.Client.CallTool call to the
// anthropic.ToolResultBlock keyed to toolUseID. A CallToolResult can carry
// multiple content blocks (none observed live as of §F3c, but the MCP
// protocol allows it); their Text is joined with newlines into the single
// string ToolResultBlock.Content expects. IsError is forwarded as-is: the
// gateway's own tools/call result already distinguishes a tool-level
// failure (isError:true, still HTTP 200) from a transport/JSON-RPC error,
// which mcp.Client.CallTool surfaces separately as a Go error, never inside
// CallToolResult.
func MCPToolResultBlock(toolUseID string, result mcp.CallToolResult) anthropic.ToolResultBlock {
	texts := make([]string, len(result.Content))
	for i, block := range result.Content {
		texts[i] = block.Text
	}
	return anthropic.NewToolResultBlock(toolUseID, strings.Join(texts, "\n"), result.IsError)
}
