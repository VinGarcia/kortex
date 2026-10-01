// Package mcp implements the kortex side of the MCP (Streamable-HTTP)
// client transport against the OpenClaw loopback MCP server: parsing the
// --mcp-config file the gateway writes for the subprocess-CLI, performing
// the initialize / notifications/initialized / tools/list handshake, and
// returning the typed tool catalog the server reports. tools/call (the
// actual tool invocation) and the GET /mcp server->client SSE channel are
// out of scope for this slice — see
// sylphie/memory/design-prefrontal-redesign.md §F3c(d) for the full
// slice-1..4 plan this package starts.
//
// Caveat (slice-1, by design): JSON-RPC responses are only decoded when
// Content-Type is application/json. The live gateway observed in §F3c's
// LIVE HANDSHAKE DUMP always answered single-response calls that way, never
// as an SSE stream, so a text/event-stream response is rejected with a
// clear error rather than parsed. Handling that shape is deferred until a
// real caller exercises it.
package mcp

import (
	"encoding/json"
	"fmt"
	"os"
)

// fileConfig is the shape of the --mcp-config file the gateway passes to
// the subprocess-CLI (kortex), keyed by server name under "mcpServers".
// Only the "openclaw" entry is consumed today; other entries in the map
// (e.g. google-calendar, per the design doc) are plugged in the same way
// but are not this client's concern.
type fileConfig struct {
	MCPServers map[string]ServerConfig `json:"mcpServers"`
}

// ServerConfig is one server entry of the --mcp-config file.
type ServerConfig struct {
	Type       string            `json:"type"`
	URL        string            `json:"url"`
	AlwaysLoad bool              `json:"alwaysLoad"`
	Headers    map[string]string `json:"headers"`
}

// openclawServerName is the key under "mcpServers" the gateway writes for
// the loopback server kortex talks to (see §F3c(a) createMcpServerConfig).
const openclawServerName = "openclaw"

// LoadServerConfig reads the --mcp-config file at path and returns the
// "openclaw" server entry, with every "${VAR}" placeholder in its header
// values expanded via getenv. The gateway writes header values as literal
// placeholders — e.g. "Authorization": "Bearer ${OPENCLAW_MCP_TOKEN}" and
// "x-openclaw-cli-capture-key": "${OPENCLAW_MCP_CLI_CAPTURE_KEY}" (§F3c LIVE
// HANDSHAKE DUMP) — and expects the client to resolve them before the first
// request goes out; sending the literal placeholder is rejected by the
// server the same way a missing header is.
//
// This package never reads the process environment itself (every
// os.Getenv lives in the composition root, main.go, per README
// §Architecture): getenv is supplied by the caller — main.go passes
// os.Getenv, tests pass a fake lookup over a map.
func LoadServerConfig(path string, getenv func(string) string) (ServerConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ServerConfig{}, fmt.Errorf("mcp: reading config %s: %w", path, err)
	}

	var file fileConfig
	if err := json.Unmarshal(data, &file); err != nil {
		return ServerConfig{}, fmt.Errorf("mcp: decoding config %s: %w", path, err)
	}

	server, ok := file.MCPServers[openclawServerName]
	if !ok {
		return ServerConfig{}, fmt.Errorf("mcp: config %s has no mcpServers.%s entry", path, openclawServerName)
	}
	if server.URL == "" {
		return ServerConfig{}, fmt.Errorf("mcp: config %s mcpServers.%s has no url", path, openclawServerName)
	}

	expanded := make(map[string]string, len(server.Headers))
	for name, value := range server.Headers {
		// os.Expand already implements general "${NAME}"/"$NAME" ->
		// mapping(NAME) substitution; getenv is that mapping function, so no
		// custom placeholder parsing is needed here. A variable absent from
		// the environment expands to "" rather than erroring — the server's
		// own auth check is what turns that into a surfaced failure (a 401
		// on the following request), matching how the live gateway behaves
		// on a bad/missing grant.
		expanded[name] = os.Expand(value, getenv)
	}
	server.Headers = expanded

	return server, nil
}
