// Registers the "kortex-cli" backend: an exact clone of the bundled claude-cli
// backend that spawns the `kortex` pass-through binary (resolved via the
// gateway's PATH) instead of `claude`. Everything else — auth FD transport,
// stream-json parsing, session handling, MCP bundling — is reused from the
// Anthropic plugin, so behavior stays identical while kortex sits in the
// middle of the stdio stream.
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

// Stable re-export inside the installed openclaw package, resolved relative to
// the openclaw package so nvm/version moves keep working. Loaded with
// require(esm) (Node >= 22, TLA-free graph) because the plugin loader needs
// this module — and register() — to stay synchronous.
const sdkEntry = require.resolve("openclaw/plugin-sdk/plugin-entry");
const backendPath = sdkEntry.replace(
  /dist[\/\\]plugin-sdk[\/\\]plugin-entry\.js$/,
  "dist/extensions/anthropic/cli-backend.js",
);
const { buildAnthropicCliBackend } = require(backendPath);

export default definePluginEntry({
  id: "kortex-cli",
  name: "Kortex CLI backend",
  description: "Claude CLI backend clone that launches the kortex pass-through binary",
  register(api) {
    const base = buildAnthropicCliBackend();
    api.registerCliBackend({
      ...base,
      id: "kortex-cli",
      // Drop the bundled claude-code artifact so command resolution cannot
      // shortcut back to the real claude binary.
      runtimeArtifact: undefined,
      config: {
        ...base.config,
        command: "kortex",
      },
    });
  },
});
