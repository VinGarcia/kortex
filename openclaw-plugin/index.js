// Registers the "kortex-cli" backend: an exact clone of the bundled claude-cli
// backend that spawns the `kortex` pass-through binary (resolved via the
// gateway's PATH) instead of `claude`. Everything else — auth FD transport,
// stream-json parsing, session handling, MCP bundling — is reused from the
// Anthropic plugin, so behavior stays identical while kortex sits in the
// middle of the stdio stream.
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(import.meta.url);

// Facet activation for the canary. The config path is the kortex feature
// flag; the OAuth token is read from the secrets file at each spawn (never
// stored in this source) and handed to the child via the env var the config
// names in tokenEnv. Missing/unreadable files degrade to pure passthrough.
const KORTEX_CONFIG_PATH =
  "/home/vingarcia/projects/sylphie/kortex/canary-config.json";
const KORTEX_LOG_PATH = "/home/vingarcia/.openclaw/kortex/kortex.log";
const KORTEX_TOKEN_PATH =
  "/home/vingarcia/.openclaw/secrets/kortex-oauth-token";

function kortexEnv() {
  try {
    const token = readFileSync(KORTEX_TOKEN_PATH, "utf8").trim();
    if (!token) return {};
    return {
      KORTEX_CONFIG: KORTEX_CONFIG_PATH,
      KORTEX_LOG: KORTEX_LOG_PATH,
      CODECOMPANION_OAUTH_TOKEN: token,
    };
  } catch {
    return {};
  }
}

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
    const basePrepare = base.prepareExecution;
    api.registerCliBackend({
      ...base,
      id: "kortex-cli",
      // Drop the bundled claude-code artifact so command resolution cannot
      // shortcut back to the real claude binary.
      runtimeArtifact: undefined,
      config: {
        ...base.config,
        command: "kortex",
        // The bundled backend omits this and relies on providerId ===
        // "claude-cli" inside isClaudeStreamJsonDialect; without it the
        // gateway treats our stdout as plain text and chats receive raw JSON.
        jsonlDialect: "claude-stream-json",
      },
      prepareExecution: (context) => {
        const merge = (prepared) => ({
          ...prepared,
          env: { ...prepared.env, ...kortexEnv() },
        });
        const prepared = basePrepare(context);
        return prepared && typeof prepared.then === "function"
          ? prepared.then(merge)
          : merge(prepared);
      },
    });
  },
});
