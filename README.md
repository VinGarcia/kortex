# kortex

A deterministic cognitive orchestrator for LLM agents, written in Go.

kortex sits between an agent harness (first target: the `claude` CLI driven by
OpenClaw) and the model, owning the canonical message history and running a
configurable set of *facets* — core voice, superego reviewer, emotional
evaluator, communication memory — over that single shared history. Each facet
sees the same conversation through its own system prompt; the orchestrator
enforces their permissions and arbitrates what actually gets sent.

## Status

Early development. Current milestone: **F1 — pass-through skeleton**: a
drop-in binary that speaks the claude-cli stream-json protocol and proxies
untouched to the real `claude` binary, validating the protocol contract with
zero behavior change before any facet exists.

## Architecture

Thin `main.go` (env/config wiring only) over two internal packages:

- `internal/proxy` — process plumbing: resolves the real `claude` binary
  (avoiding self-resolution, since kortex installs under the name `claude`),
  execs it with untouched argv/env, pumps stdio line-by-line byte-identical,
  forwards the auth file descriptor and SIGINT/SIGTERM, propagates the exit
  code, and optionally logs raw wire traffic (`KORTEX_LOG`).
- `internal/protocol` — types/parsing for the claude-cli stream-json wire
  protocol. Minimal in F1; grows in F2 when kortex starts interpreting the
  stream. `proxy` may import `protocol`; never the reverse. The F2 facet
  orchestrator will plug into `proxy.copyLines`, the single point where every
  protocol line passes.

Env: `KORTEX_CLAUDE_BIN` (explicit path to the real claude), `KORTEX_LOG`
(debug traffic log path; never written to protocol stdout/stderr).

## Design notes

- The Messages API is stateless; owning the message array is normal usage.
- Facets are configuration, not code: the orchestration mechanism is generic
  and reusable for any agent, not tied to one persona.
