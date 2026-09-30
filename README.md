# kortex

A deterministic cognitive orchestrator for LLM agents, written in Go.

kortex sits between an agent harness (first target: the `claude` CLI driven by
OpenClaw) and the model, owning the canonical message history and running a
configurable set of *facets* — core voice, superego reviewer, emotional
evaluator, communication memory — over that single shared history. Each facet
sees the same conversation through its own system prompt; the orchestrator
enforces their permissions and arbitrates what actually gets sent.

## Status

Early development. Current milestone: **F2a — observing interception**: on
top of the F1 pass-through (byte-identical proxy to the real `claude`,
running in production as a canary), kortex now parses every protocol line,
reconstructs the canonical session history in memory, and can log a
structured per-turn view — still with zero behavior change; interception is
read-only and any internal error in the new layers is swallowed, never
propagated to the stream.

## Architecture

Thin `main.go` (env/config wiring only) over three internal packages:

- `internal/proxy` — process plumbing: resolves the real `claude` binary
  (avoiding self-resolution, since kortex installs under the name `claude`),
  execs it with untouched argv/env, pumps stdio line-by-line byte-identical,
  forwards the auth file descriptor and SIGINT/SIGTERM, propagates the exit
  code, and optionally logs raw wire traffic (`KORTEX_LOG`).
- `internal/protocol` — types and lenient parsing for the claude-cli
  stream-json wire protocol (both directions, per the F0 spec). Parsing
  never fails: unknown or malformed lines come back as an `unknown` event
  and pass through untouched.
- `internal/history` — reconstructs the canonical conversation history of
  the current session (turns of user text, assistant text, summarized tool
  calls) from the parsed events. Purely observational state; this is the
  structure the F2b facets will operate on.

`proxy` may import `protocol` and `history`; `history` may import
`protocol`; never the reverse. The facet orchestrator will plug into
`proxy.copyLines`, the single point where every protocol line passes. The
import direction is enforced by `go-arch-lint` (`.go-arch-lint.yml`, run via
`make lint`). All env reading happens in `main.go`; the packages receive
everything through parameters.

Env: `KORTEX_CLAUDE_BIN` (explicit path to the real claude), `KORTEX_LOG`
(debug log path; never written to protocol stdout/stderr — raw wire traffic
goes to the path itself, and the structured per-turn event log to
`<path>.events`, one JSON object per line with a compact history snapshot
at each terminal result).

## Design notes

- The Messages API is stateless; owning the message array is normal usage.
- Facets are configuration, not code: the orchestration mechanism is generic
  and reusable for any agent, not tied to one persona.
