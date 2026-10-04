# kortex

A deterministic cognitive orchestrator for LLM agents, written in Go.

kortex sits between an agent harness (first target: the `claude` CLI driven by
OpenClaw) and the model, owning the canonical message history and running a
configurable set of *facets* — core voice, superego reviewer, emotional
evaluator, communication memory — over that single shared history. Each facet
sees the same conversation through its own system prompt; the orchestrator
enforces their permissions and arbitrates what actually gets sent.

## Status

Early development. Current milestone: **F2b — first facet (input emotional
annotator), off by default**: on top of the F2a observing interception
(parse every line, reconstruct canonical history, structured logging),
kortex can now run its first facet — a model-backed annotator that tags
each paragraph of a real user message with emotional valence/investment
before it reaches the core model. Facets only exist when the `KORTEX_CONFIG`
env points at a config file; without it kortex remains the byte-identical
pass-through proxy (F1 behavior, running in production as a canary).
Facets are strictly fail-open: any evaluator failure (API error, timeout,
malformed output) forwards the original line untouched.

## Architecture

Thin `main.go` (env/config wiring and facet construction — the composition
root; all env reading happens here) over the internal packages. `main`
selects one of two peer code paths at startup: the default pass-through
**proxy** (execs the real `claude`), or the **native** producer when
`KORTEX_STREAM_EMITTER` is set (kortex speaks the stream-json protocol
itself). Both build the same facets over the same shared packages
(`protocol`, `history`, `config`, `anthropic`, `facet`); only the entry
package differs — `proxy` versus `session`.

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
  structure later facets will operate on.
- `internal/config` — loads the facet configuration file (JSON, pointed at
  by `KORTEX_CONFIG`). Declares which facets are enabled, their model, the
  env var carrying the OAuth token (the binary reads the env; the token
  value never lives in the file), and the facet prompt/memory paths.
- `internal/anthropic` — minimal Messages API client with OAuth Bearer
  auth (subscription setup-token flow): one non-streaming call with
  system + messages, timeouts, typed API errors.
- `internal/facet` — the facets. `InputAnnotator` implements the proxy's
  to-backend interception point: for each real user prompt it calls the
  evaluator model and splices per-paragraph emotion annotations into the
  text (the contract ported from sylphie's `emotion-annotate.sh`).
  Handshakes, control messages, replays and tool-result carriers are never
  touched, and every failure path is fail-open (original line forwarded,
  error recorded in `<KORTEX_LOG>.facets`). `OutputEvaluator` runs the
  emotion evaluator over each completed assistant turn (asynchronously,
  observability mode: the stream is untouched) and computes the
  hypothetical gate decision; on a gate hit it triggers `Superego`, which
  reviews the turn against the whole canonical history in shadow mode —
  the critique goes only to the facet log and in-memory metadata. In the
  superego's active mode, `ActiveSuperego` replaces that async shadow
  review with a blocking governor that satisfies the native path's
  `session.OutputGovernor`: it gates the core's draft (output evaluator →
  deterministic gate) and, on a gate hit, runs the superego ladder
  (`ReviewDraft`, up to `DefaultLadderRounds` rounds) — approve delivers
  the draft, `hold_ask_human` delivers a hold-and-ask message instead,
  `revise` feeds the critique back to the core for a redraft. The rejected
  drafts and critiques live only in an ephemeral message tail, never
  persisted to canonical history and never emitted on the stream.
- `internal/session` — the native producer path, entered from
  `main.runNative` when `KORTEX_STREAM_EMITTER` is set. Instead of execing
  the real `claude`, `session.Run` reads the NDJSON stream-json protocol
  from stdin, answers the initialize handshake, drives the tool-loop
  (`internal/tools`) per user turn over the session's accumulated history,
  emits stream-json on stdout, and persists history for cross-process
  resume. It builds the same facets as the proxy path (`buildFacets`) and
  defers the same post-run `WaitInFlight` drain. Its facet seams mirror the
  proxy's: `session` defines its own `Annotator`, `TurnEvaluator` and
  `OutputGovernor` ports, satisfied structurally by `facet` (never the
  reverse), wired by `main`. On the active path the final turn is deferred
  (`tools` RunLoop `DeferFinal`) so the governor decides the delivered text
  before anything reaches the wire, then re-emitted via
  `StreamEmitter.EmitFinalTurn`.

Import directions (enforced by `go-arch-lint`, `.go-arch-lint.yml`, run via
`make lint`): `proxy` may import `protocol` and `history`; `history` may
import `protocol`; `facet` may import `protocol`, `anthropic` and `history`
(read-only consumption of the canonical history: the proxy hands facets a
`history.Snapshot` per completed turn via `proxy.TurnEvaluator`); never the
reverse. `proxy` never imports `facet` — facets reach the pump only through
the `proxy.Interceptor`/`proxy.TurnEvaluator` interfaces, wired by `main`.
On the native path `session` may import `protocol`, `tools`, `anthropic`
and `history`; `tools` may import `anthropic`, `mcp` and `protocol`; never
the reverse. Like the proxy, `session` never imports `facet` — the input
annotator, output evaluator and active governor reach it only through the
`session.Annotator`, `session.TurnEvaluator` and `session.OutputGovernor`
ports `session` defines, wired by `main`. Wire-format knowledge
stays in `protocol` (facets rewrite message text via
`protocol.RewriteUserText`, never raw envelope bytes). The packages receive
everything (paths, tokens, timeouts) through parameters — `main` translates
`config` into plain values, so only the composition root knows the file
format.

Env: `KORTEX_CLAUDE_BIN` (explicit path to the real claude), `KORTEX_LOG`
(debug log path; never written to protocol stdout/stderr — raw wire traffic
goes to the path itself, the structured per-turn event log to
`<path>.events`, and the structured facet-call log — latency, model,
success/error, never the token — to `<path>.facets`), `KORTEX_CONFIG`
(facet config file; absent = no facets, pure passthrough), plus the env var
the config names as the facet OAuth token source.

## Design notes

- The Messages API is stateless; owning the message array is normal usage.
- Facets are configuration, not code: the orchestration mechanism is generic
  and reusable for any agent, not tied to one persona.
