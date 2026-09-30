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

## Design notes

- The Messages API is stateless; owning the message array is normal usage.
- Facets are configuration, not code: the orchestration mechanism is generic
  and reusable for any agent, not tied to one persona.
