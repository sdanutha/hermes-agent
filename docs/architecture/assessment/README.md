# Hermes Agent — Architecture & Reduction Assessment

Analysis only: no production code was changed. Pinned to `db203baba5` (`dev`). Evidence tags:
**[V]** verified by reading code or running it, **[I]** inferred, **[U]** unknown.

| Document | Contents |
|---|---|
| [01-architecture-overview.md](01-architecture-overview.md) | Layers, surfaces, invariants, complexity hot spots |
| [02-component-inventory.md](02-component-inventory.md) | Phase 1 — every major component, entry points, init, deps |
| [03-capability-matrix.md](03-capability-matrix.md) | Phase 2 — 58 capabilities classified A–F |
| [04-dependency-map.md](04-dependency-map.md) | Phase 3 — graph, measured import facts, impact per candidate |
| [05-runtime-flows.md](05-runtime-flows.md) | Startup, init, turn, LLM call, tools, memory, skills, compression, gateway, cron, delegation, errors |
| [06-minimal-core.md](06-minimal-core.md) | Phase 4 — Minimum Viable Hermes, verified E2E |
| [07-reduction-candidates.md](07-reduction-candidates.md) | Phase 5 — scoring criteria, ranked candidates, KEEP list |
| [08-roadmap.md](08-roadmap.md) | Phases A–E with dependencies, validation, rollback |
| [09-evidence-coverage.md](09-evidence-coverage.md) | Verification log, claim ledger, doc drift, coverage, open questions |
| [verification/](verification/) | Re-runnable probes (registry, payload, import attribution, minimal E2E) |

Explorable diagrams (each node links to its source lines; automated archify checks passed, not
visually reviewed):
[architecture overview](diagrams/architecture-overview.html) ·
[one conversation turn](diagrams/agent-turn.html) ·
[proposed minimal core](diagrams/minimal-core.html).

## Executive summary

**1. Actual architecture.** One synchronous agent library (`run_agent.AIAgent`, 86 constructor
parameters, 14 mixins; phase pipeline in `agent/conversation_loop.py` + 33 `agent/turn_*.py`)
hosted in-process by five surfaces: the CLI, `tui_gateway` (serving TUI, Desktop and the web
dashboard), the messaging gateway (which also runs the cron ticker), the ACP adapter, and a legacy
single-query runner. Each turn alternates between an LLM call through a provider transport and a
tool round through a self-registering registry (88 tools, 59 toolsets), guarded by approval
checks and a single subprocess-env builder. Durable state is one SQLite file (`SessionDB`, schema
v32) that hosts pass in; the library itself persists nothing without it. Providers, web/image/video
backends, platforms, memory providers and dashboard auth are plugins. [V]

**2. Smallest practical core.** Already reachable by configuration, no code changes:
`AIAgent(provider="custom", enabled_toolsets=["file","terminal"], skip_memory=True,
skip_context_files=True)`. Verified end to end against a local fake OpenAI server: 8 tool schemas,
2 streamed calls, 1 `read_file` round, correct final answer, no database written. The safe-worker
mode (`tools/registry.py:126-134`) is a stricter in-code precedent. [V]

**3. Essential capabilities.** Config + scoped secrets; AIAgent init; turn loop with budgets,
retries and basic error recovery; byte-stable prompt builder; compression; one transport
(chat_completions) and one provider profile; model metadata; tool registry, toolsets and one
executor path; file, terminal and process tools on the local environment; approval guards,
`build_subprocess_env`, secret scope, redaction; logging. SessionDB is essential for hosts that
resume sessions, optional for the library.

**4. Optional capabilities.** Every host surface beyond a thin entry; non-chat transports and
codex app-server; credential pools, OAuth, Nous free tier, fallbacks; auxiliary routing beyond
compression; MoA; Tool Search; memory providers and built-in memory; skills and curator;
delegation; code execution; MCP; browser; web search; all media; computer use and connectors;
kanban; cron; LSP; pets; updater; opt-in telemetry; i18n; research runners; evals.

**5. Top 10 reduction opportunities** (ranked by the Net score defined in
[07](07-reduction-candidates.md)):
1. Late-import four core→host edges (`session_source→gateway`, `cronjob_tools→cron`,
   `local_env_policy→hermes_cli.auth`, `turn_context→lsp`) — SIMPLIFY, Net +6.
2. Remove legacy shims: `tool_delay`, empty `cron` extra, `providers/*.py` drop-ins,
   `hermes-agent` script — REMOVE after notice, +5.
3. Move research runners (`batch_runner`, `mini_swe_runner`, `trajectory_compressor`,
   `toolset_distributions`) out of the wheel — EXTRACT, +5.
4. Pull 73 `api_mode` branches in 15 files into `ProviderTransport` — CONSOLIDATE, +5.
5. Make plugin discovery lazy per kind (today `import run_agent` loads 42 provider, 11 web,
   8 image, 4 video, 4 dashboard-auth plugins; 1.31 s, 1,181 modules) — SIMPLIFY, +5.
6. Delete byterover/retaindb (past their 2026-10-15 removal date) once catalog entries exist —
   REMOVE, +3.
7. Skip the Tool Search bridge on tiny toolsets (it adds 3 tools to an 8-tool payload) —
   CONFIGURE, +3.
8. Pets to a catalog plugin — EXTRACT, +2.
9. google_meet, teams_pipeline, hermes-achievements to the catalog — EXTRACT, +2.
10. `hermes claw` (OpenClaw migration) — INVESTIGATE usage, then EXTRACT, +2.

**6. Biggest complexity sources.** Desktop TypeScript (690k lines incl. 267k tests); the messaging
gateway and adapters (191k Python lines, `GatewayRunner` with 19 mixins); product-specific logic
woven through the turn loop (MoA, Nous billing, Copilot, Ollama, kanban, codex app-server);
SessionDB (19 mixins, 254 methods, 1,264 commits in three months); the updater (20k lines,
906 commits); the config surface (908 keys, 549 `HERMES_*` names); provider resolution (≈ 38k lines).

**7. Low-risk removals.** `tool_delay`; the empty `cron` extra; `providers/*.py` drop-in
discovery; research runners out of the wheel; the `hermes-agent` script after one deprecation
release; stale doc lines (AGENTS.md toolset list, "~60 parameters"). byterover/retaindb are low
risk only after catalog entries exist.

**8. Needs more investigation.** Whether anything reads plugin registries before the first lookup
(decides the cost of lazy discovery); the semantic need for three tool-execution paths; SessionDB
table ownership and cross-process coordination before any split; updater churn; the env-var
surface; real usage of `hermes-agent`, drop-ins and `hermes claw`.

**9. Target architecture.** A Minimal Hermes Core — config/secrets, AIAgent, turn loop with a
middleware seam, prompt + compression, one lazily loaded transport and provider, registry + one
executor pipeline, file/terminal tools, and the unchanged safety boundary — with everything else
attached through plugins, toolsets and host processes that the core never imports
([diagram](diagrams/minimal-core.html), [06](06-minimal-core.md)). Enforced by an import-graph
contract and the minimal E2E turn in CI.

**10. What to do first and why.** Start with roadmap step **B1**, late-importing the four core→host
edges: highest Net score at high confidence, no behaviour change, and the prerequisite for every
later extraction. In parallel run **A1–A4** (shims and research runners), which are mechanical and
isolated. Leave the SessionDB split (D2) for last: it is the only step that rewrites persisted user
data, and it should wait until the import contract and CI budgets exist.
