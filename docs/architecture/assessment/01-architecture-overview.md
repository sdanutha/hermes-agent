# 01 — Architecture Overview

Pinned revision: `db203baba58bd31fbf96fbb6932a733a6bf6d5ec` (branch `dev`, not pushed; parent
`4d04303818` is on `origin/main`). Evidence tags: **[V]** verified by reading code or running it,
**[I]** inferred from code structure or docs, **[U]** unknown. See
[09-evidence-coverage.md](09-evidence-coverage.md) for the verification log.

Diagram: [diagrams/architecture-overview.html](diagrams/architecture-overview.html) (explorable,
every node links to its source lines).

## The actual architecture in one paragraph

Hermes is **one synchronous agent library** (`run_agent.AIAgent` + `agent/`) wrapped by **five
host processes** (CLI REPL, `tui_gateway` JSON-RPC backend that serves TUI, Desktop and the web
dashboard, the messaging gateway, the ACP adapter for editors, and the cron ticker that runs
inside the gateway). Every host constructs `AIAgent` in-process and calls
`run_conversation()` per user turn [V]. The agent drives a phase pipeline
(`agent/conversation_loop.py` + 33 `agent/turn_*.py` siblings) that alternates between an LLM
call through a provider transport and a tool round through a registry of self-registering tool
modules [V]. Durable state lives in one SQLite file (`state.db`, `SessionDB`, schema v32) owned by
the host, not the agent: `AIAgent` only persists when a `session_db` is passed in [V].
Almost every capability beyond file + terminal (providers, web/image/video backends, messaging
platforms, memory providers, dashboard auth, browsers) is delivered as a **plugin** discovered
from `plugins/`, `$HERMES_HOME/plugins/` or pip entry points [V].

## Layers

| Layer | What it is | Main code | Size (py lines, measured) |
|---|---|---|---|
| Host surfaces | Processes that own sessions and call the agent | `hermes_cli/main.py`, `cli.py` + `hermes_cli/cli_*_mixin.py`, `tui_gateway/`, `gateway/`, `acp_adapter/`, `hermes_cli/web_server*.py` | ≈ 260k (gateway core 99k, platform adapters 87k, tui_gateway 47k, CLI 39k, web backend 33k) |
| Agent runtime | AIAgent facade, turn loop, prompt, compression, transports, auxiliary LLM | `run_agent.py`, `agent/` | 143k (`agent/`) |
| Provider/model resolution | Which endpoint, key, api_mode, model metadata, credential pools | `providers/`, `plugins/model-providers/` (42), `hermes_cli/auth*`, `hermes_cli/model*`, `hermes_cli/runtime_provider*` | ≈ 38k |
| Tool layer | Registry, toolsets, executor, ~88 built-in tools, approval, execution environments | `tools/`, `toolsets.py`, `model_tools.py`, `agent/tool_executor.py` | 129k (`tools/`) |
| State | SessionDB (sessions, messages, FTS, leases, gateway routing), memory files, skills | `hermes_state*.py` (60 files), `tools/memory_tool*.py`, `skills/` | 27k + |
| Extension | Plugin loader, plugin catalog, MCP client, hooks | `hermes_cli/plugins*.py`, `plugin-catalog/`, `tools/mcp_*` | ≈ 34k |
| Platform/runtime | Dependency/env manager, host facts, installer, updater, observability | `pm/`, `hermes_platform/`, `hermes_cli/update*`, `hermes_cli/observability/` | ≈ 52k |
| Clients (TypeScript) | Ink TUI, Electron desktop, dashboard SPA | `ui-tui/src`, `apps/desktop/src`, `web/src` | 84k / 690k (267k tests) / 62k TS lines |

Totals (measured with `wc -l`): about **1.0M lines of non-test Python** across the shipped packages
and **1.49M lines of Python tests**; ~7,180 test files.

## Two load-bearing invariants (confirmed in code)

1. **Byte-stable system prompt per conversation.** `agent/system_prompt.py:802` caches the prompt
   on the agent and only rebuilds after compression; memory is a frozen snapshot
   (`tools/memory_tool.py:1-5`); mid-session content rides user messages or tool results
   (`agent/AGENTS.md`). [V]
2. **Narrow core, capability at the edges.** Providers, web search, image/video generation,
   browsers, messaging platforms, memory providers and dashboard auth are plugins
   (`plugins/` tree; `hermes_cli/plugins.py:1757 discover_plugins`). [V] But the *import graph*
   does not reflect the narrow waist yet: importing `run_agent` loads 42 provider plugins,
   11 web plugins, 8 image-gen plugins, 4 video-gen plugins, 4 dashboard-auth plugins and an
   A2A platform plugin eagerly (measured, see [04-dependency-map.md](04-dependency-map.md)). [V]

## Host surfaces and how they reach the agent

| Surface | Entry point | Reaches AIAgent by | Evidence |
|---|---|---|---|
| `hermes` CLI (REPL, `--oneshot`, ~63 subcommand modules) | `hermes_cli/main.py:3470 main()` | `cli.py` HermesCLI in-process | [V] |
| TUI / Desktop / Web dashboard chat | `tui_gateway/entry.py:291 main()`; dashboard `hermes_cli/web_server.py:218` | JSON-RPC to `tui_gateway`, which hosts agents; dashboard embeds the real TUI (`web/AGENTS.md`) | [V] entry / [I] embedding |
| Messaging gateway (~27 platforms) | `gateway/run.py:5111 main()`, `GatewayRunner` (19 mixins, `gateway/run.py:3341`) | agent cache per session (`GatewayAgentCacheMixin`) | [V] |
| ACP (editor) adapter | `acp_adapter/entry.py:163` | in-process | [V] |
| Cron / kanban dispatcher | ticker thread started in `gateway/run_bootstrap.py:380-460` | runs jobs as agent sessions; headless ticks refuse to spawn a gateway (`cron/AGENTS.md`) | [V] ticker / [I] job path |
| Legacy single-query runner | `hermes-agent` → `agent/legacy_cli.py` | in-process | [V] |
| MCP server mode | `mcp_serve.py` | exposes Hermes over MCP | [I] |

## Where complexity concentrates (measured)

- **Most-churned areas, last 3 months** (`git log --since=2026-07-22`, file-change counts; 40,819
  commits in the window): `apps/desktop` 31k, `hermes_cli/` 13k, `agent/` 7.6k, `tools/` 6.5k,
  `gateway/` 6.1k. Per-line, the updater (`hermes_cli/update*.py`, 20k lines, 906 commits) and
  `hermes_state*.py` (26.6k lines, 1,264 commits) are the hottest subsystems.
- **Wide facades:** `AIAgent.__init__` takes **86** parameters and mixes in **14** classes;
  `SessionDB` mixes in **19** classes with **254** public methods; `GatewayRunner` mixes in
  **19** classes (all measured by introspection). `agent/AGENTS.md` still says "~60 parameters".
- **Config surface:** `DEFAULT_CONFIG` has **100** top-level sections and **908** leaf keys;
  **549** distinct `HERMES_*` names are referenced in shipped Python (measured).
- **Turn loop feature coupling** (string occurrences in `agent/conversation_loop.py` +
  `agent/turn_*.py`): MoA 151, Nous 124, billing 97, gateway 84, kanban 52, Copilot 34,
  desktop 27, Ollama 21, LSP 17, codex_app_server 14. This is a proxy, not a call graph.
- **Provider branching outside the transport ABC:** 73 `api_mode ==/!=/in` comparisons across 15
  files outside `agent/transports/` (`agent/chat_completion_helpers.py` 16, `turn_recovery.py` 12,
  `client_lifecycle.py` 11, `agent_runtime_helpers.py` 11, `auxiliary_client.py` 9, …).
