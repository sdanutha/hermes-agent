# 02 — Component Inventory (Phase 1: Discovery)

One row per major component. **Core?** answers "is it needed for the minimal turn" (verified by the
E2E run in [06-minimal-core.md](06-minimal-core.md) where marked). Line counts are `wc -l` over
`.py` files at the pinned revision; "3-mo commits" counts commits touching the paths since
2026-07-22. Tags: [V] verified, [I] inferred, [U] unknown.

## Entry points and startup

| Component | Responsibility | Source / entry | Public interface | Init behaviour | Core? | Evidence |
|---|---|---|---|---|---|---|
| `hermes` console script | Top-level CLI dispatch | `pyproject.toml:563-566`; `hermes_cli/main.py:3470-3582` | argparse tree (~63 subcommand modules in `hermes_cli/subcommands/`) | Sets process title, truststore TLS (`agent/ssl_verify.install_truststore`), sweeps stale bytecode, recovers interrupted updates, PM boot bootstrap + runtime check, fast paths (termux/serve/chat), container redirect, `--yolo` env before plugin discovery, `_prepare_agent_startup` (plugin discovery + shell hooks), then dispatch | Partly (only the chat/oneshot path) | [V] |
| `hermes_bootstrap.py` | UTF-8 stdio on Windows, first import of every entry point | root | import side effect | No-op on POSIX | No | [V] (docstring of `agent/legacy_cli.py`) |
| `hermes_cli/_early_recovery` | Repairs a `hermes update` killed mid-pull before anything else imports | `agent/legacy_cli.py:19-23` | `restore_interrupted_pull()` | Runs before argparse | No (update safety) | [V] |
| `hermes-agent` legacy runner | Single-query runner | `agent/legacy_cli.py` (94 lines) | `--query`, `--model`, `--api-key` | Builds AIAgent, runs one turn | No | [V] |
| `hermes-acp` | ACP server for editors | `acp_adapter/entry.py:163` (6.5k lines total) | ACP protocol | Loads env, logging, runs ACP agent | No | [V] |
| `hermes_startup_watchdog.py` | Startup liveness | root; used by `cli.py`, `hermes_cli/main.py`, `hermes_state_*` | — | — | No | [I] |

## Agent runtime

| Component | Responsibility | Source | Interface | Upstream → downstream | Core? | Evidence |
|---|---|---|---|---|---|---|
| `AIAgent` facade | Holds per-session agent state; assembled from 14 mixins | `run_agent.py:214-286` (1,563 lines) | `AIAgent(**86 kwargs)`, `chat()`, `run_conversation()` | Hosts → AIAgent → `agent/agent_init.py::init_agent` | **Yes** | [V] measured params/mixins |
| Agent init | Resolve client, tools, prompt inputs, memory, pools | `agent/agent_init.py` (2,393 lines) | `init_agent` | AIAgent → providers, `model_tools.get_tool_definitions`, memory manager | **Yes** | [V] |
| Turn facade + lease | Takes the durable session turn lease, then forwards | `agent/turn_facade.py:22-75`, `agent/turn_facade_lease.py:297` | `run_conversation` | AIAgent → lease (SessionDB) → loop | Lease: only with SessionDB | [V] |
| Turn loop | Phase pipeline: context → prep → preflight → API retry loop → intake → tool round or final text → finalize | `agent/conversation_loop.py:1555-1714`; 33 `agent/turn_*.py` (14.8k lines total) | `run_conversation(agent, user_message, …) -> dict` | Calls prompt builder, compressor, transport, tool executor | **Yes** | [V] |
| System prompt | Tiered stable → context → volatile prompt, cached on agent | `agent/system_prompt.py:736-806`, `agent/prompt_builder.py` | `build_system_prompt(agent)` | Reads skills index, memory snapshot, context files, env hints | **Yes** (7,522 chars even with memory and context files off — measured) | [V] |
| Context compression | Prune old tool output, pick boundaries, LLM summary via auxiliary model; the only sanctioned cache break | `agent/context_compressor.py:2152`, `agent/conversation_compression*.py`, `agent/turn_context_compaction.py` (14.8k lines incl. trajectory compressor) | `ContextCompressor`, `compress_now` | Loop → compressor → auxiliary LLM | **Yes** for long sessions (a minimal turn never triggers it) | [V] code / [I] necessity |
| LLM transports | Per-api_mode message/tool conversion and response normalisation | `agent/transports/base.py:12-51` + `chat_completions.py:516`, `anthropic.py`, `bedrock.py`, `codex.py`, `codex_app_server*.py` | `ProviderTransport` ABC | Loop → transport → OpenAI/Anthropic/Bedrock SDKs | chat_completions **Yes**; rest optional | [V] |
| Client lifecycle + streaming | Create/refresh SDK clients, stream deltas | `agent/client_lifecycle.py`, `agent/chat_completion_helpers.py`, `agent/stream_*` (≈18k lines with transports) | mixins | — | **Yes** (streaming used by default — E2E saw `stream: true`) | [V] |
| Auxiliary LLM client | Side-model calls (compression, titles, vision, curator, session search) | `agent/auxiliary*.py` (18 files, 10k lines) | `_resolve_auto_route` | Compression/title → aux | Optional (needed only by compression/titles) | [V] |
| Error/retry/recovery | API error classification, fallback chain, credential rotation, post-exhaustion auto-recover | `agent/turn_api_error.py:52`, `agent/turn_recovery*.py`, `agent/error_*` | phase functions | — | **Yes** (basic retry); fallback/credential pools optional | [V] |
| MoA (mixture of agents) | Reference models + aggregator per turn | `agent/moa_loop.py`, `moa_alternation.py`, `moa_trace.py` (1.7k lines) + 151 mentions in loop phases | `moa_config` | Loop | No (opt-in preset) | [V] |
| LSP | Language-server diagnostics after edits | `agent/lsp/` (4k lines); `agent/turn_context.py:25,863` | `lsp.enabled` (default **True**) | Turn context → LSP | No | [V] |
| Pets | Cosmetic animated mascot | `agent/pet/` + `hermes_cli/pets.py` (2.7k lines) | `display.pet.enabled` default False | — | No | [V] |
| i18n | Layered locale lookup | `agent/i18n*.py`, `locales/` | `t(key)` | — | No (English fallback is bare key) | [I] |

## Providers and model routing

| Component | Responsibility | Source | Core? | Evidence |
|---|---|---|---|---|
| Provider profile registry | Bundled, user and pip provider profiles; user overrides bundled | `providers/__init__.py:192 get_provider_profile`, `:260 list_providers`; legacy `providers/*.py` pkgutil discovery `:701-705` | **Yes** (one profile) | [V] |
| 42 bundled provider plugins | One `plugin.yaml` + `__init__.py` each (2,446 lines total) | `plugins/model-providers/*` | One of them, or `custom` | [V] |
| Runtime provider resolution | Precedence ladder: explicit → env → config → pools → Nous free tier → Bedrock chain | `hermes_cli/runtime_provider.py:1041`; `hermes_cli/anon_auth.py` docstring | **Yes** (simplified) | [V] entry / [I] full ladder |
| Auth + credential pools | OAuth/device flows, pooled keys, refresh on 401 | `hermes_cli/auth*.py` (18 files, 10k), `agent/credential*.py` (4k) | No (API key suffices) | [V] |
| Model metadata | Context lengths, capabilities, models.dev cache | `agent/model_metadata*.py`, `hermes_cli/models*.py` | **Yes** (context length) | [V] E2E cached 128k |
| Nous free tier / anonymous identity | Mint anonymous Nous account, challenge flow | `hermes_cli/anon_*.py` (2.3k), `hermes_cli/nous*.py` | No | [V] |

## Tool layer

| Component | Responsibility | Source | Core? | Evidence |
|---|---|---|---|---|
| Tool registry | AST-scanned self-registration of `tools/*.py`, schema collection, dispatch, TTL-cached `check_fn` | `tools/registry.py:121-161` | **Yes** | [V] |
| Toolsets | 59 named groups + `_HERMES_CORE_TOOLS` (51 tools) | `toolsets.py:11-41` | **Yes** | [V] measured |
| Tool definitions | Filter by toolset, availability, tool-search deferral, cross-reference post-processing, LRU cache | `model_tools.py:215-245` | **Yes** | [V] |
| Tool Search bridge | Replaces deferred tools with `tool_search`/`tool_describe`/`tool_call` | `tools/tool_search*.py`; config `tools.tool_search` (`hermes_cli/config_defaults.py:2002-2040`) | Optional (it activated even for file+terminal because `process_manage` is deferred by default) | [V] E2E log |
| Tool executor | Concurrent, sequential and segmented execution; authorization gate, timeouts, heartbeats, middleware | `agent/tool_executor.py:1585, 1865, 1935` (1,966 lines) | **Yes** (one path) | [V] |
| Inline (agent-level) tools | `todo_list`, `memory`, `session_search`, `clarify`, desktop callbacks intercepted before the registry | `agent/inline_tool_executors.py:253-318` | No | [V] |
| File + terminal + process tools | read/write/patch/search, terminal, background processes | `tools/file_*`, `tools/terminal_*`, `tools/process_*` (24k lines with environments) | **Yes** | [V] E2E used `read_file` |
| Execution environments | local, docker, ssh, modal, daytona, singularity | `tools/environments/` (7.8k); one env builder `local.py:378 build_subprocess_env` | local **Yes** | [V] |
| Approval guards | Dangerous-command detection, user deny rules, floors, smart approval, human/gateway wait, YOLO | `tools/approval*.py` (9 files, 4.5k), `check_all_command_guards` `tools/approval.py:1151` | **Yes (safety)** | [V] |
| Code execution | Python kernels with tool RPC | `tools/code_*` (2.9k) | No | [V] |
| Delegation / async | Subagents, background completions | `tools/delegate*`, `tools/async_*`, `agent/delegation*` (6.1k) | No | [V] |
| MCP client (+ server) | Discover/register remote MCP tools; serve Hermes over MCP | `tools/mcp_*` (27 files, 11.9k), `mcp_serve.py` | No | [V] |
| Browser | CDP, cloud backends, browser-use | `tools/browser*` + `plugins/browser` (10k) | No | [V] |
| Media | TTS, STT, vision, image/video generation, voice, wake words | `tools/tts*`, `transcription*`, `vision*`, `image*`, `video*`, `voice*`, `wake*` + plugins (18.5k) | No | [V] |
| Computer use, connectors, bot desktop | Desktop automation, Nous tool gateway connectors | `tools/computer_use`, `tools/connectors`, `tools/bot_*` (15.7k) | No | [V] |
| Kanban tools | 14 board tools in `_HERMES_CORE_TOOLS`, hidden unless the profile has the `kanban` toolset or a dispatcher spawned the worker | `tools/kanban_tools.py:83-102` | No | [V] |

## State, memory, skills

| Component | Responsibility | Source | Core? | Evidence |
|---|---|---|---|---|
| SessionDB | SQLite WAL store: sessions, messages, FTS (3 variants), system prompts, turn leases, compression locks, gateway routing/heartbeats, worker receipts, Telegram topics | `hermes_state.py:463` (19 mixins, 254 public methods); 60 `hermes_state*.py` files (26.6k lines); `SCHEMA_VERSION = 32` (`hermes_state_common.py:289`) | Optional for the library, **Yes** for any host that resumes sessions | [V] (no `state.db` was created by the E2E run) |
| Built-in memory | `MEMORY.md` + `USER.md` as a frozen prompt snapshot; `memory` tool writes to disk | `tools/memory_tool.py:1-5`, `tools/memory_tool_store.py` | No (`skip_memory=True` works) | [V] |
| Memory manager + providers | Built-in provider plus at most one external plugin | `agent/memory_manager.py:196`, `agent/memory_provider.py`; `plugins/memory/byterover`, `retaindb` | No | [V] |
| Context engines | Pluggable context strategy | `agent/context_engine.py`, `plugins/context_engine` | No | [I] |
| Skills | 58 bundled `SKILL.md`, 153 optional; index in prompt, `skill_view`/`skills_list`/`skill_manage` tools, slash commands inject a user message | `agent/prompt_builder.py:1245`, `tools/skills_tool.py:290,592`, `agent/skill_commands.py` | No (but on in the default bundle) | [V] |
| Curator | Background skill maintenance via auxiliary LLM | `agent/curator*.py` (1.7k) | No | [V] |

## Hosts, scheduling, clients

| Component | Responsibility | Source | Size | Core? | Evidence |
|---|---|---|---|---|---|
| CLI REPL | prompt_toolkit REPL, slash commands, display | `cli.py` (1,797) + 40 `hermes_cli/cli_*.py` | 24.9k | No (oneshot suffices) | [V] |
| Messaging gateway | Adapters, session store, agent cache, slash commands, streaming, approvals, hygiene compression, restart/shutdown | `gateway/` (274 core files, 98.8k) + `gateway/platforms` (18.8k) + `plugins/platforms` (26 adapters, 67.9k) + relay (5.5k) | 191k | No | [V] |
| tui_gateway | JSON-RPC backend for TUI/Desktop/dashboard | `tui_gateway/` (118 files) | 46.7k | No | [V] |
| Web dashboard backend | FastAPI app, routers, dashboard auth | `hermes_cli/web*.py`, `web_routers/`, `dashboard_auth/`, `plugins/dashboard_auth` | 33k | No | [V] |
| Cron | Job store + scheduler + ticker (hosted by the gateway) | `cron/` (48 files) + `tools/cronjob*` | 21k | No | [V] |
| Kanban | Multi-agent work queue, dispatcher, board DB | `hermes_cli/kanban*`, `tools/kanban*`, `gateway/kanban*`, `plugins/kanban` | 21.5k | No | [V] |
| TS clients | Ink TUI, Electron desktop, dashboard SPA | `ui-tui/src`, `apps/desktop/src`, `web/src` | 84k / 690k / 62k TS | No | [V] |

## Extension, platform, operations

| Component | Responsibility | Source | Size | Core? | Evidence |
|---|---|---|---|---|---|
| Plugin loader | Discover, gate, load (10 s per-plugin deadline), ledger, isolation audit, hooks | `hermes_cli/plugins*.py`, `plugin_*.py`, `registration_lifecycle.py`, `plugins/plugin_*.py` | 18.7k | Needed for providers today (eager) | [V] |
| Plugin catalog | SHA-pinned out-of-tree plugin index + kill list | `plugin-catalog/*.yaml`, `removed.yaml` | YAML | No | [V] |
| Config + env | `config.yaml` defaults (100 sections / 908 keys), `.env` secrets, profile-aware paths | `hermes_cli/config*.py`, `hermes_cli/config_defaults.py` (3,156), `hermes_constants*.py` | 12.8k | **Yes** (load + defaults) | [V] |
| Secret scope | Per-profile secret lookup; fails closed under multiplex | `agent/secret_scope.py:39,230` | — | **Yes (safety)** | [V] |
| Redaction, vaults, secret sources | Log/output redaction, vault backends | `agent/redact*.py`, `agent/vault_backends`, `agent/secret_sources` | ≈ 3.8k | Redaction **Yes**; vaults optional | [V] |
| PM (dependency/env manager) | Managed interpreter, environments, extras gating, lazy installs | `pm/` | 12.8k | Installer-level, not turn-level | [V] |
| `hermes_platform` | Host facts and executable lookup | `hermes_platform/` | 2k | **Yes** (small) | [V] |
| Updater / install / backup | `hermes update`, debts, fleet restarts, backups, uninstall | `hermes_cli/update*.py` (40 files), `backup*`, `uninstall*` | 26.1k | No | [V] |
| Observability | Logging (`hermes_logging.py`), opt-in shared metrics (`telemetry.shared_metrics.enabled: False` default), monitoring | `hermes_cli/observability/` (11.2k), `agent/monitoring` | 13.8k | Logging **Yes**; metrics No | [V] |
| Research runners | Batch trajectory generation, mini-SWE runner, trajectory compressor, toolset distributions | `batch_runner.py`, `mini_swe_runner.py`, `trajectory_compressor.py`, `toolset_distributions.py` | 2.4k | No (shipped as root py-modules: `setup.py:87-99`) | [V] |
| Evals | Benchmarks/probes, not packaged | `evals/` (156 files) | 22.2k | No | [V] (absent from `[tool.setuptools.packages.find]`) |
| Other bundled standalone plugins | google_meet, teams_pipeline, hermes-achievements, disk-cleanup, security-guidance, computer_use, cron_providers | `plugins/*` | ≈ 4.8k (first three) | No | [V] |
