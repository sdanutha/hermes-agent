# 03 — Capability Matrix (Phase 2)

Classes: **A** Core (needed for the minimal agent), **B** Optional (useful, opt-in or separable),
**C** Replaceable (a commodity piece with an external or simpler substitute), **D** Redundant
(overlaps another mechanism), **E** Legacy/Unused (kept for compatibility, past its date, or
not reachable from production entry points), **F** Unknown (needs investigation).

"Default-enabled" means active in a fresh profile for `hermes` chat. "Disable" names the existing
switch; "—" means none exists. Complexity uses measured line counts: S < 2k, M 2–10k, L 10–30k,
XL > 30k.

| # | Capability | Why it exists | Implemented by | Invoked by | Default-enabled | Disable | Overlaps | Cx | Class | Ev |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Agent turn loop | Model ↔ tool alternation with budgets and retries | `agent/conversation_loop.py`, `agent/turn_*.py` | every host | yes | — | — | L | **A** | V |
| 2 | Byte-stable prompt + caching | Cost: keep the cached prefix | `agent/system_prompt.py`, `agent/prompt_caching.py` | loop | yes | `prompt_caching` | — | M | **A** | V |
| 3 | Context compression | Long sessions within the window | `agent/context_compressor.py` + siblings | loop, `/compress`, gateway hygiene | yes (50% threshold) | `compression.*` | gateway hygiene (85%) is a second layer | L | **A** | V |
| 4 | Chat-completions transport | Talk to OpenAI-compatible endpoints | `agent/transports/chat_completions.py` | loop | yes | — | — | M | **A** | V |
| 5 | Anthropic / Bedrock / Codex transports | Native APIs | `agent/transports/{anthropic,bedrock,codex}.py` | by `api_mode` | per provider | provider choice | each other | M | B | V |
| 6 | Codex app-server mode | Hand whole turn to codex subprocess | `agent/transports/codex_app_server*.py`; `conversation_loop.py:1650` | `api_mode == "codex_app_server"` | no | opt-in | turn loop | M | B | V |
| 7 | Provider profiles (42) | Endpoint/model presets | `plugins/model-providers/*` | resolution | all loaded | — (all imported) | `custom` profile covers OpenAI-compatible ones | M | B (one is A) | V |
| 8 | Credential pools, OAuth, Nous free tier | Multi-key, subscription logins | `hermes_cli/auth*`, `agent/credential*`, `hermes_cli/anon_*` | resolution | yes | config | plain API key | L | B | V |
| 9 | Fallback providers chain | Survive provider outage | `agent/turn_api_error.py`, `fallback_providers` | API error | if configured | config | credential rotation | M | B | V |
| 10 | Auxiliary LLM routing | Side tasks on cheaper models | `agent/auxiliary*.py` | compression, titles, vision, curator | yes | `auxiliary.*` | main client | L | B | V |
| 11 | MoA (mixture of agents) | Multi-model answers | `agent/moa*.py` + loop phases | `moa_config` | no (preset empty) | `moa.active_preset` | — | M | B | V |
| 12 | Tool registry + toolsets | Discovery, filtering, dispatch | `tools/registry.py`, `toolsets.py`, `model_tools.py` | init, executor | yes | `toolsets`, `tools.<platform>` | — | M | **A** | V |
| 13 | Tool executor (concurrent + sequential + segmented) | Parallel safe tools, ordered unsafe ones | `agent/tool_executor.py` | tool round | yes | — | three paths share middleware | M | **A** (one path) / D (path multiplicity) | V |
| 14 | Tool Search deferral | Shrink tool payload | `tools/tool_search*.py` | definitions | auto | `tools.tool_search.enabled: off` | toolset filtering | M | B | V |
| 15 | File tools | read/write/patch/search | `tools/file_*` | model | yes | toolsets | terminal (`cat`, `sed`) | M | **A** | V |
| 16 | Terminal + processes | Shell execution, background jobs | `tools/terminal_*`, `tools/process_*` | model | yes | toolsets | — | L | **A** | V |
| 17 | Execution backends (docker/ssh/modal/daytona/singularity) | Sandboxed or remote execution | `tools/environments/*` | `terminal.backend` | local only | config | each other | M | B | V |
| 18 | Approval guards | Stop destructive commands | `tools/approval*.py` | terminal, execute_code, file writes | yes | `--yolo`, approval mode | — | M | **A (safety)** | V |
| 19 | Subprocess env builder + secret scope | Profile isolation, no secret leaks to children | `tools/environments/local.py:378`, `agent/secret_scope.py` | every spawn | yes | — | — | M | **A (safety)** | V |
| 20 | Redaction | Keep secrets out of logs/output | `agent/redact*.py` | logging, display | yes | `security.*` | — | S | **A (safety)** | I |
| 21 | Session persistence (SessionDB) | Resume, search, leases, multi-process safety | `hermes_state*.py` | hosts pass `session_db` | yes in hosts | — | gateway session store files | L | **A** for hosts, B for library | V |
| 22 | Session search (FTS) | Recall past sessions | `hermes_state_search.py`, `tools/session_search_tool.py` | model (deferred tool) | deferred | toolsets | memory | L | B | V |
| 23 | Built-in memory | Persistent notes in prompt | `tools/memory_tool*.py` | model, prompt | yes | `skip_memory`, `memory.*` | external memory providers | S | B | V |
| 24 | External memory providers (byterover, retaindb in tree) | Vendor memory backends | `plugins/memory/*` | `memory.provider` | no | config | catalog plugins | S | **E** (past Oct 15 2026 removal date) | V |
| 25 | Context engine plugins | Alternative context strategy | `agent/context_engine.py` | config | no | config | compressor | S | B | I |
| 26 | Skills (index, view, manage, slash) | Procedural knowledge at the edge | `agent/skill*`, `tools/skills*`, `skills/` | prompt, model, `/skill` | yes | `skills.*`, toolsets | — | L | B | V |
| 27 | Curator | Skill maintenance | `agent/curator*.py` | background | config | `curator.*` | — | S | B | I |
| 28 | Delegation / subagents | Parallel isolated work | `tools/delegate*`, `agent/delegation*` | model | yes | toolsets, `delegation.*` | kanban, cron | M | B | V |
| 29 | Code execution kernels | Python with tool RPC | `tools/code_*` | model | yes | toolsets | terminal | M | B | V |
| 30 | MCP client | Remote tools | `tools/mcp_*` | config `mcp` | if configured | config | plugins | L | B | V |
| 31 | MCP server mode | Expose Hermes over MCP | `mcp_serve.py` | `hermes mcp serve` [I] | no | — | ACP | S | B | I |
| 32 | Browser automation | Web interaction | `tools/browser*`, `plugins/browser` | model | yes (browser_exec) | toolsets, `browser.backend` | computer use | L | B | V |
| 33 | Web search/extract (11 backends) | Web research | `tools/web_*`, `plugins/web/*` | model | yes | toolsets | each other | M | B / C | V |
| 34 | Media: TTS, STT, vision, image, video, voice, wake word | Multimodal | `tools/{tts,transcription,vision,image,video,voice,wake}*` + plugins | model, gateway voice | partly | toolsets | — | L | B | V |
| 35 | Computer use, connectors, bot desktop | Desktop automation, hosted connectors | `tools/computer_use`, `tools/connectors`, `tools/bot_*` | model, desktop | partly (service-gated) | config | browser | L | B | V |
| 36 | Messaging gateway + 27 adapters | Telegram, Discord, Slack, … | `gateway/`, `plugins/platforms/*` | `hermes gateway` | no | — | — | XL | B | V |
| 37 | TUI / Desktop / dashboard backends | Rich clients | `tui_gateway/`, `hermes_cli/web*` | `hermes --tui`, desktop, `hermes dashboard` | no | — | CLI REPL | XL | B | V |
| 38 | CLI REPL | Interactive terminal | `cli.py`, `hermes_cli/cli_*` | `hermes` | yes | — | TUI | L | B (thin entry is A) | V |
| 39 | ACP adapter | Editor integration | `acp_adapter/` | `hermes-acp` | no | — | MCP server | M | B | V |
| 40 | Cron scheduler | Scheduled agent jobs | `cron/`, gateway ticker | gateway, `hermes cron` | with gateway | config | kanban schedule | L | B | V |
| 41 | Kanban | Multi-agent work queue | `hermes_cli/kanban*`, `tools/kanban*` | `kanban` toolset, dispatcher | no (tools hidden by `check_fn`) | toolsets | delegation, cron | L | B | V |
| 42 | Plugin system + catalog | Extend without core edits | `hermes_cli/plugins*`, `plugin-catalog/` | startup | yes | per-plugin | entry points, `providers/*.py` drop-ins | L | **A** (as the extension seam) | V |
| 43 | Legacy `providers/*.py` drop-in discovery | Back-compat single-file profiles | `providers/__init__.py:701-705` | discovery | yes | — | plugin dirs | S | **E/D** | V |
| 44 | `hermes-agent` legacy console script | Old single-query runner | `agent/legacy_cli.py` | `hermes-agent` | installed | — | `hermes --oneshot` | S | **E/D** | V |
| 45 | `tool_delay` param, `cron = []` extra | Back-compat | `run_agent.py:242,283-285`; `pyproject.toml:279` | — | — | — | — | S | **E** | V |
| 46 | Research runners | RL/data-generation trajectories | `batch_runner.py`, `mini_swe_runner.py`, `trajectory_compressor.py`, `toolset_distributions.py` | `python <file>.py` only | n/a (shipped in wheel) | — | `run_agent --save_sample` | M | **E/B** (not reachable from any console script) | V |
| 47 | OpenClaw migration (`hermes claw`) | One-time import from OpenClaw | `hermes_cli/claw.py`, `subcommands/claw.py` | `hermes claw` | — | — | `hermes import` | S | B (migration) | V |
| 48 | Pets | Cosmetic mascot | `agent/pet`, `hermes_cli/pets.py` | display | no | `display.pet.enabled` | — | M | B | V |
| 49 | LSP diagnostics | Post-edit feedback | `agent/lsp` | turn context | **yes** | `lsp.enabled` | linters via terminal | M | B | V |
| 50 | Updater / installer / backup | Self-update, repair, fleet restarts | `hermes_cli/update*`, `backup*`, `pm/` | `hermes update`, startup | yes | `updates.*` | OS package managers, Nix | L | B | V |
| 51 | Observability: shared metrics | Opt-in usage telemetry | `hermes_cli/observability/shared_metrics*` | consent | **no** (opt-in) | `telemetry.shared_metrics.enabled` | logs | L | B | V |
| 52 | Logging | Diagnostics | `hermes_logging.py` | everywhere | yes | `logging.*` | — | S | **A** | V |
| 53 | i18n | Localised UI strings | `agent/i18n*`, `locales/` | display | yes | `display.language` | — | M | B | I |
| 54 | Google Meet, Teams pipeline, achievements | Product integrations | `plugins/google_meet`, `teams_pipeline`, `hermes-achievements` | plugin enable | no [I] | plugins | catalog policy says third-party products ship out of tree | M | **C** (catalog candidates) | I |
| 55 | Evals | Benchmarks and probes | `evals/` | developers | n/a (not packaged) | — | tests | L | B (dev-only) | V |
| 56 | `hermes_constants_scratch*.py` | Scratch-dir retention and stuck-entry tracking (production code despite the name) | root | `hermes_constants.py`, `tools/environments/local.py`, `hermes_cli/doctor_state.py` | yes | — | — | S | **A-support** (naming issue only) | V |
| 57 | Safe-worker mode | Closed import graph for troubleshooting workers | `agent/safe_worker_policy.py`, `tools/registry.py:126-134` | managed worker bootstrap | no | private | — | S | **A-support** (template for a minimal core) | V |
| 58 | `HERMES_*` env surface (549 names) | Historical config knobs and process markers | everywhere | env | — | — | `config.yaml` | — | **F** (needs per-name triage) | V (count) |

## Summary by class

- **A Core (12 + safety + support):** turn loop, prompt/caching, compression, chat transport, one
  provider profile, tool registry/toolsets/definitions, one executor path, file + terminal tools,
  approval guards, subprocess env + secret scope, redaction, logging, config load, plugin seam.
- **B Optional (most of the product):** every host surface except a thin entry, MoA, auxiliary
  routing beyond compression, all media, browser, MCP, delegation, code execution, kanban, cron,
  memory and skills, updater, telemetry, LSP, pets.
- **C Replaceable:** web search backends (11 interchangeable), product plugins that the policy
  already says belong in the catalog.
- **D Redundant:** three tool-execution paths, two provider discovery mechanisms, two single-query
  entry points, api_mode branching duplicated outside the transport ABC.
- **E Legacy/Unused:** byterover/retaindb in tree past their removal date, `tool_delay`,
  `cron = []` extra, `hermes-agent` runner, research runners in the wheel.
- **F Unknown:** the env-var surface; whether the 42 provider plugins must be importable before
  the first `get_provider_profile` call.
