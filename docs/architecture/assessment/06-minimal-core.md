# 06 — Minimum Viable Hermes and the Minimal Core Proposal (Phase 4)

Diagram: [diagrams/minimal-core.html](diagrams/minimal-core.html).

## 1. Requirements

The smallest Hermes must support: initialisation, LLM interaction, conversation/context
management, basic tool discovery and execution, essential state, error handling, and safe
execution boundaries — without weakening any security or reliability control.

## 2. Finding: the minimal core already exists inside the current code

It is reachable by **configuration alone**, with no code changes. Verified end to end:

```python
AIAgent(base_url="http://127.0.0.1:<port>/v1", api_key="sk-fake", provider="custom",
        model="fake-model", enabled_toolsets=["file", "terminal"], quiet_mode=True,
        skip_memory=True, skip_context_files=True, max_iterations=6)
.run_conversation("Read the probe file and report it.")
```

against a stdlib fake OpenAI server ([verification/](verification/)), isolated `HERMES_HOME`:

| Observation | Value | Tag |
|---|---|---|
| Import `run_agent` | 1.44 s (cold, in this run) | V |
| Construct agent | 0.45 s | V |
| Turn wall time (2 LLM calls + 1 tool) | 1.34 s against a local stub | V |
| Tools sent | `patch, read_file, search_files, terminal, write_file` + `tool_search, tool_describe, tool_call` bridges | V |
| System prompt | 7,522 chars (memory and context files off) | V |
| Messages | `user → assistant(tool_calls) → tool → assistant` | V |
| Requests | both `stream: true`, identical tool list (prompt-cache friendly) | V |
| Persistence | **no `state.db` created** — SessionDB is caller-supplied (`run_agent.py:272`) | V |
| `~/.hermes/state.db` | untouched (mtime before the run) | V |
| Modules loaded after the turn | 2,466 (openai SDK 804, `agent` 263, `tools` 220, `hermes_cli` 119, 42 provider plugins, `gateway` 31, `cron` 19) | V |

The code also has a second, stricter precedent: **safe-worker mode**
(`agent/safe_worker_policy.py`; `tools/registry.py:126-134`) imports only
`tools.file_tools`, `tools.terminal_tool` and `tools.process_registry` and skips plugin discovery
(`hermes_cli/plugins.py:1757-1762`). It is private to the managed worker bootstrap, but it proves
the tool layer works with a closed import set. [V]

## 3. Minimal Hermes Core (proposed boundary)

| Keep in core | Module(s) today | Why it is essential |
|---|---|---|
| Thin entry | `hermes --oneshot` or library use; `agent/legacy_cli.py` shows the shape | initialisation |
| Config + secrets | `hermes_cli/config.py:2088 load_config`, `hermes_cli/config_defaults.py`, `agent/secret_scope.py` | initialisation, profile isolation |
| `hermes_platform`, `hermes_constants` (paths), `hermes_logging` | as is | host facts, profile-aware paths, diagnostics |
| AIAgent + init | `run_agent.py`, `agent/agent_init.py` | conversation owner |
| Turn loop + phases | `agent/conversation_loop.py`, `agent/turn_*.py` (minus feature branches) | LLM ↔ tool alternation, budgets, retries |
| Prompt + compression | `agent/system_prompt.py`, `agent/prompt_builder.py`, `agent/context_compressor.py` | context management, cache stability |
| One transport | `agent/transports/chat_completions.py` + `base.py` | LLM interaction |
| One provider profile | `providers/` registry + `custom` (or one bundled profile) | endpoint resolution |
| Model metadata | `agent/model_metadata*.py` | context length |
| Tool registry + toolsets + definitions | `tools/registry.py`, `toolsets.py`, `model_tools.py` | discovery and dispatch |
| One executor path | `agent/tool_executor.py` (sequential path) | execution |
| File + terminal + process tools, local environment | `tools/file_*`, `tools/terminal_*`, `tools/process_*`, `tools/environments/local.py` | basic tools |
| **Safety (non-negotiable)** | `tools/approval*.py`, `build_subprocess_env`, `agent/secret_scope.py`, `agent/redact*.py`, file write guards | safe execution boundary |
| SessionDB (optional) | `hermes_state*.py` core tables only: sessions, messages, system_prompts, turn leases | resume; hosts need it, the library does not |

Everything else moves to the **edge** (already plugin-shaped or toolset-gated): host surfaces,
MCP, browser, media, delegation, code execution, kanban, cron, memory providers, skills curator,
MoA, LSP, pets, updater, telemetry, research runners.

## 4. What stops the minimal core from being *small* today

The boundary above is reachable, but the process still loads most of the system. In order of
measured impact:

1. **Eager plugin discovery on `import run_agent`** — 42 providers, 11 web, 8 image-gen, 4
   video-gen, 4 dashboard-auth, 3 browser plugins and the A2A platform plugin (which in turn imports
   `gateway.platforms`). [V]
2. **Static imports from core into hosts/features**: `agent.session_source → gateway`,
   `tools.cronjob_tools → cron`, `tools.environments.local_env_policy → hermes_cli.auth`,
   `agent.turn_context → agent.lsp`. [V]
3. **Tool self-registration imports every tool module** (88 tools) even when only two toolsets
   are enabled; filtering happens after import (`tools/registry.py:128` comment says so). [V]
4. **Feature branches inside the loop and transport-adjacent helpers** (MoA, Nous billing,
   Copilot, Ollama, codex app-server; 73 `api_mode` comparisons outside transports). [V]
5. **Tool Search bridge activates for a two-toolset agent** because `process_manage` is in the
   default `defer` list, adding three bridge tools to an eight-tool payload. [V]

## 5. Recommended minimal-core contract (to validate in Roadmap phase E)

- `python -c "import run_agent"` with `enabled_toolsets=["file","terminal"]` imports no module
  under `gateway`, `cron`, `tui_gateway`, `plugins.platforms`, `plugins.dashboard_auth`,
  `plugins.image_gen`, `plugins.video_gen`, `plugins.web`, `agent.lsp`, `agent.moa*`.
- Only the configured provider profile is imported.
- The minimal E2E turn above still passes, plus: approval denies `rm -rf /` through the terminal
  tool; secret scope raises `UnscopedSecretError` under multiplex without a bound scope; a resumed
  session with a SessionDB reproduces a byte-identical system prompt.
- Baselines to beat: 1,181 modules / 1.31 s at import; 2,466 modules after one turn.
