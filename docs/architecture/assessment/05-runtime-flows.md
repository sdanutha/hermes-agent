# 05 — Runtime Execution Flows

Diagram for the turn: [diagrams/agent-turn.html](diagrams/agent-turn.html). Tags: [V] verified by
reading the cited code or running it, [I] inferred.

## F1. CLI startup (`hermes` → chat)

1. `hermes_cli/main.py:3470 main()` sets the process title, warns on an unsupervised PID 1,
   forces UTF-8 on Windows and installs the OS trust store (`agent/ssl_verify.install_truststore`). [V]
2. Housekeeping before parsing: quarantined `.exe.old` sweep, stale-bytecode sweep when the
   checkout changed, `_recover_update_debts_on_startup()`, PM boot bootstrap unless the command is
   `update`, and a passive PM runtime check (`hermes_cli/venv_sync.check_runtime`). [V]
3. Fast paths return early: termux TUI/CLI, `serve`, chat. [V]
4. `_build_cli_parser()`; NixOS container mode re-execs into the container. [V]
5. `--yolo` sets `HERMES_YOLO_MODE` **before** discovery because `tools.approval` freezes it at
   import. [V] (`main.py` comment at the call site)
6. `_prepare_agent_startup(args)` (`main.py:2886`) runs plugin/MCP/hook discovery only for
   commands that can run a turn. [V]
7. `--oneshot` → `_run_oneshot_from_args` (`main.py:3565`); no subcommand → `_default_to_chat`;
   else `args.func(args)`. [V]

Measured: `import hermes_cli.main` 0.40 s / 496 modules; `import cli` 0.83 s; first agent
construction pulls `run_agent` (1.31 s, 1,181 modules). [V]

## F2. Agent construction

`AIAgent.__init__` (`run_agent.py:236-286`) forwards 86 keyword parameters to
`agent/agent_init.py::init_agent` (drops the deprecated `tool_delay`). Init resolves the client
(provider profile, base URL, api_mode, key or credential pool), computes tool definitions through
`model_tools.get_tool_definitions` (toolset filter → availability `check_fn` → Tool Search
deferral), loads memory unless `skip_memory`, and sets up the compressor. Measured in the E2E run:
**0.45 s** to construct with file + terminal toolsets against a local endpoint; the log shows the
OpenAI client created, Tool Search tier 1 activated (1 deferred tool), and the context length
probed and cached (128k). [V]

## F3. One conversation turn

1. `AIAgent.run_conversation` (`agent/turn_facade.py:22-75`) admits a durable turn lease
   (`agent/turn_facade_lease.py:297`) when a SessionDB exists, then forwards to
   `agent/conversation_loop.py:1717`. [V]
2. `_run_conversation_turn` (`conversation_loop.py:1555`): decode inline MoA, reset per-turn state,
   refresh `.env` credentials, `build_turn_context` (`agent/turn_context.py:1036`) restores or
   builds the system prompt (`_restore_or_build_system_prompt`, `conversation_loop.py:749`), may
   preflight-compress, binds a voice route. [V]
3. If `api_mode == "codex_app_server"` the whole turn goes to the codex subprocess
   (`conversation_loop.py:1650`). [V]
4. Scripted prelude (tool calls a prompt built-in plays first). [V]
5. Loop while `api_call_count < max_iterations and iteration_budget.remaining > 0` or one grace
   call (`conversation_loop.py:1656-1658`):
   `begin_iteration` → `prepare_iteration` → `assemble_api_request` → `run_preflight_gate`
   (may compress and `continue`) → `announce_api_call` → API retry loop (F4) →
   `apply_retry_restarts` → `normalize_model_response` → `run_tool_round` if tool calls else
   `finish_text_response`; any exception → `handle_outer_loop_error`. [V]
6. `finalize_turn` (`agent/turn_finalizer.py:525`) builds the result; `run_conversation` stamps the
   turn boundary and closes durable failed turns. [V]

Each phase receives only the loop fields its signature names and returns a verdict
(`return`/`break`/`continue`/fall through) — `_run_phase`, `conversation_loop.py:1484-1502`. [V]

## F4. LLM call

`_run_api_retry_loop` (`conversation_loop.py:1505-1534`): `nous_rate_limit_guard` →
`build_api_request` (`agent/turn_api_request.py:93`) → `perform_api_call`
(`agent/turn_api_call.py:67`) → `check_api_response` (`agent/turn_response_check.py:92`).
`InterruptedError` → `handle_api_interrupt`; any other exception → `handle_api_error`
(`agent/turn_api_error.py:52`), which can rotate credentials, activate a fallback provider (always
re-arming the preflight against the new context window), compress on overflow, or give up. [V]

The transport is chosen by `api_mode` (`agent/transports/`), but 73 further `api_mode`
comparisons in 15 files outside the transports adjust behaviour (see
[04-dependency-map.md](04-dependency-map.md)). Streaming is the default: the fake server saw
`stream: true` on both calls. [V]

## F5. Tool execution

1. `run_tool_round` (`agent/turn_tool_round.py:49`) **persists the assistant tool-call message
   before executing** ("persist-before-execute is a durability invariant"). [V]
2. Inline tools (`todo_list`, `memory`, `session_search`, `clarify`, desktop callbacks) are
   intercepted via `INLINE_TOOL_EXECUTORS` (`agent/inline_tool_executors.py:253-318`). [V]
3. Otherwise the executor runs calls concurrently (`agent/tool_executor.py:1585`), sequentially
   (`:1865`) or in segments (`:1935`), with an authorization gate, per-tool timeouts, activity
   heartbeats and middleware. [V]
4. Dispatch goes through the registry (`model_tools.py:873 handle_function_call`); handlers return
   a JSON string. [V]
5. Terminal commands pass `check_all_command_guards` (`tools/approval.py:1151`): user deny rules,
   hard floors, dangerous-pattern detection, smart approval, then a human or gateway prompt;
   `execute_code` and file writes have their own guards (`tools/approval.py:1216`,
   `tools/file_tools_write_guards.py`). [V]
6. Every child process gets its environment from `tools/environments/local.py:378
   build_subprocess_env`. [V]

E2E: one `read_file` call completed in 0.06 s; a local environment and session snapshot were
created for task `default`. [V]

## F6. Memory operations

- **Read:** at session start `MEMORY.md` / `USER.md` enter the system prompt as a frozen snapshot;
  mid-session writes go to disk and never change the prompt (`tools/memory_tool.py:1-5`). [V]
- **Write:** the `memory` tool is an inline executor (add/patch/remove/batch). [V]
- **External provider:** `MemoryManager` (`agent/memory_manager.py:196`) runs the built-in provider
  first plus at most one external provider; `on_session_end` (`:626`) fans out at session end,
  with the caller binding the owning profile scope (`agent/AGENTS.md`). [V]
- Cron sessions pass `skip_memory=True` by default. [I] (documented in `agent/AGENTS.md`)

## F7. Skills

The skills index is part of the system prompt (`agent/prompt_builder.py:1245
build_skills_system_prompt`); the model loads a skill with `skill_view`
(`tools/skills_tool.py:592`); `/skill` slash commands inject the skill as a user message so the
prompt stays stable (`agent/skill_commands.py`). [V]

## F8. Compression

Preflight (`run_preflight_gate`) or provider-proven overflow (`handle_api_error`) triggers the
`ContextCompressor` (`agent/context_compressor.py:2152`): prune old tool results without an LLM
call, choose boundaries, summarise with the auxiliary compression model, keep the same session id
for in-place compaction, rebuild the system prompt once. Manual `/compress` everywhere goes
through `agent/conversation_compression_manual.py:69 compress_now`. [V] entry points / [I] the
algorithm order (from `agent/AGENTS.md`).

## F9. Messaging gateway message

Adapter event → `GatewayRunner._handle_message` (`gateway/run_inbound.py:1242`) →
authorization, busy-session and slash handling → `_handle_message_with_agent`
(`gateway/run_turn.py:215`) → `_run_agent` (`gateway/run_turn.py:994`) which reuses a cached
`AIAgent` per session and streams deltas back to the adapter. Gateway hygiene compression runs at
85% before the agent's own 50% compressor. [V] function chain / [I] caching and thresholds.

## F10. Cron

The gateway starts a `SupervisedTickerThread` (`gateway/run_bootstrap.py:380-460`) that calls
`cron/scheduler_tick.py:33 tick()` per served profile under that profile's scope; `next_run_at`
advances before dispatch (at-most-once), a per-home tick lock prevents duplicates, headless
`hermes cron tick` refuses agent jobs when no gateway owns the home. [V] ticker / [I] rest
(`cron/AGENTS.md`).

## F11. Delegation

`delegate_task` spawns a child agent with isolated context and terminal session; synchronous by
default, `background=true` returns an id and the result re-enters through the async completion
queue; roles `leaf` / `orchestrator`, depth ≤ 2, concurrency 3 by default (`tools/AGENTS.md`,
`tools/delegate_tool.py`). [I]

## F12. Error handling and observability

- Per-call: classified API errors, retries, fallbacks, credential refresh (F4). [V]
- Per-turn: outer-loop handler, durable failed-turn records with redacted display metadata
  (`conversation_loop.py:1780-1867`). [V]
- Logs: `<home>/logs/agent.log`, `errors.log`; the E2E log shows client creation, tool-search tier,
  context-length cache, per-call latency and token usage, and turn end reason. [V]
- Telemetry: shared metrics are opt-in (`telemetry.shared_metrics.enabled: False`,
  `send: False`, `hermes_cli/config_defaults.py:2343-2352`). [V]
