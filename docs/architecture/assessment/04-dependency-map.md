# 04 — Dependency Map and Impact Analysis (Phase 3)

Dependency kinds used below: **static** (module-level import), **runtime** (imported or called
inside a function on a production path), **optional** (behind a feature check), **config**
(selected by `config.yaml`), **external** (network service or SDK), **test-only** (referenced
only from `tests/` or `evals/`).

## 1. Layered dependency graph

```
            ┌──────── hosts (B) ────────────────────────────────────────────────┐
            │ cli.py+hermes_cli  tui_gateway  gateway(+cron,kanban)  acp_adapter │
            └───────────────┬───────────────────────────────────────────────────┘
                            │ construct AIAgent, pass session_db, callbacks (runtime)
            ┌───────────────▼───────────── agent runtime (A) ─────────────────────┐
            │ run_agent.AIAgent → agent_init → conversation_loop → turn_* phases  │
            │   ├─ system_prompt / prompt_builder ── memory snapshot, skills index│
            │   ├─ context_compressor ── auxiliary_client (B)                     │
            │   ├─ transports/* ── openai / anthropic / boto3 SDKs (external)     │
            │   └─ tool_executor ── model_tools ── tools.registry                 │
            └──────┬────────────────────────────┬────────────────────────────────┘
                   │ static at import           │ runtime
       ┌───────────▼──────────┐      ┌──────────▼───────────────────────────────┐
       │ hermes_cli.config,   │      │ tools/* self-registered (A: file,terminal; │
       │ providers + 42 plugin│      │ B: everything else) → approval (A) →       │
       │ profiles, plugins.web│      │ environments/local.build_subprocess_env (A)│
       │ image_gen, video_gen │      └───────────────────────────────────────────┘
       │ dashboard_auth (!)   │
       └──────────────────────┘
       hermes_state (SessionDB) ← hosts and agent mixins (runtime, caller-supplied)
```

## 2. Measured import facts (Python 3.14 PM venv, isolated `HERMES_HOME`)

Procedure: `python -c "import X"` timed with `time.perf_counter`, module counts from
`sys.modules`, first-importer attribution with an `__import__` hook. Scripts are in
[verification/](verification/). Run twice; numbers were stable to ±0.02 s.

| Import | Wall time | Modules in `sys.modules` |
|---|---|---|
| `hermes_cli.main` | 0.40 s | 496 |
| `cli` | 0.83 s | 784 |
| `gateway.run` | 1.00 s | 953 |
| `run_agent` | 1.31 s | 1,181 |
| after one minimal turn (file + terminal toolsets) | — | 2,466 (openai SDK alone 804) |

Packages loaded by `import run_agent` (first-party only): `tools` 187, `agent` 151,
`hermes_cli` 70, `plugins.model_providers` 42, `gateway` 26, `plugins.web` 24, `cron` 19,
`plugins.browser` 8, `hermes_platform` 5, `pm` 4, A2A platform plugin 4, plus single-module
plugins for 8 image-gen, 4 video-gen, 4 dashboard-auth, 3 browser and 11 web backends.

First importer of each surprising subsystem (from the hook):

| Subsystem pulled into `run_agent` | First imported by | Kind |
|---|---|---|
| `gateway` | `agent.session_source` | static |
| `gateway.platforms` | A2A platform plugin tools (`hermes_plugins.platforms__a2a.tools`) | static via plugin discovery |
| `cron`, `cron.scheduler` | `tools.cronjob_tools` | static (tool self-registration) |
| `hermes_cli.auth` | `tools.environments.local_env_policy` | static |
| `agent.lsp` | `agent.turn_context` | static |
| `plugins.*` (providers) | provider discovery triggered during import (`plugins.model_providers.solstice` seen first) | runtime-at-import |

Consequence: the "lazy" provider discovery documented in `providers/__init__.py:14-20` is in
practice triggered during `import run_agent` [V]. Every agent process (CLI, delegation child,
cron job, ACP) pays for all bundled plugins.

## 3. Tool payload (measured with `model_tools.get_tool_definitions`, fresh home, no keys)

| Toolsets | Tools sent | JSON chars |
|---|---|---|
| `hermes-cli` (default CLI bundle) | 20 (incl. 3 tool-search bridges) | 40,420 |
| `file` + `terminal` | 8 (incl. 3 bridges) | 10,715 |
| `safe` | 2 | 1,920 |
| all 88 registered schemas (upper bound) | 88 | 131,112 |
| `_HERMES_CORE_TOOLS` list (51) before gating | 51 | 94,832 |

Within the 51-tool core list, kanban is 14 tools and 25,877 chars (27%), but `check_fn` hides it
unless the profile has the `kanban` toolset or a dispatcher spawned the worker
(`tools/kanban_tools.py:83-102`) [V], so it costs nothing by default.

## 4. Impact analysis per removable candidate

Each row: who depends on it, what breaks, and how to validate. "Indep." = can be disabled on its
own today without code changes.

| Candidate | Dependents (kind) | Workflows affected | What breaks if removed | Startup impact | Effect on reasoning / tools / conversation | Persisted data / config | Risk | Indep. | Validating tests |
|---|---|---|---|---|---|---|---|---|---|
| byterover + retaindb in tree (778 lines) | `hermes_cli/memory_provider_migration.py:36` lists them in `_LEFT_CORE` (runtime); `hermes_cli/subcommands/memory.py:15` help text; config descriptors `hermes_cli/config_defaults.py:2983-2989`; 10 test files | Homes with `memory.provider: byterover/retaindb` | Those homes lose the provider until a catalog plugin exists — **no `plugin-catalog/retaindb.yaml` or `byterover.yaml` exists** | none | memory tools `retaindb_*` / `brv_query` vanish | memory data lives with the vendor; config keys stay | Medium (blocked by missing catalog entries) | yes (`memory.provider`) | `tests/plugins/` memory tests; migration E2E with a temp `HERMES_HOME` |
| Research runners (2,400 lines) | `batch_runner` docstring refs in `hermes_bootstrap.py:7`, logger group `hermes_logging.py:312`; `mini_swe_runner` → `trajectory_compressor` (static); tests: 7 files for batch_runner, 1 each for mini_swe / distributions, 2 provider tests import `trajectory_compressor` | RL/data generation run as `python batch_runner.py` | Nothing on any console-script path. Trajectory format itself lives in `agent/trajectory.py` and `agent/session_persistence.py`, which stay | none | none | none | Low | yes (not invoked) | `tests/test_batch_runner*`, `tests/test_mini_swe_runner.py`, `tests/test_trajectory_compressor*.py` move with the code |
| `hermes-agent` console script | `pyproject.toml:565`; installer PATH launcher per `agent/legacy_cli.py` docstring | Scripts calling `hermes-agent -q` | That command | none | none | none | Low–Medium (external scripts) | no | `tests/hermes_cli/test_update_interrupted_pull.py` etc. reference it |
| `providers/*.py` drop-in discovery | `providers/__init__.py:701-705` (runtime); no in-tree `providers/*.py` profiles remain | Out-of-tree single-file profiles in editable installs | Those drop-ins | small | none | none | Low | no | provider registry tests |
| `tool_delay` param, `cron = []` extra | `run_agent.py:242,283-285`; `pyproject.toml:279` | callers passing `tool_delay`; installs with `[cron]` | A `TypeError` for old callers; pip resolution of `[cron]` | none | none | none | Low | n/a | — |
| Eager plugin import (not a removal: make lazy) | every host via `import run_agent` | all | Registration timing if a caller reads a registry before first lookup | **saves up to ~1 s and ~600 modules per process [I: estimate from module counts]** | none if registry access goes through the lookup functions | none | Medium | n/a | import-time contract test + full suite |
| MoA hooks in the loop | `agent/moa*.py`; 151 mentions across 10 loop files | MoA presets | MoA, if moved without a seam | none | MoA turns only | `moa.*` config | Medium–High (deep coupling) | yes (`moa.active_preset` empty) | `tests/agent/*moa*` |
| Pets | `display.pet`; CLI/TUI/desktop renderers | cosmetics | mascot | small | none | `<home>/pets/` | Low | yes | `tests/*pet*` |
| LSP | `agent/turn_context.py:25,863`; `lsp.enabled` default True | post-edit diagnostics | edit feedback notes | spawns servers on demand | fewer edit diagnostics | none | Low–Medium | yes | `tests/agent/*lsp*` |
| SessionDB platform mixins (Telegram topics, gateway routing/heartbeats) | gateway (runtime), `hermes_state_telegram.py` (370 lines) | Telegram topics, gateway routing | Must move with data migration, not delete | none | none | **tables in `state.db` (schema v32)** | High (data) | no | `tests/hermes_state/*`, gateway E2E |
| Tool executor paths | `agent/turn_tool_round.py`, `agent/tool_executor.py:1585,1865,1935` | all tool rounds | ordering/parallelism semantics | none | tool ordering, timeouts | none | High | no | `tests/agent/*tool_exec*`, `tests/run_agent/*` |
| Product plugins (google_meet, teams_pipeline, achievements) | plugin enable lists [I] | users of those plugins | the plugin until catalogued | none | their tools | plugin config | Low–Medium | yes | their own `tests/` dirs |

## 5. Tight coupling that needs refactoring before any extraction

1. **Agent core → gateway / cron / hermes_cli.auth at import** (table in §2). Extracting gateway or
   cron into optional packages is impossible until `agent.session_source`,
   `tools.cronjob_tools` and `tools.environments.local_env_policy` stop importing them at module
   level. [V]
2. **Feature logic inside loop phases** (MoA, Nous billing/entitlements, Copilot credential
   staleness, Ollama context growth, codex app-server, bot chat). These need a middleware or
   provider-hook seam first. [V] occurrences / [I] seam shape.
3. **`api_mode` branching outside `ProviderTransport`** (73 sites, 15 files). A transport method
   per decision is the prerequisite for making non-chat transports optional. [V]
4. **SessionDB as a 19-mixin god object** holding gateway- and Telegram-specific tables. Splitting
   needs a schema-owned-by-module rule and a migration. [V]
5. **`_last_resolved_tool_names` process global** saved/restored around delegation
   (`agent/AGENTS.md`). Blocks running children concurrently in one process safely. [I]
