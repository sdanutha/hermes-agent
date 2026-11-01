# 07 — Reduction Candidate Report (Phase 5)

## 1. Scoring criteria (1 = low, 5 = high)

| Criterion | Meaning | How scored |
|---|---|---|
| **C** Core criticality | How much the minimal turn needs the thing being removed or changed | 5 = on the verified minimal path; 1 = not on any default workflow |
| **K** Coupling | How entangled it is with the core | 5 = imported at module level by core or many call sites; 1 = isolated. Counted as value: decoupling is the point |
| **O** Overlap | How much another mechanism already covers it | 5 = fully duplicated |
| **M** Maintenance burden | Code and churn it carries (measured) | larger bucket of lines (<1k:1, <3k:2, <10k:3, <25k:4, else 5) and 3-month commits (<20:1, <50:2, <200:3, <600:4, else 5) |
| **R** Runtime overhead | Cost every process pays | measured (import time, module count, schema chars) where marked *m*, else estimated *e* |
| **Rk** Removal risk | Data loss, security or user-visible breakage | 5 = persisted data or a security control |
| **E** Effort | Work to do it safely | 1 = one small PR; 5 = multi-PR refactor with migrations |
| **Cf** Confidence | In the scores themselves | H / M / L |

**Value = M + R + O + K. Cost = C + Rk + E. Net = Value − Cost.** Rank by Net, then confidence,
then Value. No weights beyond this; anything not measured is marked *e*.

Actions: **KEEP**, **CONFIGURE** (change a default or setting), **EXTRACT** (move out of the core
tree or wheel, keep working), **SIMPLIFY** (same behaviour, less mechanism), **CONSOLIDATE**
(merge duplicates), **REMOVE**, **INVESTIGATE**.

## 2. Ranked candidates

| Rank | Candidate | C | K | O | M | R | Rk | E | Value | Cost | Net | Cf | Action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Core → host/feature **static imports** (`agent.session_source→gateway`, `tools.cronjob_tools→cron`, `tools.environments.local_env_policy→hermes_cli.auth`, `agent.turn_context→agent.lsp`) | 1 | 5 | 1 | 2 | 3*e* | 2 | 2 | 11 | 5 | **+6** | H | SIMPLIFY (late imports at the use site) |
| 2 | **Legacy shims**: `tool_delay` param, `cron = []` extra, `providers/*.py` pkgutil drop-ins, `hermes-agent` console script | 1 | 1 | 5 | 2 | 1*e* | 2 | 1 | 9 | 4 | **+5** | H (script: M) | REMOVE after a deprecation notice |
| 3 | **Research runners** in the wheel: `batch_runner.py`, `mini_swe_runner.py`, `trajectory_compressor.py`, `toolset_distributions.py` (2,400 lines, 51 commits) | 1 | 1 | 3 | 3 | 1*e* | 1 | 1 | 8 | 3 | **+5** | H | EXTRACT (to `scripts/research/` or a separate package; keep `agent/trajectory.py`) |
| 4 | **`api_mode` branching outside transports** (73 sites in 15 files; transports+helpers 14.6k lines, 891 commits) | 3 | 5 | 4 | 5 | 1*e* | 3 | 4 | 15 | 10 | **+5** | M | CONSOLIDATE into `ProviderTransport` methods |
| 5 | **Eager plugin discovery** at `import run_agent` (42 provider + 11 web + 8 image + 4 video + 4 dashboard-auth + 3 browser + A2A) | 1 | 4 | 1 | 2 | 5*m* | 3 | 3 | 12 | 7 | **+5** | M | SIMPLIFY (lazy, per-kind, on first lookup) |
| 6 | **byterover + retaindb** still in tree after the 2026-10-15 removal date (778 lines, 36 commits) | 1 | 1 | 4 | 2 | 1*m* | 2 | 2 | 8 | 5 | **+3** | H | REMOVE once catalog entries exist (none today) |
| 7 | **Tool Search bridge on tiny toolsets** (3 bridge tools added to an 8-tool payload because `process_manage` is deferred by default) | 2 | 1 | 3 | 1 | 2*m* | 1 | 1 | 7 | 4 | **+3** | M | CONFIGURE (skip the bridge below a tool-count threshold) |
| 8 | **Pets** (`agent/pet`, `hermes_cli/pets.py`; 2.7k lines; default off) | 1 | 2 | 1 | 2 | 1*e* | 1 | 2 | 6 | 4 | **+2** | H | EXTRACT to a catalog plugin |
| 9 | **Product plugins in tree**: google_meet, teams_pipeline, hermes-achievements (4.8k lines) | 1 | 1 | 2 | 3 | 1*e* | 2 | 2 | 7 | 5 | **+2** | M | EXTRACT to catalog (matches `plugins/AGENTS.md` June 2026 policy) |
| 10 | **OpenClaw migration** `hermes claw` (590 lines, 25 commits) | 1 | 1 | 2 | 2 | 1*e* | 2 | 1 | 6 | 4 | **+2** | L | INVESTIGATE usage, then EXTRACT to a plugin |
| 11 | Loop **feature branches** (MoA 151, Nous 124, billing 97, kanban 52, Copilot 34, Ollama 21 mentions; loop 14.8k lines, 801 commits) | 4 | 5 | 2 | 5 | 2*e* | 4 | 5 | 14 | 13 | +1 | M | SIMPLIFY (turn-middleware seam; move MoA, billing, provider quirks behind it) |
| 12 | **SessionDB platform-specific mixins/tables** (Telegram topics, gateway routing, heartbeats; SessionDB 19 mixins, 254 methods, 26.6k lines, 1,264 commits) | 3 | 4 | 2 | 5 | 1*e* | 5 | 4 | 12 | 12 | 0 | M | INVESTIGATE, then EXTRACT to gateway-owned modules with a migration |
| 13 | **Updater/installer** (`hermes_cli/update*.py` 20.2k lines, 906 commits) | 1 | 2 | 2 | 5 | 1*e* | 4 | 5 | 10 | 10 | 0 | L | INVESTIGATE (why so much churn per line) |
| 14 | **Tool executor paths** (concurrent / sequential / segmented, 1,966 lines, 143 commits) | 4 | 3 | 4 | 3 | 1*e* | 4 | 4 | 11 | 12 | −1 | M | INVESTIGATE, then CONSOLIDATE shared middleware |
| 15 | `hermes_constants_scratch*.py` misleading name (production scratch-dir retention) | 3 | 2 | 1 | 1 | 1 | 2 | 2 | 5 | 7 | −2 | H | KEEP (optionally rename; low value) |
| 16 | `HERMES_*` env surface (549 distinct names in shipped Python) | ? | ? | ? | ? | ? | ? | ? | — | — | — | L | INVESTIGATE (triage: secret / process marker / config that belongs in `config.yaml`) |

Explicit **KEEP** decisions (scored low or protective; listed so nobody re-proposes them):

| Item | Why keep | Evidence |
|---|---|---|
| Approval guards, `build_subprocess_env`, secret scope, redaction, file-write guards | Security boundary of the minimal core | `tools/approval.py:1151`, `tools/environments/local.py:378`, `agent/secret_scope.py:230` |
| Kanban tools in `_HERMES_CORE_TOOLS` | Hidden by `check_fn` unless opted in; zero default schema cost (default `hermes-cli` payload has no kanban tools, measured) | `tools/kanban_tools.py:83-102` |
| 42 provider profiles as plugins | Thin (2,446 lines total); cost is the eager import (rank 5), not their existence | `plugins/model-providers/` |
| 11 web backends | Interchangeable plugins, class C; no single default to collapse to | `plugins/web/` |
| Two compression layers (gateway hygiene 85%, agent 50%) | Documented on purpose in `agent/AGENTS.md`; different owners | — |
| Persist-before-execute in the tool round | Resume safety after destructive tools | `agent/turn_tool_round.py:49-59` |
| Shared-metrics telemetry (11.2k lines, 218 commits) | Opt-in and off by default (`telemetry.shared_metrics.enabled: False`), matching the rubric's opt-in rule; revisit its size later | `hermes_cli/config_defaults.py:2343-2352` |
| Evals | Not packaged; developer-only | `pyproject.toml` package list |

## 3. Notes per top candidate

**1 — Static imports.** Each of the four edges is a module-level import found by the import hook
(see [04-dependency-map.md](04-dependency-map.md) §2). Moving them to the use site keeps behaviour
and lets a minimal agent import without `gateway`, `cron`, `hermes_cli.auth`, `agent.lsp`.
Prerequisite for every EXTRACT of gateway or cron. Validation: import-graph contract test
(no `gateway*`, `cron*` in `sys.modules` after `import run_agent` with minimal toolsets), plus the
existing suites for those modules.

**2 — Legacy shims.** `tool_delay` already warns and is ignored (`run_agent.py:283-285`); the
`cron` extra is empty "for back-compat" (`pyproject.toml:279`); no in-tree `providers/*.py`
profiles remain, so the pkgutil path serves only out-of-tree drop-ins; `hermes-agent` duplicates
`hermes --oneshot`. Removal needs a release note and, for the console script, one release of a
deprecation message pointing to `hermes --oneshot`.

**3 — Research runners.** No console script reaches them; `setup.py:87-99` ships every root
`*.py` as a module, so they are installed everywhere. `trajectory_compressor.py` carries its own
provider detection (`_detect_provider`, tested by `tests/tools/test_config_null_guard.py`) — a
second resolution path. Keep the trajectory format in `agent/trajectory.py`, which
`run_agent --save_sample` and `save_trajectories` use.

**4 — `api_mode` branching.** The transport ABC already exists (`agent/transports/base.py:12-51`)
but behaviour decisions leak into `chat_completion_helpers.py` (16), `turn_recovery.py` (12),
`client_lifecycle.py` (11), `agent_runtime_helpers.py` (11), `auxiliary_client.py` (9) and 10 more
files. Moving each decision behind a transport method (capability flags) shrinks the loop and is
the precondition for loading non-chat transports lazily.

**5 — Eager plugin discovery.** Measured in an isolated home: `import run_agent` is 1.31 s and
1,181 modules, including every bundled provider/web/image/video/dashboard-auth plugin and the A2A
platform plugin. Discovery should be per kind and on first lookup (`get_provider_profile`,
web/image backend selection, platform adapter start). Risk: code that reads a registry directly
before a lookup; mitigate with a contract test for each kind.

**6 — Memory providers.** The policy date has passed (`plugins/AGENTS.md`;
`website/docs/user-guide/features/memory-providers.md:829`) and
`hermes_cli/memory_provider_migration.py:36` already treats both as left-core, but
`plugin-catalog/` has no `retaindb.yaml` or `byterover.yaml`, so removal would strand users.
Order: publish catalog entries (or mark them unmaintained/removed) → verify the startup
auto-install path with a temp `HERMES_HOME` → delete `plugins/memory/{byterover,retaindb}`.

## 4. Biggest complexity sources (not all are reduction candidates)

| Source | Measure |
|---|---|
| Desktop app TypeScript | 690k lines (267k tests), 31k file changes in 3 months |
| Messaging gateway + adapters | 191k Python lines; `GatewayRunner` 19 mixins |
| Turn loop and its feature branches | 14.8k lines, 801 commits; 10+ product concerns inside phases |
| SessionDB | 26.6k lines, 19 mixins, 254 methods, 1,264 commits, schema v32 |
| Updater | 20.2k lines, 906 commits |
| Config surface | 100 sections, 908 keys, 549 `HERMES_*` names |
| Provider resolution | ≈ 38k lines across auth, models, runtime_provider, credentials |
