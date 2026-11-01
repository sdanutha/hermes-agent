# 08 — Reduction Roadmap

Nothing here has been started. Every phase needs explicit approval. Each item lands as its own PR
rebased on `main`, with 1–2 invariant tests proven red on the base (root `AGENTS.md`), run through
`scripts/run_tests.sh` and `python scripts/check`.

Baselines to measure against (from this assessment, isolated `HERMES_HOME`):
`import run_agent` 1.31 s / 1,181 modules; one minimal turn 2,466 modules; `hermes-cli` payload
20 tools / 40,420 chars; `file`+`terminal` payload 8 tools / 10,715 chars.

## Phase A — Low-risk removals (rank 2, 3, 6 in [07](07-reduction-candidates.md))

| Step | Change | Depends on | Validation | Rollback |
|---|---|---|---|---|
| A1 | Drop `tool_delay` from `AIAgent.__init__`; drop the empty `cron` extra | release note | full `tests/run_agent`, `tests/agent`; `hermes pm lock` clean | revert PR |
| A2 | Remove `providers/*.py` pkgutil drop-in discovery (`providers/__init__.py:701-705`) | release note naming the plugin-dir replacement | provider registry tests; user plugin dir still overrides bundled (E2E with temp home) | revert PR |
| A3 | Deprecate `hermes-agent`: one release prints "use `hermes --oneshot`", next release removes `[project.scripts] hermes-agent` and `agent/legacy_cli.py` | A3a ships first | installer launcher tests (`tests/hermes_cli/test_update_interrupted_pull.py` and siblings that reference it) | restore the script entry |
| A4 | Move research runners to `scripts/research/` (or a separate package); exclude from the wheel; move their tests | none | wheel contents no longer contain `batch_runner`, `mini_swe_runner`, `trajectory_compressor`, `toolset_distributions`; `run_agent --save_sample` still writes a trajectory | `git revert` (pure move) |
| A5 | byterover/retaindb: add catalog entries (or mark removed), then delete `plugins/memory/{byterover,retaindb}` | catalog entries merged; maintainer decision on unmaintained repos | E2E: temp home with `memory.provider: retaindb` → startup installs the catalog plugin; `retaindb_*` tools reappear | keep the in-tree copy until the catalog path is proven |

Exit criteria: suites green; no change in the minimal E2E turn; wheel smaller.

## Phase B — Optional-feature isolation (rank 1, 5, 7)

| Step | Change | Depends on | Validation | Rollback |
|---|---|---|---|---|
| B1 | Late-import the four core→host edges (`agent.session_source→gateway`, `tools.cronjob_tools→cron`, `tools.environments.local_env_policy→hermes_cli.auth`, `agent.turn_context→agent.lsp`) | none | contract test: after `import run_agent`, no `gateway*`, `cron*`, `agent.lsp*` in `sys.modules`; gateway, cron and LSP suites green; patch-where-production-reads checks for moved imports | revert per edge |
| B2 | Lazy, per-kind plugin discovery: providers on first `get_provider_profile`, web/image/video backends on first selection, platform plugins only in the gateway, dashboard-auth only in the web server | B1 | contract test per kind (registry populated after first lookup; user plugin overrides still win); import baseline drops; full suite | flag-free revert; keep the eager call as the fallback during review |
| B3 | Tool Search: skip the bridge when the eager payload is below a threshold (e.g. < 12 tools), or drop `process_manage` from default `defer` | none | `file`+`terminal` payload returns 5 tools; Tool Search tests; prompt-cache stability (identical tool list across calls) | config revert |
| B4 | Extract pets, google_meet, teams_pipeline, hermes-achievements to catalog plugins using the existing `LEFT_CORE` migration row mechanism | catalog repos | migration E2E with temp home (plugin auto-installs, config keys preserved) | keep in-tree until migration proven |

Exit criteria: `import run_agent` module count and time measurably below baseline; minimal E2E
turn passes with the import contract.

## Phase C — Consolidation (rank 4, 14)

| Step | Change | Depends on | Validation | Rollback |
|---|---|---|---|---|
| C1 | Inventory the 73 `api_mode` comparisons; add transport methods/capability flags; replace call sites file by file, starting with `chat_completion_helpers.py` and `turn_recovery.py` | B2 (lazy transports become possible) | provider-wire tests and `evals/provider_wire` probes per api_mode; E2E with fake servers for chat_completions and anthropic message shapes | per-file revert |
| C2 | Tool executor: document the semantic difference of concurrent/sequential/segmented; extract shared middleware so each path is a scheduler over one call pipeline | investigation write-up | ordering, timeout, interrupt and authorization-gate tests; delegation tests | revert |
| C3 | Fold `trajectory_compressor`'s provider detection into normal provider resolution (if A4 keeps it in-repo) | A4 | its tests | revert |

## Phase D — Architectural simplification (rank 11, 12)

| Step | Change | Depends on | Validation | Rollback |
|---|---|---|---|---|
| D1 | Turn-middleware seam (before request / after response / on error) with a concrete first consumer: move MoA preparation and Nous billing/entitlement handling behind it | C1 | MoA tests, billing tests, message-shape invariants (alternation, byte-stable prompt); no new hook without a consumer (root rubric) | revert consumer by consumer |
| D2 | SessionDB ownership: move gateway routing/heartbeats and Telegram topics into gateway-owned siblings with an explicit schema migration (schema v33) | design review; backup/restore tests | migration up/down tests on real `state.db` fixtures; gateway E2E; `hermes backup` round trip | down-migration + revert; **data-bearing, highest risk** |
| D3 | Narrow `AIAgent.__init__` (86 params) into grouped option objects while keeping keyword compatibility for one release | D1 | all host constructors; ACP; delegation children | revert |

## Phase E — Minimal-core validation

| Step | Check | Pass condition |
|---|---|---|
| E1 | Minimal E2E turn ([verification/](verification/)) in CI against the fake server | final answer, 4 messages, identical tool list on both calls |
| E2 | Import contract | no `gateway*`, `cron*`, `tui_gateway*`, `plugins.platforms*`, `plugins.dashboard_auth*`, `plugins.image_gen*`, `plugins.video_gen*`, `plugins.web*`, `agent.lsp*`, `agent.moa*` after `import run_agent` with minimal toolsets |
| E3 | Safety | terminal tool denies a floor command (`rm -rf /`) without a prompt; `UnscopedSecretError` under multiplex without scope; child env built by `build_subprocess_env` excludes launch-profile secrets |
| E4 | State | with a SessionDB, a resumed session reproduces a byte-identical system prompt; persist-before-execute holds after a simulated crash in a tool |
| E5 | Budgets | record import time/modules and payload size; fail CI on regression beyond an agreed margin |

## What to do first, and why

**B1 (late-import the four core→host edges), then A1–A4 in parallel.** B1 has the best
value-to-cost ratio (Net +6, high confidence), changes no behaviour, and is the prerequisite for
every later extraction (gateway, cron, LSP, lazy plugins). A1–A4 are mechanical, isolated,
high-confidence removals that shrink the wheel and the surface without touching the turn loop.
Defer D2 (SessionDB) until the cheaper phases have produced an import contract and CI budgets,
because it is the only step that rewrites persisted user data.
