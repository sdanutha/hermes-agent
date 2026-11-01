# 09 — Evidence, Verification Log, Coverage and Open Questions

## 1. Repository identity

- Revision `db203baba58bd31fbf96fbb6932a733a6bf6d5ec` on local branch `dev`, clean worktree at
  the start; `dev` is not pushed (`git branch -r --contains HEAD` is empty); its parent
  `4d04303818` is on `origin/main`. Diagram source links are therefore `local-only`.
- Origin: `https://github.com/sdanutha/hermes-agent.git` (no credentials in the URL).

## 2. Verification procedures

| # | Procedure | Performed | Result |
|---|---|---|---|
| P1 | Read entry points: `pyproject.toml [project.scripts]`, `hermes_cli/main.py:3470-3582`, `agent/legacy_cli.py`, `acp_adapter/entry.py`, `tui_gateway/entry.py`, `gateway/run.py` | yes | entry chain in [05](05-runtime-flows.md) F1 |
| P2 | Read the turn driver `agent/conversation_loop.py:1484-1774` and phase signatures | yes | F3/F4 |
| P3 | Tool registry probe (import `model_tools`, list registry entries per toolset, schema sizes) | yes, isolated `HERMES_HOME` | 88 tools, 59 toolsets, core list 51 tools / 94,832 chars |
| P4 | Tool payload probe for `hermes-cli`, `safe`, `file`+`terminal` | yes | 20/40,420; 2/1,920; 8/10,715 |
| P5 | Import timing and module counts for 5 entry modules, two runs each | yes | [04](04-dependency-map.md) §2 |
| P6 | First-importer attribution with an `__import__` hook | yes | gateway, cron, auth, lsp, plugin edges |
| P7 | Minimal-core E2E: AIAgent + file/terminal against a stdlib fake OpenAI server | yes | passed; see [06](06-minimal-core.md) §2 |
| P8 | Leak check: no writes to the real `~/.hermes/state.db` | yes (mtime unchanged) | clean |
| P9 | Introspection of `AIAgent` params/mixins, `SessionDB` mixins/methods | yes | 86/14; 19/254 |
| P10 | `DEFAULT_CONFIG` size, `HERMES_*` name count | yes | 100 sections, 908 keys; 549 names |
| P11 | Churn and size per candidate (`git log --since=2026-07-22`, `wc -l`) | yes | in [07](07-reduction-candidates.md) |
| P12 | Reverse dependencies of legacy candidates (grep across shipped code and tests) | yes | [04](04-dependency-map.md) §4 |
| P13 | Run the full test suite | **no** | not needed for an analysis-only task; required per roadmap step |
| P14 | Gateway, TUI, desktop and cron runtime traces | **no** | flows F9–F11 rely on code reading and area docs [I] |
| P15 | Perceptual review of the HTML diagrams | **no** | archify automated gates passed (validate, deliver, strict check, real-browser check); the browser pane could not open local files |
| P16 | Measure import savings of lazy plugin discovery | **no** | stated as an estimate in rank 5 |

## 3. Claim ledger (selected)

| Claim | Tag | Evidence |
|---|---|---|
| Hosts construct AIAgent in-process; library persists only with `session_db` | V | `run_agent.py:272`; P7 created no `state.db` |
| System prompt is cached and rebuilt only after compression | V | `agent/system_prompt.py:802-806` |
| Persist-before-execute in tool rounds | V | `agent/turn_tool_round.py:49-59` |
| Fallback activation re-arms preflight | V | `agent/turn_api_error.py:52-62` docstring |
| `import run_agent` loads all bundled provider/web/image/video/dashboard-auth plugins | V | P5/P6 |
| Discovery is meant to be lazy | V | `providers/__init__.py:14-20` docstring |
| Kanban tools are hidden by default | V | `tools/kanban_tools.py:83-102`; P4 shows none in `hermes-cli` |
| Safe-worker mode imports only three tool modules | V | `tools/registry.py:126-134` |
| byterover/retaindb past their removal date and without catalog entries | V | `plugins/AGENTS.md`; docs `memory-providers.md:829`; `ls plugin-catalog` |
| Research runners are not reachable from console scripts but ship in the wheel | V | `pyproject.toml:563-566`; `setup.py:87-99` |
| Telemetry is opt-in | V | `hermes_cli/config_defaults.py:2343-2352` |
| Cron ticker runs inside the gateway | V | `gateway/run_bootstrap.py:380-460` |
| Gateway caches agents per session and streams deltas | I | `gateway/run_turn.py:994` name/shape, `GatewayAgentCacheMixin` |
| Delegation semantics (roles, depth, concurrency) | I | `tools/AGENTS.md` |
| Compression algorithm order (prune → boundaries → summary) | I | `agent/AGENTS.md` |
| Lazy plugin discovery would save up to ~1 s per process | I (estimate) | difference between `hermes_cli.main` (0.40 s) and `run_agent` (1.31 s) is an upper bound, not a measurement of the change |
| Feature-coupling counts in the loop | V as string counts; I as coupling | grep over `agent/conversation_loop.py` + `agent/turn_*.py` |

## 4. Documentation drift found (code is authoritative)

| Doc statement | Code reality |
|---|---|
| `agent/AGENTS.md`: `AIAgent.__init__` "takes ~60 parameters" | 86 (introspection) |
| `tools/AGENTS.md`: TOOLSETS keys include `messaging`, `moa`, `rl` | none of the three exist among the 59 keys (registry probe) |
| `providers/__init__.py`: discovery is lazy | triggered during `import run_agent` |
| Root `AGENTS.md`: dependencies carry upper bounds `>=floor,<next_major` | `pyproject.toml:20-33` says every direct dependency is exact-pinned `==X.Y.Z` (stricter); the two rules disagree in wording |

## 5. Coverage

| Area | Depth |
|---|---|
| Entry points, startup, agent init, turn loop, LLM call, tool execution, approval, env builder | read + executed (P1–P7) |
| Tool registry/toolsets/definitions, Tool Search | read + measured |
| Providers, plugin discovery | read + measured at import |
| SessionDB | schema tables, mixins, constructors; no migration analysis |
| Memory, skills, compression | entry points read; algorithms from area docs [I] |
| Gateway, tui_gateway, web backend, ACP, cron, kanban | entry points and sizes; flows partly inferred |
| PM, hermes_platform, updater, observability | sizes, defaults, entry points; internals not traced |
| TypeScript clients | sizes only |
| `evals/`, `website/`, `nix/`, `docker/` | sizes/listing only |

## 6. Open questions

1. Does any code read a plugin registry directly (not through `get_provider_profile` or a backend
   lookup) before the first lookup? This decides whether B2 is a refactor or a one-liner.
2. Who still uses `hermes-agent` and `providers/*.py` drop-ins? No telemetry answers this by
   design; a deprecation release is the only safe probe.
3. Are the byterover/retaindb standalone repos going to be catalogued as unmaintained, or removed
   outright with a migration message?
4. Is the concurrent/sequential/segmented split in `agent/tool_executor.py` a semantic necessity
   (ordering of side-effecting tools) or history? Needs a design write-up before C2.
5. Which of the 549 `HERMES_*` names are secrets, process markers or config that belongs in
   `config.yaml` (root rubric)? Needs a scripted triage.
6. Why does the updater carry 906 commits on 20k lines in three months — platform breadth, or
   repeated repair of the same failure classes?
7. Does the gateway rely on SessionDB tables for cross-process coordination that a split would
   break (turn leases, admissions, worker receipts)?
