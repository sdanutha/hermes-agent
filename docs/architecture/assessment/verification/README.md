# Verification scripts

Read-only probes behind the measured numbers in this assessment. None of them edits the
repository. Run them from the repository root with a Python 3.14 interpreter whose environment
has Hermes' dependencies (this assessment used a PM-managed venv), and **always set an isolated
`HERMES_HOME`**: the probes create caches and logs there.

```bash
export HERMES_HOME="$(mktemp -d)" HERMES_RUNTIME_DIR="$(mktemp -d)"
PY=/path/to/hermes/venv/bin/python
```

| Script | Measures | Command |
|---|---|---|
| `probe_registry.py` | Registered tools per toolset, `_HERMES_CORE_TOOLS` schema size, TOOLSETS keys | `$PY docs/architecture/assessment/verification/probe_registry.py` |
| `probe_tool_payload.py` | Tools and JSON size actually sent for `hermes-cli`, `safe`, `file`+`terminal` | `$PY docs/architecture/assessment/verification/probe_tool_payload.py` |
| `probe_core_schema_by_toolset.py` | Schema chars per toolset inside the core bundle | `$PY docs/architecture/assessment/verification/probe_core_schema_by_toolset.py` |
| `who_imports.py` | First importer of a subsystem when importing a module | `$PY docs/architecture/assessment/verification/who_imports.py run_agent gateway cron hermes_cli.auth agent.lsp` |
| `fake_llm.py` + `run_minimal_turn.py` | Minimal-core E2E: AIAgent with file + terminal toolsets, one tool round against a stdlib fake OpenAI server | see below |

Minimal-core E2E:

```bash
echo probe-content-42 > /tmp/probe.txt
$PY -I docs/architecture/assessment/verification/fake_llm.py 18931 /tmp/llm.log /tmp/probe.txt &
$PY docs/architecture/assessment/verification/run_minimal_turn.py 18931 /tmp/mods.txt
kill %1; cat /tmp/llm.log
```

Expected: `final` starts with `DONE:` and contains the probe content, roles
`user, assistant, tool, assistant`, two logged requests with `stream: true` and the same 8 tool
names, and no `state.db` in `$HERMES_HOME` (the library does not persist without a
`session_db`).

Import timing used the one-liner
`$PY -c "import time,sys;t=time.perf_counter();import run_agent;print(time.perf_counter()-t, len(sys.modules))"`.
