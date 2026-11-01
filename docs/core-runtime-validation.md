# Core Runtime validation

Validated on 2026-11-01 on macOS arm64 in a fresh PM-built Python 3.14 environment. This environment has no prompt_toolkit, browser_harness or ACP package.

## Executed checks

- 279 tests passed, two host-gated tests skipped across the core owner, real MCP/plugin/skill extensions, edit approvals, profile isolation, plugin loading, shared session authority, scheduler and PM runtime selection.
- 19 configuration loading/migration tests passed.
- Four transport serialization tests passed after splitting the historical backend test file.
- The Docker contract passed: built the actual image, started the owner, checked readiness and UID 10000, verified Node/npm are absent, executed the real read_file tool, and stopped the container with SIGTERM.
- PM verified uv.lock is current with pyproject.toml.
- All 12 scripts/check gates passed, including code health with zero blocking/advisory findings. The check used a separate temporary Git index containing this change, excluding the user’s untracked `.claude/` and `docs/architecture/`; the real Git index was left untouched.
- git diff --check and production Python compilation passed.

Reproduce the core selection through the repository runner:

```sh
HERMES_PYTHON=/path/to/pm-built-test-env/bin/python scripts/run_tests.sh \
  tests/test_hermes_core.py tests/tools/test_core_extensions.py \
  tests/hermes_runtime/test_edit_approval.py tests/tools/test_mcp_tool.py \
  tests/gateway/test_profile_isolation_runtime.py tests/hermes_cli/test_plugins.py \
  tests/gateway/test_session_authority.py tests/cron/test_cron_script.py \
  tests/cron/test_cron_memory_contract.py tests/cron/test_shutdown_interrupt.py \
  tests/pm/test_core_runtime_selection.py
```

The integration fixture launches the production owner and a deterministic local model, executes file tools, replays events, cancels work, verifies persisted state and exercises the HTTP example client with SSE and stop. Extensions use a real stdio MCP server and temporary filesystem skills/plugins. The shared-authority probe verifies authenticated clients reuse a warm agent. Test homes are isolated from user state.

## Limits

No paid provider calls were made. Provider credentials and live model availability require operator configuration. Docker was tested on Linux arm64, with macOS as host; amd64 and Windows were not tested. The entire historical upstream suite was not run; it still contains tests describing removed products. Historical Markdown reference pages also describe those upstream surfaces.

The supported external integration is the authenticated HTTP API; trusted same-user clients can use the private canonical protocol. UI RPCs and bundled feature tools are outside this distribution. SIGTERM retains the existing gateway drain policy and exit code 1; the container test checks that deliberate behavior.
