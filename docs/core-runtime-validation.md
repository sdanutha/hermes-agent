# Core Runtime validation

Validated on 2026-11-01 on macOS arm64 with isolated Python 3.14 environments, temporary Hermes homes/runtime directories, deterministic local model fixtures, a real local stdio MCP server, and Docker Desktop's Linux arm64 engine.

## Executed checks

- `python -m hermes_core pm lock`: lock regenerated through PM, then verified current.
- `scripts/run_tests.sh` with an explicit PM-built `HERMES_PYTHON`: 371 passing tests across the selected files below; 13 host-gated tests skipped on macOS. The real MCP round trip is separately exercised by `test_core_extensions.py`, which passed on this host.
- `docker build -t hermes-core:local .`: built successfully without frontend build steps.
- `tests/docker/test_core_runtime.py`: built and booted the actual image, checked readiness, UID 10000, absence of Node/npm, real file-tool execution, and SIGTERM shutdown.
- `docker compose config --no-env-resolution --quiet`: Compose structure validated without reading provider secrets.
- `git diff --check`: passed.
- `python scripts/check`: executed; the ordinary working-tree run also scans pre-existing, untracked `.claude/worktrees` and `docs/architecture` material and fails on their lint/path/blob findings. A second run with `--staged` and a separate temporary Git index validates only this change, preserving the user's actual index and untracked work. All 12 checks passed in that isolated snapshot.

The selected tests are:

```sh
HERMES_PYTHON=/path/to/pm-built-test-env/bin/python scripts/run_tests.sh \
  tests/test_hermes_core.py \
  tests/tools/test_core_extensions.py \
  tests/tools/test_skills_tool.py \
  tests/tools/test_mcp_tool.py \
  tests/hermes_cli/test_plugins.py \
  tests/gateway/test_profile_isolation_runtime.py \
  tests/gateway/test_session_authority.py \
  tests/pm/test_core_runtime_selection.py \
  tests/cron/test_cron_script.py \
  tests/cron/test_cron_memory_contract.py \
  tests/cron/test_shutdown_interrupt.py \
  tests/e2e/core/mcp_plugins/test_mcp_stdio_conformance.py \
  tests/scripts/test_code_health_cli.py \
  tests/docker/test_core_runtime.py
```

`test_hermes_core.py` launches the production owner, calls a deterministic local provider, executes `read_file` through the real agent loop, observes tool events, cancels a blocked turn with the generation-bound session control, checks persistence, and exercises the HTTP client for submission, SSE results and stop. It also probes live health and verifies SIGTERM releases the owner's PID file. `test_core_extensions.py` discovers a filesystem skill and a mounted Python plugin and executes them and a real stdio MCP tool through `model_tools.handle_function_call`. The profile suite verifies runtime scope separation. The PM selection test verifies that adding later dependencies retains the HTTP/MCP closure.

The shared-authority probe now uses the private native ticket protocol, with the actual profile-home identity and a generic test allowlist; it no longer scrapes a dashboard token. Messaging fixture imports are deferred so removed adapters do not prevent collecting unrelated core tests.

## Compatibility limits

- No real paid provider call was made. Provider authentication and live model availability depend on operator credentials. Deterministic tests exercise the production OpenAI-compatible invocation path.
- Linux arm64 Docker startup was verified. Amd64 images and Windows host behavior were not verified locally. Host-gated tests remain gated rather than faking the OS.
- This is a source/PM and Docker distribution. The upstream wheel/sdist guard remains. The entire upstream test suite was not run; presentation/messaging-specific suites are outside this distribution and some historical tests/reference pages still describe those removed products.
- Shared Python backend code remains under `tui_gateway/`; legacy CLI/admin modules are retained where shared runtime imports still depend on them. UI-specific RPCs and plugins have no renderer and are not a supported integration surface. The supported remote surface is the authenticated HTTP adapter, and trusted same-user clients can use the private canonical session protocol.
- The Python base image links SQLite 3.46.1 on the tested target. Existing Hermes safeguards detect it and select DELETE journal mode rather than vulnerable WAL behavior. No security guard was bypassed.
- A raw SIGTERM drains gracefully and exits 1 under the existing gateway external-stop policy. Planned stops/SIGINT use the clean-stop verdict. Container stop tests verify the deliberate code; they do not equate it with an abrupt kill.
- Tools requiring Chromium, Node, FFmpeg, system services, additional SDKs or messaging platforms need explicit optional installation or a compatible external plugin. No Docker socket or host-wide privileged access is supplied.
