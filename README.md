# Hermes Core Runtime

A headless Hermes backend for embedding agent execution in other systems. Python 3.14 is the supported runtime. Installation and startup require no Node.js, npm, Electron, or frontend build tools.

The agent loop, model providers/custom endpoints, tool registry/toolsets, skills, MCP clients, compatible runtime plugins, memory, compression, sessions, delegation, configuration/profiles, secrets, logging, approvals and scheduler remain. Electron Desktop, the browser dashboard, Ink TUI, presentation assets, bundled messaging plugins and documentation website build infrastructure have been removed from this branch.

The existing gateway is the execution owner: it owns profile-bound session authorities, persistent admissions, warm agents, cancellation, job ticking and graceful shutdown. `run_agent.py` and `agent/` own inference; `model_tools.py` and `tools/` own tool execution. `hermes_state*` owns persistence. The Python `tui_gateway/` package is retained because it contains shared session contracts, transports, callbacks and profile isolation machinery; it is backend code and runs without Ink. Some legacy Python CLI/admin helpers remain internal dependencies of these shared paths. No dashboard is mounted by the runtime listener.

Per-conversation prompts and tool schemas remain frozen by the existing session policy. New configuration, skills and tool selections should take effect in a new session. Profiles retain separate credentials, stores and activity scopes.

## Local installation

Use a checkout, not a wheel (the repository's wheel guard remains). Build dependencies through Hermes PM:

```sh
export HERMES_HOME="$PWD/.core-state"
export HERMES_RUNTIME_DIR="$PWD/.core-runtime"
mkdir -p "$HERMES_HOME"
chmod 700 "$HERMES_HOME"
python3.14 -m pm.build_env --source . --out .core-venv --extra runtime
.core-venv/bin/python -m hermes_core --help
```

Create `$HERMES_HOME/config.yaml` with your provider and model:

```yaml
model:
  provider: openrouter
  default: your-provider-model
terminal:
  backend: local
  cwd: /absolute/path/to/workspace
gateway:
  multiplex_profiles: false
```

Supply `OPENROUTER_API_KEY` through the environment or `$HERMES_HOME/.env` (mode 0600). Other providers use their existing credential names. Then:

```sh
.core-venv/bin/python -m hermes_core run
# In another shell with the same HERMES_HOME:
.core-venv/bin/python -m hermes_core health
```

`hermes-core` and `hermes` are non-interactive command entry points in a PM project environment. `python -m hermes_core` also works directly from the checkout. Management commands reuse existing implementations: `config`, `profile`, `cron`, `tools`, `skills`, `plugins`, `mcp`, `pm`, `sessions`, `logs`, `gateway`, and `doctor`. A one-shot `chat -q "task"` command remains for unattended workers. Some management subcommands are interactive setup wizards; use explicit arguments or edit configuration for automation. Bare interactive chat, Desktop, dashboard and TUI commands are not exposed by this entry point.

## Docker and Compose

Set `model.default` in [docker/core-config.yaml](docker/core-config.yaml). Create a local `.env` containing only runtime secrets:

```dotenv
OPENROUTER_API_KEY=replace-with-provider-key
API_SERVER_KEY=replace-with-a-random-secret-at-least-32-characters
```

Generate the API credential with `openssl rand -hex 32`; do not use the example value. Keep `.env` private and outside version control. It is excluded from the image build context.

```sh
docker build -t hermes-core:local .
docker compose up -d --build
docker compose exec hermes-core python -m hermes_core health
docker compose logs -f hermes-core
docker compose down
```

The image runs as UID/GID 10000 under `tini`, forwards SIGTERM, and gives the existing gateway time to settle agents, jobs, MCP and stores. Compose allows 90 seconds before forced termination. The health check queries the private live owner and fails while it is absent, starting or draining. Inference provider availability is a separate check and requires valid credentials.

The `hermes-state` volume at `/var/lib/hermes` contains config, profiles, secrets, sessions/state.db, memories, skills, plugins, logs and runtime metadata. Compose mounts the example config read-only; edit it on the host and restart. Config-writing commands require removing that read-only mount or using a writable config. Preserve this volume across upgrades; back it up while the runtime is stopped. Do not use `docker compose down -v` unless deliberately deleting state.

Terminal tools with `backend: local` execute **inside the container**, as the runtime user, with `/workspace` as cwd. `hermes-workspace` persists that workspace. Host paths are inaccessible unless explicitly mounted. There is no Docker socket mount or privileged mode. Bind mounts must be writable by UID 10000; a state home must be private (0700) for control-socket authentication. Optional Docker terminal backends need separate, explicit operator setup.

Mount extra skills at `/var/lib/hermes/skills/<name>` and compatible plugins at `/var/lib/hermes/plugins/<name>`; preserve their manifests and supporting files. Install Python/plugin extras through PM, preferably in a derived image for immutable deployments. Browser tools, `npx` MCP servers, JavaScript plugins and individual skills may need Chromium, Node.js or other executables; those are optional capabilities and are not installed by this image.

## External integration

Reuse the existing authenticated HTTP API; no new inference transport or user database was introduced. Compose enables it at `127.0.0.1:8642` on the host. The adapter refuses startup without a usable `API_SERVER_KEY`. Requests use `Authorization: Bearer <key>`. For remote deployments, put TLS and your network access policy in front of this listener; the key authorizes agent/tool execution.

```sh
# Export the same API_SERVER_KEY used by the service, without printing it.
python3 examples/core_client.py --health
python3 examples/core_client.py 'Read the files in /workspace and summarize them'
python3 examples/core_client.py --session SESSION_ID 'Continue the task'
python3 examples/core_client.py --stop RUN_ID
```

The [working client](examples/core_client.py) prints the admission receipt, follows SSE execution events, and fetches the final run record. `POST /v1/runs` creates/selects a session with `input` and optional `session_id`; `GET /v1/runs/{id}` returns status/results; `GET /v1/runs/{id}/events` observes activity; `POST /v1/runs/{id}/stop` cancels work. Session resources are under `/api/sessions`. `/v1/capabilities` describes the actual supported endpoints. Retain returned session/run identities; after an ambiguous disconnect inspect status rather than blindly resubmitting side effects.

For trusted same-user Python hosts, `hermes_cli.gateway_client.connect_gateway` uses the existing private control socket to obtain a one-use WebSocket ticket. Its `rpc` and `events` interfaces support `session.create`, `session.resume`, `prompt.submit`, replay, and generation-bound `session.interrupt`. This transport binds loopback only, rejects browser-origin connections, and never publishes a token from a page. It is not a public remote API.

## Runtime configuration

Use `config.yaml` for behavior and `.env`/runtime environment for secrets. Profiles are independent; set `HERMES_HOME` before starting a separate owner. Multiplexing remains available through `gateway.multiplex_profiles`, using the existing profile scopes. Do not mutate process environment per request to switch profiles.

- Providers: `model.provider`, `model.default`, and `model.base_url` for custom OpenAI-compatible endpoints. Existing provider catalogs, auth and auxiliary-model selection remain in `agent/` and `hermes_cli/`.
- Tools: configure `platform_toolsets` for the calling source (for example `api_server`); registry checks and security/approval policies still apply. A remote caller is capable of invoking every enabled tool.
- Skills: install beneath the profile's `skills/`; bundled skills remain under `skills/`. Skill instructions enter the normal user-message path.
- MCP: use `mcp_servers` entries with `command`/`args` for stdio or the existing HTTP configuration. The `runtime` extra includes the MCP SDK. Put server credentials in secret references. Additional executables belong in the container or a derived image.
- Plugins: compatible Python tool/provider/memory/context plugins use existing discovery and loading. Bundled messaging adapters are absent; UI plugins have no renderer. Mounted plugins may declare additional dependencies and need explicit PM preparation.
- Jobs: `python -m hermes_core cron --help` manages schedules. The owner ticks jobs even without messaging adapters. Use local/no-delivery jobs; outputs and run records remain in profile state. Cron intentionally skips memory-provider recall/commit, as before.

Developer and operational reference material remains as Markdown under `website/docs/`, with no website build system. Historical references to removed presentation surfaces describe the upstream architecture, not available Core products. See [validation notes](docs/core-runtime-validation.md) for checks and known limits on this branch.

SIGTERM follows the existing owner's external-stop policy: it drains and releases state but exits 1 unless a planned-stop marker was supplied. SIGINT/planned owner stops exit cleanly. Container supervisors should use readiness and normal stop semantics rather than treating every SIGTERM exit code as data loss.
