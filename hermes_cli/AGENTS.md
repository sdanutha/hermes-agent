# Headless management CLI

Applies with the root guide. `main.py` handles profile and environment bootstrap; `core_commands.py` composes management commands. `hermes_core.py` launches the owner and health check. No REPL, TUI, dashboard or desktop entry point is supported.

Finite `chat -q` submissions use the canonical gateway authority; never create an independent execution owner in the CLI. Keep reusable file, config, provider and text helpers in `runtime_*.py`, and import their defining modules directly. Do not restore imports of removed `cli.py` or UI mixins.

`gateway.py` remains the process/service-management facade. Preserve its process identity checks and profile ownership. Never infer process identity from an argv substring.

Skills enter the normal user message path. Changes to prompt-affecting config, skills or tools apply to new sessions; preserve per-conversation prompt caching. Plugins extend the existing registry and loader.

## Config system (`hermes_cli/config.py`)

- **config.yaml option:** add to `DEFAULT_CONFIG`. Bump `_config_version` ONLY to actively
  migrate/transform existing config (rename keys, restructure); new keys deep-merge automatically.
  Top-level sections (non-exhaustive): `model, agent, terminal, compression, display, stt, tts,
memory, security, delegation, smart_model_routing, checkpoints, auxiliary, curator, skills,
gateway, logging, cron, profiles, plugins, honcho`. `auxiliary` = per-task side-LLM overrides
  (`agent/AGENTS.md`); `curator` = `enabled, interval_hours, min_idle_hours, stale_after_days,
archive_after_days, backup.*`.
- **.env = SECRETS ONLY** (keys, tokens, passwords): add to `OPTIONAL_ENV_VARS` with
  `{"description", "prompt", "url", "password": True, "category": provider|tool|messaging|setting}`.
  Non-secret settings go in config.yaml; if internal code needs an env mirror, bridge it in code
  (`gateway_timeout`; `terminal.cwd` → `TERMINAL_CWD`). `MESSAGING_CWD` is removed and `TERMINAL_CWD`
  in `.env` is deprecated — the loader warns; canonical is `terminal.cwd`. `hermes config
set/get/unset <NAME>` route any bare name registered in `OPTIONAL_ENV_VARS` / `_EXTRA_ENV_KEYS`
  (or carrying a `setup_hidden_env` platform suffix) to `.env` via `config_env_routing.py` — the
  file the platform setup flows write — never to the top level of config.yaml.
- **One writer seam.** Every write of a `config.yaml` (main or profile) goes through
  `hermes_cli.config.atomic_config_write` (refuses deletion by omission) or the explicit
  `atomic_config_replace` full-state path (→ `utils.atomic_roundtrip_yaml_save`, ruamel
  round-trip): comments, key order, quoting and blank lines survive, and the fail-closed unreadable-file
  guard runs first. Deliberate `pop()`/unset/migration paths use `atomic_config_replace`; additive
  writers stay on `atomic_config_write`. Never call
  `atomic_yaml_write` / `yaml.dump` / `yaml.safe_dump` on a config path — `scripts/check_config_yaml_writers.py`
  (CI lint) rejects it, and `tests/hermes_cli/test_config_yaml_comment_preservation.py` guards each
  path (#92554). The commented example blocks are appended only when the file is created.
- **Three loaders — know which you're in:** `load_config()` (headless CLI); `load_config()`
  (`hermes tools/setup`, most subcommands, `hermes_cli/config.py`, merges `DEFAULT_CONFIG`);
  `hermes_cli/config_effective.py::load_user_config_effective()` (gateway runtime via
  `gateway/run.py::_load_gateway_config`, TUI gateway `_load_cfg`, cron, `hermes send`, doctor,
  `hermes_time`/`hermes_logging`: user file + managed overlay + `${VAR}` expansion + model-key
  canon, NO defaults — for presence-sensitive readers). If the CLI sees a key and the gateway
  doesn't (or vice versa), you're on the wrong loader — check `DEFAULT_CONFIG` coverage. Never
  hand-roll raw-read → overlay → expand; `read_user_config_raw` is for write-back round-trips only.
- **Every `DEFAULT_CONFIG` key has a runtime reader, and every reader a registry entry.** Both drift
  modes are silent: a registered key nothing reads (a knob that does nothing) and a reader of a key
  never registered (never shown, never migrated; new roots also go in `_KNOWN_ROOT_KEYS`). For a new
  key: `rg -n '"<key>"' hermes_cli/config_defaults.py` AND `rg -n '<key>' --glob '!tests' .` both hit,
  and one invariant test sets it in a temp `config.yaml` and asserts the behaviour through the loader
  the consuming surface uses. Under multiplex the process env never overrides a profile's YAML.
- **Working directory:** CLI uses `os.getcwd()`; messaging uses `terminal.cwd`, bridged to
  `TERMINAL_CWD` for child tools.

## Profiles (multi-instance)

`_apply_profile_override()` in `hermes_cli/main.py` sets `HERMES_HOME` before any module import for
single-profile commands (`hermes -p x <cmd>`), so there `get_hermes_home()` scopes to the active
profile. The multiplex gateway and the Desktop/dashboard `serve` backend instead bind the active
profile per activity via a contextvar override while `os.environ["HERMES_HOME"]` keeps the launch
profile — a module constant or import-time read there freezes to the launch profile (rules in
root). Profiles are independent
islands by design — no live config inheritance; `--clone` copies at creation, minus messaging
channels (`profile_channels.py`: ownership-based inventory evaluated in the SOURCE's plugin scope —
adapter-declared keys + canonical/alias prefixes + `GATEWAY_ALLOW*`/`GATEWAY_RELAY_*`; prefixes shared
with tools (`TWILIO_`/`EMAIL_`, plus a plugin platform's `shared_env_prefixes`) are stripped only when the source runs that adapter; never a hand
list; a platform that left core keeps its ownership from its `LEFT_CORE` row while the plugin is absent). `--clone-channels` opts in and its live-multiplexer refusal lives in `create_profile` (CLI, REST
and TUI all go through it). Clones are built in `profiles/.<name>.staging-<pid>` (hidden → invisible to
`_iter_named_profile_dirs` and the hot-serve rescan) and published by one `os.rename` after the strip;
symlinked `.env`/`config.yaml` are materialized first so a clone never writes through to its source. Multiplex
(`gateway.multiplex_profiles`) secret-scope rules: `gateway/AGENTS.md`. The served set is
`profiles.py::profiles_to_serve(multiplex=True)` = default + every live dir under `profiles/` — live =
carries an identity marker (`hermes_constants.named_profile_has_identity`: `config.yaml`/`.env`/`SOUL.md`/
`profile.yaml`/`auth.json`/`state.db`) and is not tombstoned. A marker-less dir (cron/log side-effect
shell, stray infrastructure dir) is never listed, served, ticked, `.env`-backfilled or resolvable via `-p`
(#95188, #99392); `profile create` replaces it only when it is also tombstoned (a live marker-less dir may hold user
files — fail closed, never rmtree). A dangling symlinked marker still counts as identity (`is_symlink()`).
`tools/bot_mode_probe._roster` (Bot Mode teammate roster, `bot_relay.deliver` target check) applies the same
predicate. There is no allowlist (`gateway.multiplex_profile_allowlist` was retired in config v43).
Enumeration is a pure read: never `mkdir` a profile home from a served path (`SessionDB`, logging,
cron all go through `mkdir_under_hermes_home` / `_ensure_cron_dir`, which refuse a deleted or
missing named profile, #94590). Process-global per-profile slots (MCP discovery in `mcp_startup.py`,
tool registry overlays) key on `hermes_constants.hermes_home_key()`, never a single flag.
`gateway.multiplex_profiles` defaults to **on**, but `GatewayConfig` keeps an unset flag `None` and
`gateway_multiplex_mode.resolve_multiplex_mode` settles it once per boot (called from
`load_gateway_config_for_runner`): default profile, >= 2 profiles, no standalone secondary gateway,
no preflight blocker, migratable host → `True`; else `False` + a logged reason. Explicit values pass
through. CLI/dashboard readers use `default_gateway_multiplexes` (live `served_profiles` record, then
the explicit flag) — never the merged default, which would guess a verdict only the gateway makes.
Migration from per-profile gateways: `hermes_cli/gateway_migrate.py` (`hermes gateway migrate
--multiplex`, the only mode — `--standalone` is deleted and a per-profile fleet is not a supported
target; table-driven `_PREFLIGHT_CHECKS`; manifest `<default>/gateway_migration.json` = UNFINISHED,
a re-run resumes from it; a named profile's `gateway install|start|run` refuse without `--force` via
`gateway.py::_named_profile_refused_under_multiplexer`, dashboard twin
`web_server_gateway.py::multiplexed_profile_refusal`);
`update_cmd_fleet_verify._verify_fleet_after_update` calls `maybe_auto_migrate_after_update` on the success
path only; `gateway_migrate_guards.py` holds the auto-path-only refusals (table `_AUTO_MIGRATION_GUARDS`:
other service domain / UNIX user / HERMES_HOME outside `profiles/` — notices for the explicit command,
blockers for the hook) and the `gateway.auto_multiplex_migration` opt-out (#109954). Blockers reuse `GatewayRunner._adapter_credential_fingerprint` and `platform_binds_port`;
"has a `/p/<profile>/` ingress" is the adapter class attribute `serves_profile_prefix` — set it on a
new HTTP-inbound adapter when it answers the prefix, never extend a list here.
`hermes gateway restart` for a gateway Hermes did not install (custom launchd agent / unit running
`gateway run --external-supervisor`): `gateway_supervised_restart.py` — the gateway's SELF-declared
supervisor (control-socket `identify` answering anything but `manual`, OR the argv marker) decides; hand back via SIGUSR1 and wait
for a fresh supervised PID, never stop + foreground `run_gateway` (that stamps the CLI's PID and wedges
every KeepAlive respawn, #110637).

Service installs are a matrix, not a unit file: `gateway.py::generate_systemd_unit(system=,
run_as_user=)` (systemd unit generation / `systemd_unit_is_current` / `refresh_systemd_unit_if_needed` are PM-aware facade owners; user unit AND `--system` unit with `User=`; an unresolvable `User=` is a blocker,
never a dir-owner fallback), `gateway_launchd.py::generate_launchd_plist` (`gui/<uid>` then `user/<uid>` domains, never a
`~/Library/LaunchAgents` glob; the whole launchd backend — plist refresh, `launchctl` bootstrap/kickstart,
`launchd_start/stop/restart/status`, detached-process degrade — lives in that sibling, with the domain cache
`_resolved_launchd_domain` staying a facade global), Windows Scheduled Task and the Desktop-spawned backend all carry the
profile's `HERMES_HOME` (and `HOME` for the service user) explicitly — a supervisor starts with an
empty environment, so the env override that makes `-p` work interactively does not exist there. A
change to install/restart/status regenerates and diffs every kind; both user and system units are
recorded when both exist. Process liveness is `(pid, start_time)` or the canonical matchers
(`gateway.status.live_gateway_pid_for_home`), never bare PID existence.

## Nous free tier (`hermes_cli/anon_auth.py`)

`anon_auth.run_sign_in()` is the one sign-in composition: it yields `SignInState`s that carry their
own `.copy` / `.copy_terminal`, persists only after a completed promotion AND a token grant, and runs
`settle_after_upgrade` exactly once per completion. `hermes auth upgrade`, the CLI and gateway
`/login` and the desktop poller are renderers over it; the desktop's plain device-code login
(`_nous_plain_poller`) stays a separate path. Long form: `cli-internals.md` § Nous free tier sign-in.
