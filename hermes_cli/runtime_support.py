"""Runtime provider readiness, model consent and checkout identity (no frontend startup)."""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

def _require_tty(command_name: str) -> None:
    """Exit 1 if stdin is not a terminal: curses/input() prompts spin at 100% CPU on a pipe."""
    if not sys.stdin.isatty():
        print(
            f"Error: 'hermes {command_name}' requires an interactive terminal.\n"
            f"It cannot be run through a pipe or non-interactive subprocess.\n"
            f"Run it directly in your terminal instead.",
            file=sys.stderr,
        )
        sys.exit(1)

def _read_packed_ref(common_dir: Path, ref: str) -> str | None:
    """Look up a ref in .git/packed-refs without spawning git.

    packed-refs lines look like ``<sha> <ref>`` with optional ``^<sha>``
    peel lines and ``#``-prefixed comments / ``# pack-refs with:`` header.
    """
    try:
        text = (common_dir / "packed-refs").read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if not line or line.startswith("#") or line.startswith("^"):
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1].strip() == ref:
            return parts[0].strip()
    return None

def _read_git_revision_fingerprint(repo_root: Path) -> str | None:
    """Return a cheap checkout fingerprint without spawning git."""
    git_dir = repo_root / ".git"
    try:
        if git_dir.is_file():
            for line in git_dir.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                key, _, value = line.partition(":")
                if key.strip() == "gitdir" and value.strip():
                    git_dir = (repo_root / value.strip()).resolve()
                    break
        # Worktrees point HEAD at a per-worktree gitdir but pack their refs
        # in the main repo's gitdir (referenced via ``commondir``). Resolve
        # that up front so packed-refs lookups hit the right file.
        common_dir = git_dir
        commondir_file = git_dir / "commondir"
        if commondir_file.exists():
            try:
                rel = commondir_file.read_text(encoding="utf-8-sig", errors="replace").strip()
                if rel:
                    common_dir = (git_dir / rel).resolve()
            except OSError:
                pass
        head = (git_dir / "HEAD").read_text(encoding="utf-8-sig", errors="replace").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            # Loose refs may live in the worktree gitdir OR the common dir
            # (branches created via `git worktree add` typically live in the
            # common dir's refs/heads/).
            for candidate in (git_dir, common_dir):
                ref_file = candidate / ref
                if ref_file.exists():
                    return f"git:{ref}:{ref_file.read_text(encoding='utf-8-sig', errors='replace').strip()}"
            packed_sha = _read_packed_ref(common_dir, ref)
            if packed_sha:
                return f"git:{ref}:{packed_sha}"
            # Ref name is known but unresolved — still stable across launches,
            # and the version/release fallback in the caller will invalidate
            # after `hermes update`.
            return f"git:{ref}:unresolved"
        return f"git:HEAD:{head}"
    except OSError:
        return None

def _dotenv_has_provider_key(env_file: Path, provider_env_vars: set) -> bool:
    """True if ~/.hermes/.env assigns a non-empty value to any provider key."""
    if not env_file.exists():
        return False
    try:
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            line = line.removeprefix("export ")
            key, _, val = line.partition("=")
            if key.strip() in provider_env_vars and val.strip().strip("'\""):
                return True
    except Exception:
        pass
    return False

def _auth_store_logged_in(auth_file: Path, registry, strict_profile_scope: bool) -> bool:
    """True if auth.json's active provider is logged in (api_key providers ignored under strict scope)."""
    from hermes_cli.auth import get_auth_status

    if not auth_file.exists():
        return False
    try:
        auth = json.loads(auth_file.read_text(encoding="utf-8-sig"))
        active = auth.get("active_provider")
        active_config = registry.get(str(active or "").strip().lower())
        if active and not (
            strict_profile_scope and active_config and active_config.auth_type == "api_key"
        ):
            return bool(get_auth_status(active).get("logged_in"))
    except Exception:
        pass
    return False

def _has_any_provider_configured(*, strict_profile_scope: bool = False) -> bool:
    """Check if at least one inference provider is usable. Never creates one: the Nous free tier
    counts only once its identity exists, and the boot bootstrap (``hermes_cli.free_tier_bootstrap``)
    creates it at boot (a first connector use can too); ``cmd_chat`` runs the bootstrap before asking.

    ``strict_profile_scope``: the caller has bound a NAMED profile's home and
    secret scope and wants an answer for that profile only — launch-process
    env and host-wide fallbacks (gh auth, Claude Code credentials) must not
    make it appear ready. Unscoped callers keep the legacy behavior.
    """
    from hermes_cli.config import DEFAULT_CONFIG, get_env_path, get_hermes_home, load_config
    from hermes_cli.auth import PROVIDER_REGISTRY, get_auth_status

    cfg = load_config()
    model_cfg = cfg.get("model")
    _model_name = model_cfg if isinstance(model_cfg, str) else ""
    if isinstance(model_cfg, dict):
        _model_name = model_cfg.get("default") or ""
        if isinstance(_model_name, dict):
            from hermes_cli.config import split_model_config_default
            _model_name, _ = split_model_config_default(_model_name)
    _model_name = str(_model_name).strip()
    # "Explicitly configured" = model differs from the hardcoded default; gates
    # Claude Code credentials so they don't skip setup on a fresh install.
    _has_hermes_config = _model_name and _model_name != DEFAULT_CONFIG.get("model", "")

    # Env vars (.env or shell). OPENAI_BASE_URL alone counts — local models
    # (vLLM, llama.cpp) often need no API key.
    provider_env_vars = {
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "OPENAI_BASE_URL",
    }
    for pconfig in PROVIDER_REGISTRY.values():
        if pconfig.auth_type == "api_key":
            provider_env_vars.update(pconfig.api_key_env_vars)
    if strict_profile_scope:
        from agent.secret_scope import current_secret_scope

        read_provider_env = (current_secret_scope() or {}).get
    else:
        read_provider_env = os.getenv
    if any(read_provider_env(v) for v in provider_env_vars):
        return True
    if _dotenv_has_provider_key(get_env_path(), provider_env_vars):
        return True

    # Cheap on-disk checks (auth.json, config.yaml) first: the PROVIDER_REGISTRY
    # sweep below spawns subprocesses (gh) and can take 15-20s — long enough
    # that desktop setup.status calls time out.
    if _auth_store_logged_in(get_hermes_home() / "auth.json", PROVIDER_REGISTRY, strict_profile_scope):
        return True

    # model as a dict with provider/base_url/api_key means setup ran (fresh
    # installs have a plain string); also covers custom endpoints kept in config.
    if isinstance(model_cfg, dict) and any(
        (model_cfg.get(k) or "").strip() for k in ("provider", "base_url", "api_key")
    ):
        return True

    # Provider-specific auth fallbacks (e.g. Copilot via gh auth).
    if not strict_profile_scope:
        try:
            if any(
                (status := get_auth_status(pid)).get("logged_in")
                and status.get("key_source") != "keyless"
                for pid, pconfig in PROVIDER_REGISTRY.items()
                if pconfig.auth_type == "api_key"
            ):
                return True
        except Exception:
            pass

    # Claude Code OAuth credentials count only once Hermes is explicitly
    # configured — having Claude Code installed isn't consent to use its tokens.
    if _has_hermes_config and not strict_profile_scope:
        try:
            from agent.anthropic_credentials import read_claude_code_credentials, is_claude_code_token_valid

            creds = read_claude_code_credentials()
            if creds and (
                is_claude_code_token_valid(creds) or creds.get("refreshToken")
            ):
                return True
        except Exception:
            pass

    return False

def _confirm_startup_expensive_model_override(args) -> None:
    """Guard startup -m/--provider overrides before the first API call."""
    explicit_model = (getattr(args, "model", None) or "").strip()
    explicit_provider = (getattr(args, "provider", None) or "").strip()
    if not explicit_model and not explicit_provider:
        return

    try:
        from hermes_cli.config import load_config
        from hermes_cli.model_selection_guards import (
            combined_message,
            selection_warnings,
        )
    except Exception as exc:
        logger.warning("startup model cost guard unavailable: %s", exc)
        return

    try:
        config = load_config()
    except Exception as exc:
        logger.warning("startup model cost guard could not load config: %s", exc)
        config = {}
    _dict = lambda v: v if isinstance(v, dict) else {}
    config = _dict(config)
    model_cfg = _dict(config.get("model"))
    security_cfg = _dict(config.get("security"))

    model = explicit_model or (model_cfg.get("default") or "").strip()
    if not model:
        return
    provider = (explicit_provider or model_cfg.get("provider") or "").strip()
    try:
        # Unified registry: cost guard + id-keyed guards (e.g. the
        # data-training-tier warning) all fire at startup too.
        warnings = selection_warnings(
            model,
            provider=provider,
            base_url=(model_cfg.get("base_url") or ""),
            api_key=(model_cfg.get("api_key") or ""),
        )
    except Exception as exc:
        logger.warning("startup model cost guard failed for %s/%s: %s", provider, model, exc)
        return
    if not warnings:
        return

    # Intentionally independent of --yolo / --accept-hooks: those approve local
    # command risk, not paid aggregator spend or a surprising provider route.
    is_interactive = sys.stdin.isatty()
    if not is_interactive and security_cfg.get("allow_data_training_tiers_noninteractive") is True:
        acknowledged = [w for w in warnings if w.kind == "data_policy"]
        if acknowledged:
            sys.stderr.write(combined_message(acknowledged) + "\n")
            sys.stderr.write(
                "Proceeding in non-interactive mode because "
                "security.allow_data_training_tiers_noninteractive is true.\n"
            )
            warnings = [w for w in warnings if w.kind != "data_policy"]
            if not warnings:
                return

    message = combined_message(warnings)
    if not is_interactive:
        sys.stderr.write(message + "\n")
        if any(warning.kind == "data_policy" for warning in warnings):
            sys.stderr.write(
                "To acknowledge data-training tiers for unattended runs, set "
                "security.allow_data_training_tiers_noninteractive to true "
                "in config.yaml.\n"
            )
        sys.stderr.write(
            "Refusing this startup model override in non-interactive mode. "
            "Run interactively and confirm if you intend to use it.\n"
        )
        raise SystemExit(1)

    sys.stderr.write(message + "\n")
    try:
        reply = input("Use this model for this invocation? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        reply = ""
    if reply not in {"y", "yes"}:
        sys.stderr.write("Model override cancelled.\n")
        raise SystemExit(1)

def _resolve_workspace_key() -> Optional[str]:
    """The current workspace identity for cwd-scoped resume.

    Git repo root when CWD is inside a repo (so all sessions across its
    subdirs/worktrees group together), else the CWD itself. Returns None when
    neither can be determined — callers fall back to the global MRU then.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return os.path.abspath(result.stdout.strip())
    except Exception:
        pass
    try:
        return os.getcwd()
    except Exception:
        return None

def _first_run_setup_guard(args):
    raise SystemExit("No provider configured. Set model.provider/model.default in config.yaml and supply its credentials.")
