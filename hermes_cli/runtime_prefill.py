"""Runtime prefill and request-option parsing."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any
from hermes_constants import get_hermes_home
logger = logging.getLogger(__name__)

def _load_prefill_messages(file_path: str, base_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load prefill messages (JSON array) from *file_path*; missing/empty -> [].

    Relative paths resolve against *base_dir*, defaulting to the CLI's hermes home.
    """
    if not file_path:
        return []
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = (base_dir or get_hermes_home()) / path
    if not path.exists():
        logger.warning("Prefill messages file not found: %s", path)
        return []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("Prefill messages file must contain a JSON array: %s", path)
            return []
        return data
    except Exception as e:
        logger.warning("Failed to load prefill messages from %s: %s", path, e)
        return []

def _resolve_prefill_messages_file(config: dict[str, Any]) -> str:
    """Prefill file path: env, then top-level ``prefill_messages_file``, then legacy ``agent.*``."""
    from agent.session_client_env import session_env
    agent_cfg = config.get("agent", {})
    return (
        session_env("HERMES_PREFILL_MESSAGES_FILE").strip()
        or str(config.get("prefill_messages_file", "") or "").strip()
        or (str(agent_cfg.get("prefill_messages_file", "") or "").strip() if isinstance(agent_cfg, dict) else "")
    )

def _parse_reasoning_config(effort) -> dict | None:
    """Parse a reasoning effort level (string or YAML bool; ``false``/``off`` = disabled)."""
    from hermes_constants import parse_reasoning_effort
    result = parse_reasoning_effort(effort)
    if effort and str(effort).strip() and result is None:
        logger.warning("Unknown reasoning_effort '%s', using default (medium)", effort)
    return result

def _parse_service_tier_config(raw: str) -> str | None:
    """Parse a persisted fast-mode preference: None, "priority", "ultrafast", "auto", or "cold"."""
    from agent.fast_mode import NORMAL_TIER_WORDS, parse_service_tier

    tier = parse_service_tier(raw)
    if tier is None and str(raw or "").strip().lower() not in NORMAL_TIER_WORDS:
        logger.warning("Unknown service_tier '%s', ignoring", raw)
    return tier
