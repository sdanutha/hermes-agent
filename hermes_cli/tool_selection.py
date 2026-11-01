"""Profile-scoped tool selection and persistence, without provider setup menus."""
from __future__ import annotations

import logging

import os

from hermes_cli.config import cfg_get, save_config, get_env_value

from hermes_cli.platforms import PLATFORMS as _PLATFORMS_REGISTRY

from hermes_cli.toolset_scope import _TOOLSET_PLATFORM_RESTRICTIONS, toolset_allowed_for_platform as _toolset_allowed_for_platform

from hermes_cli.toolset_validation import parse_platform_toolsets_value

logger = logging.getLogger(__name__)

_warned_invalid_platform_toolsets: set[str] = set()

PLATFORMS = {k: {"label": info.label, "default_toolset": info.default_toolset} for k, info in _PLATFORMS_REGISTRY.items()}

from toolsets import TOOLSETS
CONFIGURABLE_TOOLSETS = [(name, name, spec["description"]) for name, spec in TOOLSETS.items() if not name.startswith("hermes-")]

_DEFAULT_OFF_TOOLSETS = {"spotify", "discord", "discord_admin", "video", "video_gen", "x_search", "a2a", "kanban"}

def _xai_credentials_present() -> bool:
    """Cheap offline check for xAI credentials (auth store + env only); the runtime ``check_fn`` still gates
    schema registration if creds expire. Also used by ``provider_readiness_status`` for ``xai_grok`` rows."""
    try:
        from hermes_cli.auth import _read_xai_oauth_tokens
        _read_xai_oauth_tokens()
        return True
    except Exception:
        pass
    if str(get_env_value("XAI_API_KEY") or "").strip():
        return True
    try:
        from agent.secret_scope import get_secret
    except ImportError:  # pragma: no cover — secret_scope is in-repo
        get_secret = os.environ.get
    return bool(str(get_secret("XAI_API_KEY") or "").strip())

_gateway_plugin_toolsets = None

def _projected_plugin_toolsets():
    provider = _gateway_plugin_toolsets
    return provider() if provider is not None else None

def _get_plugin_toolset_keys() -> set:
    """Return the set of toolset keys provided by plugins."""
    try:
        if (projected := _projected_plugin_toolsets()) is not None:
            return {key for key, _, _ in projected[0]}
        # Non-blocking on CLI startup: while background discovery is still importing, serve last
        # launch's persisted key set instead of joining the discovery thread.
        from hermes_cli.plugins import get_plugin_toolset_keys_nowait
        return get_plugin_toolset_keys_nowait()
    except Exception:
        return set()

def _platform_default_toolset(platform: str) -> str:
    """Composite toolset a platform falls back to (plugin platforms derive ``hermes-<platform>``)."""
    return PLATFORMS[platform]["default_toolset"] if platform in PLATFORMS else f"hermes-{platform}"

def _cfg_section(config: dict, key: str) -> dict:
    """Return ``config[key]`` as a dict, replacing a missing or non-dict value with ``{}``."""
    section = config.setdefault(key, {})
    if not isinstance(section, dict):
        section = {}
        config[key] = section
    return section

def enabled_mcp_server_names(config: dict) -> set[str]:
    """MCP servers globally enabled in config.yaml or by a plugin (shared by platform + cron resolvers). Enabled
    unless ``enabled`` is explicitly falsey; portable-plugin servers (in-memory) count — enabling the plugin is
    the opt-in."""
    from tools.mcp_tool_common import mcp_server_enabled

    mcp_servers = (config or {}).get("mcp_servers") or {}
    names = {
        str(name) for name, server_cfg in mcp_servers.items()
        if isinstance(server_cfg, dict) and mcp_server_enabled(server_cfg)
    }
    try:
        projected = _projected_plugin_toolsets()
        if projected is None:
            from hermes_cli.plugins import get_portable_mcp_server_names_nowait
        portable = projected[1] if projected is not None else get_portable_mcp_server_names_nowait()
        names |= portable - set(mcp_servers)  # native config wins on a name collision (mirrors _load_mcp_config)
    except Exception:
        logger.debug("Failed to include portable MCP servers", exc_info=True)
    return names

_RECENTLY_SHIPPED_TOOLSETS: frozenset = frozenset()

def _enable_recently_shipped_toolsets(enabled_toolsets: set[str], config: dict, platform: str) -> None:
    """Turn on toolsets that shipped after this platform's saved list (mutates ``enabled_toolsets``). Both "no"s
    outlive this: unchecking records ``known_builtin_toolsets`` (declined), and ``agent.disabled_toolsets`` is
    subtracted last in :func:`_get_platform_tools`."""
    from toolsets import resolve_toolset

    offered = (config.get("known_builtin_toolsets") or {}).get(platform)
    declined = {str(ts) for ts in offered} if isinstance(offered, list) else set()
    default_ts = _platform_default_toolset(platform)
    composite_tools = None
    for ts_key in sorted(_RECENTLY_SHIPPED_TOOLSETS):
        if ts_key in enabled_toolsets or ts_key in declined or not _toolset_allowed_for_platform(ts_key, platform):
            continue
        # Only enable where staying on the composite would have enabled it anyway; deliberately narrow
        # composites (hermes-acp, hermes-webhook) stay narrow.
        ts_tools = set(resolve_toolset(ts_key, include_registry=False))
        if composite_tools is None:
            composite_tools = set(resolve_toolset(default_ts))
        if not ts_tools or not ts_tools.issubset(composite_tools):
            continue
        enabled_toolsets.add(ts_key)

def _configurable_subset_of(tool_names: set[str], platform: str) -> set[str]:
    """Configurable toolsets whose STATIC membership is within ``tool_names`` (``include_registry=False``: a
    runtime-registered tool the composite never listed must not drop the whole toolset)."""
    from toolsets import resolve_toolset

    return {
        ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS if _toolset_allowed_for_platform(ts_key, platform)
        and (ts_tools := set(resolve_toolset(ts_key, include_registry=False))) and ts_tools <= tool_names}

def _default_off_toolsets(platform: str, explicitly_configured: bool) -> set[str]:
    """Toolsets to strip from an implicit (composite-derived) enable set. A platform named after a default-off
    toolset keeps it, except platform-restricted ones (``discord`` on discord stays OFF).
    Platform-native default-off toolsets (``discord`` on discord) are off for unconfigured platforms as a
    security opt-in — an explicitly saved list IS that opt-in and lets them through."""
    default_off = set(_DEFAULT_OFF_TOOLSETS)
    if platform in default_off and platform not in _TOOLSET_PLATFORM_RESTRICTIONS:
        default_off.remove(platform)
    if explicitly_configured:
        default_off -= {ts for ts in default_off if platform in (_TOOLSET_PLATFORM_RESTRICTIONS.get(ts) or ())}
    return default_off

def _configurable_keys() -> set[str]:
    return {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS}

def _platform_default_keys() -> set[str]:
    return {p["default_toolset"] for p in PLATFORMS.values()}

def _explicit_toolsets(
    toolset_names: list[str], explicit_known_keys: set[str], config: dict, platform: str,
    explicitly_configured: bool) -> set[str]:
    """Enabled set when the saved list names configurable/plugin keys directly (subset inference over
    ``hermes-cli`` would re-enable disabled toolsets). A mixed list (``[hermes-cli, discord]``) still expands the
    composite; _DEFAULT_OFF_TOOLSETS applies to that implicit expansion only."""
    from toolsets import resolve_toolset, TOOLSETS

    enabled = {ts for ts in toolset_names if ts in explicit_known_keys and _toolset_allowed_for_platform(ts, platform)}
    composite_tools = {
        t for ts_name in toolset_names if ts_name not in explicit_known_keys and ts_name in TOOLSETS
        for t in resolve_toolset(ts_name)}
    if composite_tools:
        enabled |= _configurable_subset_of(composite_tools, platform) - _default_off_toolsets(platform, explicitly_configured)
    _enable_recently_shipped_toolsets(enabled, config, platform)
    return enabled

def _composite_toolsets(toolset_names: list[str], platform: str, explicitly_configured: bool) -> set[str]:
    """Enabled set inferred from composite names by reverse-mapping tool names (only while no explicit list is
    saved). ``x_search`` is not in any composite, so inject it when xAI creds exist and exempt it from default-off."""
    from toolsets import resolve_toolset

    all_tool_names = {t for ts_name in toolset_names for t in resolve_toolset(ts_name)}
    enabled = _configurable_subset_of(all_tool_names, platform)
    default_off = _default_off_toolsets(platform, explicitly_configured)
    if _toolset_allowed_for_platform("x_search", platform) and _xai_credentials_present():
        enabled.add("x_search")
        default_off.discard("x_search")
    return enabled - default_off

def _enabled_plugin_toolsets(config: dict, platform: str, toolset_names: list[str], plugin_ts_keys: set[str]) -> set[str]:
    """Plugin toolsets: on by default unless default-off (catalog spotify) or "known" for this platform
    (``known_plugin_toolsets``, written on every save) and absent from the saved list."""
    known_for_platform = set((config.get("known_plugin_toolsets", {}) or {}).get(platform, []) or [])
    return {
        pts for pts in plugin_ts_keys
        if pts in toolset_names or (pts not in _DEFAULT_OFF_TOOLSETS and pts not in known_for_platform)
    }

def _context_engine_active(config: dict) -> bool:
    context_cfg = config.get("context") or {}
    name = str(context_cfg.get("engine") or "compressor").strip().lower() if isinstance(context_cfg, dict) else "compressor"
    return bool(name) and name != "compressor"

def _coerce_platform_toolsets_value(value, platform: str):
    """Read a list-literal string saved for ``platform_toolsets.<platform>`` as the list it encodes.

    The parser is shared with ``hermes doctor`` and ``hermes plugins`` (``toolset_validation``
    ``parse_platform_toolsets_value``) so every surface agrees on the user's selection (#115866).
    Any other non-list value is warned about once (naming the expected shape) and left as-is, so
    the default fallback below is loud rather than silent.
    """
    if value is None:
        return None
    parsed = parse_platform_toolsets_value(value)
    if parsed is not None:
        return parsed
    if platform not in _warned_invalid_platform_toolsets:
        _warned_invalid_platform_toolsets.add(platform)
        logger.warning(
            "platform_toolsets.%s is %r, expected a YAML list of toolset names "
            "(e.g. [terminal, file, web]) - falling back to the platform default. "
            "Run `hermes tools` to reconfigure.", platform, value)
    return value

def _platform_toolsets_explicitly_saved(config: dict, platform: str) -> bool:
    """True when ``platform_toolsets.<platform>`` holds an explicitly saved LIST (even ``[]``).

    ``_get_platform_tools``'s ``explicitly_configured`` flag without re-running the resolver
    (post-coercion, so a list-literal string counts too): an unset key or a non-list value
    falls back to the platform default. Callers use this to tell an explicit zero-tool
    selection (fail closed, #82010) from an absent one ("no restriction").
    """
    platform_toolsets = config.get("platform_toolsets") or {}
    raw = platform_toolsets.get(platform)
    return isinstance(_coerce_platform_toolsets_value(raw, platform), list)

def _get_platform_tools(config: dict, platform: str, *, include_default_mcp_servers: bool = True) -> set[str]:
    """Resolve which individual toolset names are enabled for a platform."""
    platform_toolsets = config.get("platform_toolsets") or {}
    toolset_names = _coerce_platform_toolsets_value(platform_toolsets.get(platform), platform)
    # An explicitly saved list (even a composite like ``hermes-discord``) is an opt-in to the platform's
    # native default-off toolsets — see _default_off_toolsets.
    # Track whether the user explicitly saved a toolset list for this platform (vs. falling back to the
    # platform default). See #35527.
    explicitly_configured = isinstance(toolset_names, list)
    if not explicitly_configured:
        toolset_names = [_platform_default_toolset(platform)]
    # YAML may parse bare numeric names (``12306:``) as int; normalise so sorted() never mixes types.
    toolset_names = [str(ts) for ts in toolset_names]

    # Expand legacy toolset aliases.  Older Hermes versions and clients used
    # bare ``"hermes"`` as a composite toolset covering both the CLI and the
    # API-server surface.  Modern code expects ``"hermes-cli"`` (and
    # ``"hermes-api-server"`` for the HTTP endpoint), so configs persisted
    # by those older versions still carry the legacy name; without expansion
    # ``resolve_toolset("hermes")`` returns ``[]`` — all tools silently
    # disappear.
    _LEGACY_TOOLSET_ALIASES: dict = {
        "hermes": ("hermes-cli", "hermes-api-server"),
    }
    expanded: list = []
    for name in toolset_names:
        aliases = _LEGACY_TOOLSET_ALIASES.get(name)
        if aliases:
            expanded.extend(aliases)
        else:
            expanded.append(name)
    toolset_names = expanded

    configurable_keys = _configurable_keys()
    plugin_ts_keys = _get_plugin_toolset_keys()
    platform_default_keys = _platform_default_keys()
    # Plugin toolsets are first-class on a saved list: ``[hermes-cli, a2a]`` must survive filtering.
    # Plugin-provided toolsets are first-class on a platform-toolsets list — explicit config like
    # ``[hermes-cli, a2a]`` must survive filtering just like a built-in configurable toolset would. See
    # issue #81163.
    explicit_known_keys = configurable_keys | plugin_ts_keys

    if any(ts in explicit_known_keys for ts in toolset_names):
        enabled_toolsets = _explicit_toolsets(toolset_names, explicit_known_keys, config, platform, explicitly_configured)
    else:
        enabled_toolsets = _composite_toolsets(toolset_names, platform, explicitly_configured)

    _recover_platform_native_toolsets(enabled_toolsets, platform, skip=configurable_keys | plugin_ts_keys | platform_default_keys)
    if plugin_ts_keys:
        enabled_toolsets |= _enabled_plugin_toolsets(config, platform, toolset_names, plugin_ts_keys)

    # Context-engine tools are runtime-provided, not in any static composite: keep them for a non-default
    # engine even after an explicit save. An explicit EMPTY list means none, unless ``context_engine`` is added by hand.
    if _context_engine_active(config) and not (explicitly_configured and not toolset_names):
        enabled_toolsets.add("context_engine")

    # Explicit non-configurable entries (custom toolsets, MCP server names) pass through.
    explicit_passthrough = {ts for ts in toolset_names if ts not in explicit_known_keys and ts not in platform_default_keys}
    enabled_toolsets |= _merge_mcp_servers(config, toolset_names, explicit_passthrough, include_default_mcp_servers)

    # Legacy profile opt-in is a fallback only. A saved platform list (even
    # empty) is authoritative, so a later disable cannot silently re-enable it.
    if not explicitly_configured and "kanban" in (config.get("toolsets") or []):
        enabled_toolsets.add("kanban")

    # agent.disabled_toolsets is a global suppression list (#86661) and runs LAST so it overrides everything
    # above. It may arrive as a JSON-array string ("['memory']") from `hermes config set` or a JSON-mode editor.
    disabled_toolsets = (config.get("agent") or {}).get("disabled_toolsets")
    if disabled_toolsets:
        from agent.skill_utils import parse_config_string_list
        disabled_names = [name.strip() for name in parse_config_string_list(disabled_toolsets) if name.strip()]
        enabled_toolsets = _prune_toolsets_stripped_by_disabled(enabled_toolsets, disabled_names)

    if explicitly_configured and toolset_names:
        _warn_all_invalid_platform_toolsets(platform, toolset_names)
    return enabled_toolsets

def _prune_toolsets_stripped_by_disabled(enabled_toolsets: set[str], disabled_names: list[str]) -> set[str]:
    """Drop disabled names AND every toolset whose tools the runtime would strip anyway.

    The agent subtracts ``agent.disabled_toolsets`` at TOOL granularity (``model_tools._select_tool_names``),
    so disabling a composite like ``debugging`` removes the terminal/web/file tools even though those names
    never appear in the list. A name-only subtraction here left inspection surfaces (``hermes tools
    --summary``, banner, ``/tools``) showing toolsets as enabled that no session could call (#97015).
    Passthrough entries (MCP server names) and toolsets with no static tools (``context_engine``) are kept.
    """
    from model_tools import _apply_toolset_selection
    from toolsets import resolve_toolset, validate_toolset

    remaining = enabled_toolsets - set(disabled_names)
    resolved = {name: set(resolve_toolset(name)) if validate_toolset(name) else set() for name in remaining}
    surviving: set[str] = set().union(*resolved.values())
    _apply_toolset_selection(surviving, disabled_names, quiet_mode=True, disable=True)
    return {name for name, tools in resolved.items() if not tools or tools & surviving}

def _recover_platform_native_toolsets(enabled_toolsets: set[str], platform: str, *, skip: set[str]) -> None:
    """Add non-configurable platform toolsets (discord, feishu_*) in place: in the default composite but not in
    CONFIGURABLE_TOOLSETS, so never in a checklist or saved list. Runs for BOTH ``_get_platform_tools`` branches."""
    from toolsets import resolve_toolset, TOOLSETS

    platform_tool_universe = set(resolve_toolset(_platform_default_toolset(platform)))
    configurable_tool_universe = {t for ts_key, _, _ in CONFIGURABLE_TOOLSETS for t in resolve_toolset(ts_key)}
    claimed = {t for ts_key in enabled_toolsets for t in resolve_toolset(ts_key)}
    skip = skip | {k for k in TOOLSETS if k.startswith("hermes-")} | (set(_DEFAULT_OFF_TOOLSETS) - {platform})
    for ts_key, ts_def in TOOLSETS.items():
        # Posture toolsets (``coding``) are session-level selections made by agent/coding_context.py, not
        # per-platform capabilities to recover.
        if ts_key in skip or ts_def.get("includes") or ts_def.get("posture"):
            continue
        # Static membership: a registry-added tool absent from the platform composite must not block recovery
        # of a non-configurable toolset whose authored tools the composite lists.
        ts_tools = set(resolve_toolset(ts_key, include_registry=False))
        if not ts_tools or not ts_tools <= platform_tool_universe or ts_tools <= configurable_tool_universe:
            continue
        if not ts_tools <= claimed:
            enabled_toolsets.add(ts_key)
            claimed.update(ts_tools)

def _merge_mcp_servers(
    config: dict, toolset_names: list[str], explicit_passthrough: set[str], include_default_mcp_servers: bool
) -> set[str]:
    """Explicit passthrough entries plus this platform's MCP servers: listed names form an allowlist, else every
    globally enabled server (when ``include_default_mcp_servers``); the ``no_mcp`` sentinel disables all."""
    enabled_mcp_servers = enabled_mcp_server_names(config)
    configured = config.get("mcp_servers") if isinstance(config.get("mcp_servers"), dict) else {}
    known = enabled_mcp_servers | {str(name) for name in configured}
    # ``mcp-<server>`` names the same server as its bare alias: either spelling selects only that server
    # (unless a server is literally named ``mcp-<server>``).
    explicit_passthrough = {ts[4:] if ts.startswith("mcp-") and ts not in known
                            and ts[4:] in known else ts for ts in explicit_passthrough}
    result = explicit_passthrough - known
    if "no_mcp" in toolset_names:
        return result - {"no_mcp"}
    explicit_mcp_servers = explicit_passthrough & enabled_mcp_servers
    # A listed server that is configured but disabled is still the user's server selection: it stays
    # off, and it must not fall through to every enabled server.
    if include_default_mcp_servers and not explicit_passthrough & known:
        return result | enabled_mcp_servers
    return result | explicit_mcp_servers

def _warn_all_invalid_platform_toolsets(platform: str, explicit: list) -> None:
    """Warn once when an explicit platform list has only invalid names (``hermes`` for ``hermes-cli`` → no
    native tools), at session tool resolution rather than only in update/doctor."""
    from toolsets import validate_toolset

    named = [str(t) for t in explicit if isinstance(t, str) and t]
    if named and not any(validate_toolset(t) for t in named) and platform not in _warned_invalid_platform_toolsets:
        _warned_invalid_platform_toolsets.add(platform)
        logger.warning(
            "platform '%s' has no valid toolsets configured (unknown "
            "name(s): %s) - tools will be unavailable. Run `hermes tools` "
            "to reconfigure. See issue #38798.",
            platform, ", ".join(named))

def _save_platform_tools(config: dict, platform: str, enabled_toolset_keys: set[str]):
    """Save the selected toolset keys for a platform to config."""
    config.setdefault("platform_toolsets", {})
    # Drop platform-scoped toolsets that don't apply here, so the "Configure all platforms" checklist (or a
    # hand-edited config.yaml) can't turn on `discord` for Telegram.
    enabled_toolset_keys = {ts for ts in enabled_toolset_keys if _toolset_allowed_for_platform(ts, platform)}
    plugin_keys = _get_plugin_toolset_keys()
    # Preserve only existing entries that are neither configurable nor platform defaults (i.e. MCP server
    # names): platform defaults (hermes-cli, ...) resolve to ALL tools and would silently override the user's
    # unchecked selections on the next read. Saving from the picker is consent to clear the "no_mcp" sentinel
    # (no checkbox for it; users who once set it by hand could otherwise never re-enable MCP via the UI).
    drop = _configurable_keys() | plugin_keys | _platform_default_keys() | {"no_mcp"}
    existing_toolsets = _coerce_platform_toolsets_value(
        cfg_get(config, "platform_toolsets", platform, default=[]), platform
    )
    preserved_entries = {str(e) for e in (existing_toolsets if isinstance(existing_toolsets, list) else [])
                         if str(e) not in drop}
    config["platform_toolsets"][platform] = sorted(enabled_toolset_keys | preserved_entries)
    # Record which plugin toolsets this platform "knows" (distinguishes "new plugin, default enabled" from
    # "user disabled it"). _cfg_section normalizes a present-but-null key that setdefault alone would not replace.
    if plugin_keys:
        _cfg_section(config, "known_plugin_toolsets")[platform] = sorted(plugin_keys)
    # Same record for builtin toolsets the checklist offered; without it an unchecked toolset is
    # indistinguishable from one shipped after the save and _enable_recently_shipped_toolsets re-enables it.
    _cfg_section(config, "known_builtin_toolsets")[platform] = sorted(_configurable_keys())
    # Reconcile with agent.disabled_toolsets, which _get_platform_tools applies as a final override: a toolset
    # listed there stays OFF no matter what this writes (Blank Slate installs pre-populate ~27 entries, making
    # the desktop Toolsets UI unable to re-enable anything). Only toolsets just explicitly enabled FOR THIS
    # PLATFORM are cleared, so the list keeps working as a cross-platform suppression list for everything else.
    # See #49995.
    agent_cfg = config.get("agent")
    newly_enabled = enabled_toolset_keys - preserved_entries
    if isinstance(agent_cfg, dict) and agent_cfg.get("disabled_toolsets") and newly_enabled:
        from agent.skill_utils import parse_config_string_list
        parsed_disabled = parse_config_string_list(agent_cfg["disabled_toolsets"])
        remaining = [ts for ts in parsed_disabled if ts not in newly_enabled]
        if remaining != parsed_disabled:
            agent_cfg["disabled_toolsets"] = remaining
    save_config(config)


def _parse_enabled_flag(value, default: bool = True) -> bool:
    """Parse bool-like config values used by tool/platform settings."""
    if isinstance(value, (bool, int)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on", "false", "0", "no", "off"}:
            return lowered in {"true", "1", "yes", "on"}
    return default
