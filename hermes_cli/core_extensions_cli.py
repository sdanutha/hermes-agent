"""Headless skill and tool management using the runtime's own selection policy."""
from __future__ import annotations
import json


def cmd_tools(args):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.tool_selection import _get_platform_tools, _save_platform_tools
    from hermes_cli.plugins import discover_plugins
    from toolsets import validate_toolset
    discover_plugins()
    action = getattr(args, 'tools_action', None) or 'list'
    platform = getattr(args, 'platform', 'cli')
    config = load_config()
    enabled = _get_platform_tools(config, platform, include_default_mcp_servers=False)
    if action == 'list':
        from model_tools import get_tool_definitions
        print(json.dumps({'platform': platform, 'toolsets': sorted(enabled),
                          'tools': get_tool_definitions(enabled_toolsets=sorted(enabled), quiet_mode=True)}, indent=2))
        return
    if action not in {'enable', 'disable'}:
        raise SystemExit('Core supports tools list, enable and disable; install optional tools through plugins or MCP.')
    names = args.names
    groups = [name for name in names if ':' not in name]
    unknown = [name for name in groups if not validate_toolset(name)]
    if unknown:
        raise SystemExit('Unknown toolsets: ' + ', '.join(unknown))
    servers = config.get('mcp_servers') or {}
    for target in (name for name in names if ':' in name):
        server, tool = target.split(':', 1)
        if not tool or not isinstance(servers.get(server), dict):
            raise SystemExit('Unknown MCP tool target: ' + target)
        policy = servers[server].setdefault('tools', {})
        excluded = list(policy.get('exclude') or [])
        if action == 'disable' and tool not in excluded:
            excluded.append(tool)
        elif action == 'enable':
            excluded = [name for name in excluded if name != tool]
        policy['exclude'] = excluded
    selected = enabled | set(groups) if action == 'enable' else enabled - set(groups)
    _save_platform_tools(config, platform, selected)
    save_config(config)
    print('Saved. The selection applies to new sessions.')


def cmd_skills(args):
    action = getattr(args, 'skills_action', None)
    if action == 'config':
        raise SystemExit('Configure skills through config.yaml; interactive menus are not included in Core.')
    if action in {'trust', 'untrust'}:
        return _cmd_skills_trust(args)
    from hermes_cli.skills_hub import skills_command
    return skills_command(args)


def _cmd_skills_trust(args):
    """``hermes skills trust|untrust [path]`` — manage ``skills.trusted_project_dirs``.

    With no path, operates on the project root enclosing the current directory
    (nearest ancestor with ``.git``).
    """
    from pathlib import Path
    from agent.skill_utils import (
        PROJECT_SKILLS_SUBDIRS,
        _candidate_project_skills_dirs,
        find_project_root,
        iter_skill_index_files)
    from hermes_cli.config import load_config, save_config
    action = args.skills_action
    raw_path = getattr(args, "path", None)
    if raw_path:
        root = Path(raw_path).expanduser().resolve()
        if not root.is_dir():
            print(f"Not a directory: {root}")
            return
    else:
        root = find_project_root()
        if root is None:
            print(
                "Not inside a git checkout. Run from a project directory or "
                "pass the project root path explicitly.")
            return

    config = load_config()
    skills_cfg = config.setdefault("skills", {})
    trusted = skills_cfg.get("trusted_project_dirs") or []
    if not isinstance(trusted, list):
        trusted = [trusted]
    trusted = [str(t) for t in trusted]
    root_str = str(root)

    def _same(t: str) -> bool:
        return str(Path(t).expanduser().resolve()) == root_str

    if action == "untrust":
        kept = [t for t in trusted if not _same(t)]
        if len(kept) == len(trusted):
            print(f"{root} was not trusted.")
            return
        skills_cfg["trusted_project_dirs"] = kept
        save_config(config)
        print(f"Untrusted: {root}")
        print("Project skills from this repo will no longer load.")
        return

    if any(_same(t) for t in trusted):
        print(f"Already trusted: {root}")
    else:
        trusted.append(root_str)
        skills_cfg["trusted_project_dirs"] = trusted
        save_config(config)
        print(f"Trusted: {root}")

    # Show what this unlocks
    count = sum(
        sum(1 for _ in iter_skill_index_files(d, "SKILL.md"))
        for d in _candidate_project_skills_dirs(root))
    if count:
        print(
            f"{count} project skill(s) will load in sessions started inside "
            "this repo (they take precedence over same-named profile skills).")
    else:
        subdirs = " or ".join(PROJECT_SKILLS_SUBDIRS)
        print(f"No project skills found yet — add them under {subdirs}.")
