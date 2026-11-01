"""Non-interactive command composition for the runtime and its configuration."""
from __future__ import annotations

import argparse
import importlib


_COMMANDS = {
    'config': ('hermes_cli.config', 'config_command'),
    'cron': ('hermes_cli.cron', 'cron_command'),
    'profile': ('hermes_cli.profile_cmd', 'cmd_profile'),
    'plugins': ('hermes_cli.plugins_cmd', 'plugins_command'),
    'mcp': ('hermes_cli.mcp_config', 'mcp_command'),
    'sessions': ('hermes_cli.sessions_cmd', 'cmd_sessions'),
    'logs': ('hermes_cli.core_commands', 'logs_command'),
    'tools': ('hermes_cli.core_extensions_cli', 'cmd_tools'),
    'skills': ('hermes_cli.core_extensions_cli', 'cmd_skills'),
    'gateway': ('hermes_cli.gateway', 'gateway_command'),
    'doctor': ('hermes_cli.doctor', 'run_doctor'),
}


def _dispatch(name):
    def run(args):
        module, function = _COMMANDS[name]
        return getattr(importlib.import_module(module), function)(args)
    return run


def _chat_parser(subparsers):
    parser = subparsers.add_parser('chat', help='Submit one task to the runtime')
    parser.add_argument('-q', '--query', required=True)
    for name in ('model', 'provider', 'base-url', 'api-key', 'resume', 'reasoning', 'skills'):
        parser.add_argument('--' + name)
    parser.add_argument('-m', dest='model')
    parser.add_argument('-t', '--toolsets')
    parser.add_argument('--max-turns', type=int)
    parser.add_argument('--output-format', choices=('text', 'stream-json'), default='text')
    for flag in ('yolo', 'safe-mode', 'ignore-rules', 'ignore-user-config', 'accept-hooks'):
        parser.add_argument('--' + flag, action='store_true')
    def run(args):
        from hermes_cli.gateway_chat_startup import launch_gateway_chat
        return launch_gateway_chat(args)
    parser.set_defaults(func=run)


def build_parser():
    parser = argparse.ArgumentParser(prog='hermes', description='Hermes Core Runtime management')
    subparsers = parser.add_subparsers(dest='command', required=True)
    for name in _COMMANDS:
        module = importlib.import_module('hermes_cli.subcommands.' + name)
        builder = getattr(module, f'build_{name}_parser')
        kwargs = {'cmd_' + name: _dispatch(name)}
        if name == 'gateway':
            kwargs.update(cmd_proxy=_unavailable, cmd_gateway_enroll=_unavailable)
        builder(subparsers, **kwargs)
    _chat_parser(subparsers)
    return parser


def _unavailable(args):
    raise SystemExit('This command is outside the Core Runtime distribution.')


def logs_command(args):
    """View and filter Hermes log files."""
    from hermes_cli.logs import tail_log, list_logs

    log_name = getattr(args, "log_name", "agent") or "agent"

    if log_name == "list":
        list_logs()
        return

    tail_log(
        log_name,
        num_lines=getattr(args, "lines", 50),
        follow=getattr(args, "follow", False),
        level=getattr(args, "level", None),
        session=getattr(args, "session", None),
        since=getattr(args, "since", None),
        component=getattr(args, "component", None),
    )
