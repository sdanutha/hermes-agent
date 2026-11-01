"""Non-interactive entry point for Hermes Core Runtime."""
from __future__ import annotations

import argparse
import json
import os
import sys


def health() -> int:
    """Probe the live owner's private control socket without starting a process."""
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_runtime_discovery import query_identify

    try:
        descriptor = query_identify(get_hermes_home(), timeout=5)
    except (OSError, ValueError, TimeoutError) as exc:
        print(json.dumps({"status": "unavailable", "reason": type(exc).__name__}))
        return 1
    ready = descriptor.get("state") == "ready"
    print(json.dumps({"status": "ready" if ready else "unavailable",
                      "instance_id": descriptor.get("instance_id")}))
    return 0 if ready else 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Hermes Core Runtime (no interactive UI)")
    parser.add_argument("command", choices=("run", "health", "config", "profile", "cron", "tools",
                                            "skills", "plugins", "mcp", "pm", "sessions", "logs",
                                            "chat", "gateway", "doctor"))
    args, rest = parser.parse_known_args(argv)
    if args.command == "health":
        if rest:
            parser.error("health takes no arguments")
        raise SystemExit(health())
    if args.command == "run":
        # Keep the canonical owner argv for process identity, singleton checks and job ownership.
        os.execv(sys.executable, [sys.executable, "-m", "gateway.run", *rest])
    if args.command == "chat" and not any(value in {"-q", "--query"} for value in rest):
        parser.error("chat requires -q/--query; interactive chat is unavailable in Core")
    # Reuse the existing management commands and their profile/config/security semantics.
    from hermes_cli.main import main as manage
    sys.argv = [sys.argv[0], args.command, *rest]
    manage()


if __name__ == "__main__":
    main()
