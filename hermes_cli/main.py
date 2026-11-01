"""Headless command entry point; interactive product launchers are not part of Core."""
from __future__ import annotations

import argparse
import os
import sys


def _profile(argv):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('-p', '--profile')
    args, rest = selector.parse_known_args(argv)
    if args.profile and args.profile != 'default':
        from hermes_cli.profiles import validate_profile_name, profile_exists, get_profile_dir
        validate_profile_name(args.profile)
        if not profile_exists(args.profile):
            selector.error(f'Profile does not exist: {args.profile}')
        os.environ['HERMES_HOME'] = str(get_profile_dir(args.profile))
    return rest


def main(argv=None):
    argv = _profile(list(sys.argv[1:] if argv is None else argv))
    from hermes_constants import get_hermes_home
    from hermes_cli.env_loader import load_hermes_dotenv
    from agent.ssl_verify import install_truststore
    load_hermes_dotenv(hermes_home=get_hermes_home())
    install_truststore()
    if argv[:1] == ['pm']:
        from pm.cli import main as package_main
        raise SystemExit(package_main(argv[1:]))
    from hermes_cli.core_commands import build_parser
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    if isinstance(result, int):
        raise SystemExit(result)


if __name__ == '__main__':
    main()
