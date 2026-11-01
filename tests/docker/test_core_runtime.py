"""Build and boot the actual Python-only, non-root Core image when Docker is available."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.waiting import wait_until


def docker(*args, timeout=30):
    return subprocess.run(['docker', *args], capture_output=True, text=True, timeout=timeout, check=False)


def test_core_image_build_boot_health_and_sigterm():
    if not shutil.which('docker') or docker('info').returncode:
        pytest.skip('Docker daemon unavailable')
    root = Path(__file__).resolve().parents[2]
    name = f'hermes-core-contract-{os.getpid()}'
    image = name + ':test'
    built = docker('build', '-t', image, str(root), timeout=1200)
    assert built.returncode == 0, built.stdout + built.stderr
    try:
        started = docker('run', '-d', '--name', name, image)
        assert started.returncode == 0, started.stderr
        def healthy():
            state = docker('inspect', '--format', '{{.State.Running}}', name)
            assert state.stdout.strip() == 'true', docker('logs', name).stdout
            return docker('exec', name, 'python', '-m', 'hermes_core', 'health').returncode == 0
        wait_until(healthy, timeout=120, interval=1, what='container runtime readiness')
        result = docker('exec', name, 'python', '-c',
                        'import os,shutil,json; print(json.dumps([os.getuid(),shutil.which("node"),shutil.which("npm")]))')
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == [10000, None, None]
        tool = docker('exec', name, 'python', '-c',
            'from model_tools import discover_builtin_tools,handle_function_call; '
            'discover_builtin_tools(); '
            'assert "Hermes Core Runtime" in handle_function_call("read_file", {"path":"/opt/hermes/README.md"})')
        assert tool.returncode == 0, tool.stdout + tool.stderr
        stopped = docker('stop', '--time', '90', name, timeout=100)
        assert stopped.returncode == 0, stopped.stderr
        # Existing gateway external-stop policy distinguishes raw SIGTERM from a planned stop.
        state = docker('inspect', '--format', '{{.State.ExitCode}}', name)
        assert state.stdout.strip() == '1', docker('logs', name).stdout
    finally:
        docker('rm', '-fv', name)
        docker('image', 'rm', image)
