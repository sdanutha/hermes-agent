"""Production discovery and execution of mounted runtime extensions without a frontend."""
import json
from pathlib import Path
import sys


def test_skills_plugins_and_real_stdio_mcp_execute_through_registry(tmp_path, monkeypatch):
    home = tmp_path / 'state'
    home.mkdir(mode=0o700)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_RUNTIME_DIR', str(tmp_path / 'runtime'))
    skill = home / 'skills' / 'core-proof'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\nname: core-proof\ndescription: Core fixture\n---\n\nCORE_SKILL_PROOF\n')
    plugin = home / 'plugins' / 'core_probe'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text('name: core_probe\nversion: 1.0.0\n')
    (plugin / '__init__.py').write_text('''def register(ctx):
    ctx.register_tool(name="core_probe", toolset="plugin_core_probe",
        schema={"name": "core_probe", "description": "Probe", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: '{"proof": "CORE_PLUGIN_PROOF"}')
''')
    fixture = Path(__file__).resolve().parents[1] / 'e2e/core/mcp_plugins/mcp_fixture_server.py'
    (home / 'config.yaml').write_text(json.dumps({
        'plugins': {'enabled': ['core_probe']},
        'mcp_servers': {'core': {'command': sys.executable, 'args': [str(fixture)],
                                 'env': {'MCPE2E_CANARY': 'CORE_MCP_PROOF'}}}}))
    from hermes_cli.plugins import discover_plugins
    from model_tools import discover_builtin_tools, handle_function_call
    from tools.mcp_tool_discovery import discover_mcp_tools
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers

    discover_builtin_tools()
    discover_plugins(force=True)
    assert 'CORE_SKILL_PROOF' in handle_function_call('skill_view', {'name': 'core-proof'})
    assert 'CORE_PLUGIN_PROOF' in handle_function_call('core_probe', {})
    try:
        names = discover_mcp_tools()
        assert 'mcp__core__ro_probe' in names, names
        result = handle_function_call('mcp__core__ro_probe', {'nonce': 'roundtrip'})
        assert 'RO:CORE_MCP_PROOF:roundtrip' in result, result
    finally:
        shutdown_mcp_servers()
