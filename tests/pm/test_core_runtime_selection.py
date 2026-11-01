"""Installing the first optional plugin must retain the headless transport closure."""
import sys


def test_first_dependency_generation_carries_runtime_extra(tmp_path):
    from pm.extras import legacy_selection

    site = tmp_path / '.venv' / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
    for module in ('aiohttp', 'mcp'):
        package = site / module
        package.mkdir(parents=True)
        (package / '__init__.py').touch()
    assert 'runtime' in legacy_selection(tmp_path)
