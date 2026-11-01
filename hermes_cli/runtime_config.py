"""Profile-aware persistent configuration updates shared by runtime controls."""
import logging
import os
from hermes_constants import get_hermes_home
logger = logging.getLogger(__name__)

def save_config_value(key_path: str, value: any) -> bool:
    """Persist dot-separated ``key_path`` = value into HERMES_HOME/config.yaml; True on success.

    Never the repo's cli-config.yaml: no config reader loads it, so the value would vanish.
    """
    config_path = get_hermes_home() / 'config.yaml'

    try:
        from hermes_constants import mkdir_under_hermes_home
        mkdir_under_hermes_home(config_path.parent)
        from utils import atomic_roundtrip_yaml_update
        atomic_roundtrip_yaml_update(config_path, key_path, value)
        try:  # owner-only: config files contain API keys
            os.chmod(config_path, 0o600)
        except (OSError, NotImplementedError):
            pass
        return True
    except Exception as e:
        logger.error("Failed to save config: %s", e)
        return False


import threading
from contextlib import contextmanager
_CONFIG_MUTATION_LOCK = threading.RLock()


@contextmanager
def config_write_scope(profile):
    if profile is not None:
        raise ValueError("Bind the owning profile runtime scope before changing configuration")
    with _CONFIG_MUTATION_LOCK:
        yield
