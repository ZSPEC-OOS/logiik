"""
Logiik configuration loader.
Reads config.yaml and resolves ${ENV_VAR} references from environment.
"""
import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent.parent  # repo root
load_dotenv(_ROOT / ".env", override=False)

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _resolve_env_vars(obj):
    """Recursively resolve ${VAR_NAME} patterns in config values."""
    if isinstance(obj, str):
        def replacer(match):
            var = match.group(1)
            val = os.environ.get(var)
            if val is None:
                return match.group(0)  # leave unresolved if not set
            return val
        return re.sub(r'\$\{(\w+)\}', replacer, obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(i) for i in obj]
    return obj


def load_config() -> dict:
    """Load and return the fully resolved Logiik configuration."""
    with open(_CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f)
    return _resolve_env_vars(raw)


# Module-level singleton — import this directly
CONFIG = load_config()
