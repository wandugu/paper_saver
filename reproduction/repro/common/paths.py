"""Expand deployment paths in reproduction configurations and receipts."""

import os
import re


_DEPLOYMENT_VARIABLES = {
    "DATASET_ROOT", "MODEL_ROOT", "P24_RUN_ROOT", "P24_REPO_ROOT",
    "P24_FORMAL_HOST", "P24_FORMAL_HOSTNAME", "P24_PILOT_HOSTNAME",
    "P24_LOCK_HOST", "P24_LOCK_ROOT", "P24_TMUX_PREFIX",
}
_PLACEHOLDER = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def require_env(name):
    """Get an explicitly configured, nonblank deployment value."""
    value = os.environ.get(name)
    if not value or not value.strip():
        raise RuntimeError(f"Set {name} to a nonempty value before using this entry point")
    return value


def _resolve_placeholder(match):
    name = match[1]
    if name not in _DEPLOYMENT_VARIABLES:
        raise ValueError(f"Unsupported deployment placeholder: {match[0]}")
    return require_env(name)


def expand_config_paths(value):
    """Resolve deployment placeholders in scalar fields, including embedded paths."""
    if isinstance(value, dict):
        return {key: expand_config_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_config_paths(item) for item in value]
    if isinstance(value, str):
        return _PLACEHOLDER.sub(_resolve_placeholder, value)
    return value
