"""Prevent reproduction runs from importing or opening paper-derived data.

The process-wide guard is installed when the package is imported. Deployment
blocklists extend the built-in rules and cannot disable them.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from importlib.abc import MetaPathFinder
import os
import sys


class TraceStoreAccessError(RuntimeError):
    """Raised when a run opens a blocked paper-derived path."""


_INSTALLED = False
_DEFAULT = ("trace_store*", "*_trace_store*")
_module_patterns = ()
_path_patterns = ()


def _normalize(value):
    return value.lower().replace("-", "_").replace("\\", "/")


def _patterns(variable):
    return tuple(_normalize(part.strip()) for part in os.environ.get(variable, "").split(";")
                 if part.strip())


def _blocked_component(components):
    return any(fnmatchcase(component, pattern) for component in components for pattern in _DEFAULT)


def _blocked_module(name):
    normalized = _normalize(name)
    components = normalized.split(".")
    prefixes = (".".join(components[:length]) for length in range(1, len(components) + 1))
    return _blocked_component(components) or any(
        fnmatchcase(prefix, pattern) for prefix in prefixes for pattern in _module_patterns
    )


class _GuardFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if _blocked_module(fullname):
            raise ImportError(f"Reproduction guard blocks module: {fullname}")
        return None


def _blocked_path(value):
    normalized = _normalize(value)
    parts = normalized.split("/")
    return _blocked_component(parts) or any(
        fnmatchcase(candidate, pattern)
        for pattern in _path_patterns
        for candidate in (normalized, *parts)
    )


def _audit_hook(event, args):
    if event != "open" or not args:
        return
    try:
        path = os.fsdecode(os.fspath(args[0]))
    except TypeError:  # An already-open file descriptor supplies no pathname.
        return
    if _blocked_path(path) or _blocked_path(os.path.realpath(path)):
        raise TraceStoreAccessError(f"Reproduction guard blocks path: {path}")


def install() -> None:
    """Install the guard; reject an unsafe process that already loaded blocked code."""
    global _INSTALLED, _module_patterns, _path_patterns
    if not _INSTALLED:
        _module_patterns = _patterns("REPRO_BLOCK_MODULES")
        _path_patterns = _patterns("REPRO_BLOCK_PATHS")
    for name in tuple(sys.modules):
        if _blocked_module(name):
            raise ImportError(f"Reproduction guard found a blocked module already imported: {name}")
    if not _INSTALLED:
        sys.meta_path.insert(0, _GuardFinder())
        sys.addaudithook(_audit_hook)
        _INSTALLED = True


def is_installed() -> bool:
    return _INSTALLED


__all__ = ["TraceStoreAccessError", "install", "is_installed"]

