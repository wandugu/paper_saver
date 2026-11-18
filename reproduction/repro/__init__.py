"""Self-contained framework for preregistered reproduction experiments.

Importing :mod:`repro` installs the mandatory trace-store guard before any
runner code is loaded.
"""

from .common.guard import install as _install_guard

_install_guard()

__all__ = ["common"]

