"""Deprecated alias for :mod:`alphapepttools.io._registry`."""

import warnings

from ._registry import list_available_reader

# Kept so that direct `alphapepttools.io.utils` imports in downstream code keep working
warnings.warn(
    "`alphapepttools.io.utils` is deprecated and will be removed in a future release. "
    "Use `alphapepttools.io.list_available_reader` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["list_available_reader"]
