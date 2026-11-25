"""Deprecated alias for :mod:`alphapepttools.io._registry`, scheduled for removal in version 0.7.0."""

import warnings

from ._registry import list_available_reader

# Kept so that direct `alphapepttools.io.utils` imports in downstream code keep working
warnings.warn(
    "`alphapepttools.io.utils` is deprecated and will be removed in version 0.7.0. "
    "Use `alphapepttools.io.list_available_reader` instead.",
    FutureWarning,
    stacklevel=2,
)

__all__ = ["list_available_reader"]
