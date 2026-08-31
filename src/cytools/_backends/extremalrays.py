# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""Adapter for the extremalrays acceleration package."""

import importlib.util

import numpy as np

from cytools._typing import Matrix


def is_available() -> bool:
    """Whether the optional accelerator can be imported."""
    return importlib.util.find_spec("extremalrays") is not None


def exhaustive_indices(rays: Matrix, *, verbose: bool = False) -> np.ndarray:
    """Return sorted indices of the extremal rows in *rays*."""
    import extremalrays

    indices = extremalrays.exhaustive(np.asarray(rays), verbosity=1 if verbose else 0)
    return np.sort(np.asarray(indices, dtype=int))


__all__ = ["exhaustive_indices", "is_available"]
