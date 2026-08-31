# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""PyNormaliz adapter."""

import numpy as np

from cytools._typing import Matrix


def hilbert_basis(rays: Matrix) -> np.ndarray:
    """Compute a cone's Hilbert basis without exposing PyNormaliz objects."""
    try:
        from PyNormaliz import Cone as NormalizCone
    except ModuleNotFoundError as exc:
        if exc.name != "PyNormaliz":
            raise
        raise ImportError(
            "Hilbert basis computation requires the optional PyNormaliz "
            "binding. Install it with "
            "`python -m pip install \"cytools[normaliz]\"`."
        ) from exc

    cone = NormalizCone(cone=np.asarray(rays, dtype=int).tolist())
    return np.asarray(
        cone.HilbertBasis(),  # ty: ignore[unresolved-attribute]  # dynamic binding
        dtype=int,
    )


__all__ = ["hilbert_basis"]

