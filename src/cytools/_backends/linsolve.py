# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""Sparse solvers for the normal equations (M^T M) x = -M^T C.

The system is symmetric positive definite. CHOLMOD's supernodal Cholesky is
the right factorization for it and SuperLU's LU is not, which is why the gap
between these two engines is a factor rather than a few percent: measured on
real intersection-number systems, 428.2 -> 150.5 ms at h11=150 and
1105.9 -> 401.1 ms at h11=300, about 2.8x on the whole per-geometry payload.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["cholmod_solve", "superlu_solve"]


def cholmod_solve(M: sp.csr_matrix, C) -> np.ndarray | None:
    """
    **Description:**
    Solve via a CHOLMOD Cholesky factorization of the normal equations.

    Requires `scikit-sparse` and SuiteSparse headers, supplied by the
    `cytools-workbench[performance]` extra.

    **Arguments:**
    - `M`: The matrix.
    - `C`: The constant term.

    **Returns:**
    The solution, or None if the factorization failed numerically.
    """
    from sksparse.cholmod import (  # ty: ignore[unresolved-import]  # compiled extension, no stubs
        CholmodError,
        cho_factor,
    )

    Mt = M.transpose()
    try:
        solution = cho_factor((Mt @ M).tocsc()).solve(-Mt.dot(C))
    except (CholmodError, ArithmeticError, ValueError):
        return None
    return np.asarray(solution).ravel()


def superlu_solve(M: sp.csr_matrix, C) -> np.ndarray | None:
    """
    **Description:**
    Solve via SciPy's SuperLU. Always available, and the wrong factorization
    for an SPD system, but the only option without SuiteSparse.

    :::note
    The `MMD_ATA` ordering is measured, not guessed. On real
    intersection-number systems it beats the COLAMD default by ~1.2x at equal
    residual, while `MMD_AT_PLUS_A` is 45x *slower* and `NATURAL` 100x slower.
    Do not change it without re-measuring.
    :::

    **Arguments:**
    - `M`: The matrix.
    - `C`: The constant term.

    **Returns:**
    The solution, or None if the solve failed.
    """
    # One transpose, reused. `M.transpose()` was called twice here, which also
    # made this the only one of the two engines not to hoist it -- `cholmod_solve`
    # already did.
    Mt = M.transpose()
    try:
        solution = sp.linalg.spsolve(Mt @ M, -Mt @ C, permc_spec="MMD_ATA")
    except Exception:
        return None
    return np.asarray(solution).ravel()
