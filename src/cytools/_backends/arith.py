# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""Small exact-arithmetic helpers shared by the engine adapters.

Duplicated rather than imported from `cytools.utils`: backend modules must not
import domain code, which `tests/test_architecture.py` pins mechanically.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from flint import fmpz_mat

from cytools._typing import Matrix

__all__ = ["exact_rank", "gcd_int", "is_unimodular"]


def gcd_int(values: Iterable[float]) -> int:
    """The exact integer gcd of `values`, never zero."""
    g = 0
    for v in values:
        g = math.gcd(g, abs(int(v)))
    return g or 1


def _integral(matrix: Matrix) -> list[list[int]]:
    """`matrix` as nested Python ints, refusing anything not integral.

    Refusing is the point. Silently rounding would answer a question about a
    different matrix, and every caller here is asking about a lattice object
    whose entries are integers by construction.

    Nothing is routed through `float` on the way. An object-dtype array can
    hold integers wider than float64 -- which is exactly the regime this
    module exists for -- so casting to check integrality would corrupt the
    values it was meant to validate.
    """
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-dimensional matrix, got shape {array.shape}")

    rows = []
    for row in array:
        entries = []
        for value in row:
            as_int = int(value)
            if as_int != value:
                raise ValueError(
                    "exact integer linear algebra needs an integral matrix; "
                    f"found {value!r}"
                )
            entries.append(as_int)
        rows.append(entries)
    return rows


def exact_rank(matrix: Matrix) -> int:
    """
    **Description:**
    The rank of an integer matrix, computed exactly.

    `numpy.linalg.matrix_rank` decides rank by comparing singular values to
    `S.max() * max(M, N) * eps`, which is a statement about a *nearby* matrix.
    On an ill-conditioned integer matrix that is the wrong answer, not a less
    precise one: a unimodular 16x16 built as a product of unit-triangular
    integer factors has condition number ~2e17 and reads as rank 15, and at
    32x32 as rank 29. A cone built on such rays then reports itself as not
    full-dimensional, which is a false claim about its geometry rather than a
    rounding error in a number.

    **Arguments:**
    - `matrix`: An integral matrix.

    **Returns:**
    *(int)* The exact rank.

    **Example:**
    ```python {2}
    from cytools._backends.arith import exact_rank
    exact_rank([[1, 0], [0, 1]])
    # 2
    ```
    """
    array = np.asarray(matrix)
    if array.size == 0:
        return 0
    return int(fmpz_mat(_integral(array)).rank())


def is_unimodular(matrix: Matrix) -> bool:
    """
    **Description:**
    Whether a square integer matrix has determinant +-1, computed exactly.

    A determinant grows like the product of the singular values, so float64
    loses it long before it loses a rank: on the unimodular families above the
    exact determinant is 1 while `numpy.linalg.det` returns 1.19 at 16x16 and
    -4.5e29 at 32x32.

    **Arguments:**
    - `matrix`: A square integral matrix.

    **Returns:**
    *(bool)* Whether the matrix is unimodular.
    """
    rows = _integral(matrix)
    if len(rows) != len(rows[0]):
        raise ValueError("unimodularity is a property of a square matrix")
    return abs(int(fmpz_mat(rows).det())) == 1
