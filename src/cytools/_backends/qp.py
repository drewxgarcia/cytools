# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""Quadratic-programming engines for the tip of a stretched cone.

The problem is

    minimize    (1/2) x.P.x + q.x
    subject to  G.x <= h

with P = 2I, q = 0, G = -A and h = -c, i.e. the minimum-norm point at distance
at least `c` from every defining hyperplane.

Only `osqp` and `mosek` solve it as stated. The LP engines registered for this
task minimise a *linear* functional over the same feasible region, which finds
a point in the stretched cone but not the minimum-norm one -- they are
approximations, and are registered without the `MINIMUM_NORM` promise implied
by the task name. Callers that need the true tip must not silently receive an
LP answer, which is why the LP fallback is expressed as a separate engine
rather than as a branch inside this one.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from cytools._typing import Matrix

__all__ = ["cvxopt_tip", "mosek_tip", "osqp_tip"]


def _qp(
    hyperplanes: Matrix,
    c: float,
    solver: str,
    max_iter: int,
    verbose: bool,
    **kwargs,
) -> np.ndarray | None:
    """Shared QP assembly and dispatch through `qpsolvers`."""
    import qpsolvers

    hp = np.asarray(hyperplanes)
    n = hp.shape[1]

    P = 2 * sparse.identity(n, dtype=float, format="csc")
    q = np.zeros(n, dtype=float)
    G = -1 * sparse.csc_matrix(hp, dtype=float)
    h = np.full(hp.shape[0], -float(c), dtype=float)

    return qpsolvers.solve_qp(
        P, q, G, h, solver=solver, max_iter=max_iter, verbose=verbose, **kwargs
    )


def osqp_tip(
    hyperplanes: Matrix,
    c: float = 1,
    max_iter: int = 10**6,
    verbose: bool = False,
) -> np.ndarray | None:
    """
    **Description:**
    Operator-splitting QP. The default: open source, always installed, and
    solves the true minimum-norm problem.

    The tolerances and `scaling` below are the values this project has always
    used; they are loose enough to converge on the narrow cones that arise at
    large h11 and are checked against `constraint_error_tol` by the caller.

    **Arguments:**
    - `hyperplanes`: The inward-facing hyperplanes A.
    - `c`: The stretching.
    - `max_iter`: Iteration cap.
    - `verbose`: Whether to let the solver print.

    **Returns:**
    The tip, or None if the solver did not converge.
    """
    return _qp(
        hyperplanes,
        c,
        "osqp",
        max_iter,
        verbose,
        scaling=50,
        eps_abs=1e-4,
        eps_rel=1e-4,
        polishing=True,
    )


def mosek_tip(
    hyperplanes: Matrix,
    c: float = 1,
    max_iter: int = 10**6,
    verbose: bool = False,
) -> np.ndarray | None:
    """
    **Description:**
    Interior-point QP on Mosek. Faster and markedly more accurate than the
    open-source engines at large ambient dimension, but closed source and
    licence-gated, so it is only ever selected when actually activated.

    **Arguments:**
    See `osqp_tip`.

    **Returns:**
    The tip, or None if the solver did not converge.
    """
    return _qp(hyperplanes, c, "mosek", max_iter, verbose)


def cvxopt_tip(
    hyperplanes: Matrix,
    c: float = 1,
    max_iter: int = 10**6,
    verbose: bool = False,
) -> np.ndarray | None:
    """
    **Description:**
    Interior-point QP on CVXOPT. An optional extra; kept because it is a
    genuinely independent implementation and so is useful for cross-checking
    OSQP on cones where the two disagree.

    **Arguments:**
    See `osqp_tip`.

    **Returns:**
    The tip, or None if the solver did not converge.
    """
    return _qp(hyperplanes, c, "cvxopt", max_iter, verbose)
