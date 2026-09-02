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

with P = 2I, q = 0, G = -A and h = -c: the minimum-norm point of the region
`A x >= c`.

`A` holds primitive *integral* lattice normals, so each row of `A x >= c` says
that a curve's volume is at least `c` -- the pairing in which the type IIB
literature stretches the Kahler cone. This is not Euclidean distance `c` from
each facet, which would read `a.x >= c ||a||`. The distinction is not
cosmetic: on a real h11 = 30 Kahler cone the row norms of `A` span a factor of
eight, so the two regions genuinely differ.

The LP engines registered for the feasibility task minimise a *linear*
functional over this same region, so they return a point of the stretched cone
rather than its minimum-norm point -- a different mathematical object. That is
why none of them is registered here.

Optimality is checked, not assumed. The Lagrangian dual of the problem above
is

    maximize    c 1.lam - (1/4) ||A'.lam||^2      subject to  lam >= 0

attained at x = (1/2) A'.lam. Any feasible `x` bounds the optimum from above
and *any* `lam >= 0` bounds it from below, so a solver's primal-dual pair can
validate optimality to explicit tolerances without a second solver -- see
`certify_tip`. `highs_tip` returns a point only once that check passes,
returns `None` only where the region is proved empty, and raises
`SolverFailure` otherwise. `cytools._backends.lp` states the same contract,
for the same reason: a numerical failure must not be readable as a claim about
the geometry.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import sparse

from cytools._backends.lp import SolverFailure, highs_feasibility
from cytools._typing import Matrix

__all__ = [
    "GAP_TOL",
    "FEASIBILITY_TOL",
    "TipCertificate",
    "certify_tip",
    "cvxopt_tip",
    "highs_tip",
    "mosek_tip",
    "osqp_tip",
]


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


# certification
# -------------
# Both tolerances are stated in quantities meaningful to this problem rather
# than opaque solver-internal residuals.

#: Largest tolerated violation of `A x >= c`. Absolute, because the constraint
#: is a curve volume against `c`, and that is the scale the physics fixes.
FEASIBILITY_TOL = 1e-6

#: Largest tolerated relative duality gap. Relative, because the objective
#: itself can vary by many orders of magnitude across cones.
GAP_TOL = 1e-8


class TipCertificate(NamedTuple):
    """Two-sided evidence about a candidate tip, independent of its solver.

    **Attributes:**
    - `objective`: ||x||^2 for the candidate. An upper bound on the true
        minimum *only* if `worst_violation` is non-negative.
    - `dual_bound`: The dual objective at the reported multipliers. A lower
        bound on the true minimum unconditionally -- the dual is valid for
        every `lam >= 0`, including one a solver stopped short of optimising.
    - `relative_gap`: `(objective - dual_bound)` over the objective scale.
    - `worst_violation`: The most negative entry of `A x - c`.
    - `relative_stationarity`: How far `2x = A'.lam` misses, relative to the
        size of the gradient. Diagnostic only; the gap is the claim.
    """

    objective: float
    dual_bound: float
    relative_gap: float
    worst_violation: float
    relative_stationarity: float

    def holds(
        self,
        feasibility_tol: float = FEASIBILITY_TOL,
        gap_tol: float = GAP_TOL,
    ) -> bool:
        """Whether the primal-dual check meets the given tolerances."""
        return (
            self.worst_violation >= -feasibility_tol
            and abs(self.relative_gap) <= gap_tol
        )


def certify_tip(
    hyperplanes: Matrix,
    c: float,
    x: Matrix,
    dual: Matrix,
) -> TipCertificate:
    """
    **Description:**
    Bound the true minimum norm from both sides using a solver's primal-dual
    pair, so that "this is the tip" becomes a checked statement rather than
    merely the solver's opinion about its own convergence.

    Negative multipliers are clipped to zero. That only weakens the lower
    bound and can never invalidate it, which is what makes the result usable
    on output a solver did not finish polishing.

    A *negative* `relative_gap` is not a paradox. It means the candidate is
    infeasible, so `objective` was never an upper bound in the first place;
    `worst_violation` reports by how much. This is the ordinary signature of a
    first-order method stopped early, and it is why feasibility and the gap
    are both required rather than either alone.

    **Arguments:**
    - `hyperplanes`: The inward-facing hyperplanes A.
    - `c`: The stretching.
    - `x`: The candidate tip.
    - `dual`: The multipliers the solver reports for `A x >= c`.

    **Returns:**
    The certificate. `TipCertificate.holds` tests it.

    **Example:**
    The tip of the stretched first quadrant is (c, c), and certifying it
    against its exact multipliers closes the gap.
    ```python {4}
    A = np.eye(2)
    x = np.array([1.0, 1.0])
    lam = np.array([2.0, 2.0])
    certify_tip(A, 1, x, lam).holds()
    # True
    ```
    """
    A = np.asarray(hyperplanes, dtype=float)
    point = np.asarray(x, dtype=float)
    lam = np.clip(np.asarray(dual, dtype=float), 0.0, None)

    # `A'.lam` is both half the dual gradient and, at the optimum, the primal
    # gradient `2x`; computing it once is what lets the gap and the
    # stationarity residual share a single matvec.
    dual_gradient = A.T @ lam
    objective = float(point @ point)
    dual_bound = float(c * lam.sum() - 0.25 * (dual_gradient @ dual_gradient))
    gradient = 2.0 * point

    residuals = A @ point - c
    return TipCertificate(
        objective=objective,
        dual_bound=dual_bound,
        relative_gap=(objective - dual_bound) / max(1.0, abs(objective)),
        # An unconstrained problem is vacuously feasible; `min` of nothing is
        # not.
        worst_violation=float(residuals.min()) if residuals.size else float("inf"),
        relative_stationarity=(
            float(np.abs(gradient - dual_gradient).max())
            / max(1.0, float(np.abs(gradient).max()))
            if point.size
            else 0.0
        ),
    )


def highs_tip(
    hyperplanes: Matrix,
    c: float = 1,
    max_iter: int = 10**6,
    verbose: bool = False,
) -> np.ndarray | None:
    """
    **Description:**
    Null-space active-set QP on HiGHS. This is the default because HiGHS is a
    hard dependency, its primal and dual output support an independent
    optimality check, and its infeasibility path is confirmed exactly by the
    feasibility adapter.

    **Arguments:**
    - `hyperplanes`: The inward-facing hyperplanes A.
    - `c`: The stretching.
    - `max_iter`: Iteration cap, forwarded as HiGHS' `qp_iteration_limit`.
        Note that `max_iter` is not itself a HiGHS option: passing it as one
        sets nothing at all and leaves the solver effectively unbounded.
    - `verbose`: Whether to let the solver print.

    **Returns:**
    The certified tip, or `None` when the region `A x >= c` is *proved* empty,
    in which case the stretched cone has no tip because it has no points.

    **Raises:**
    `SolverFailure` when HiGHS neither solved the problem nor proved it
    infeasible, or when the point it returned fails `certify_tip`. Neither
    outcome is a statement about the geometry, so neither may be returned as
    a `None` that a caller could read as one.
    """
    import qpsolvers

    A = np.asarray(hyperplanes)
    rows, n = A.shape

    solution = qpsolvers.solve_problem(
        qpsolvers.Problem(
            2 * sparse.identity(n, dtype=float, format="csc"),
            np.zeros(n, dtype=float),
            -1 * sparse.csc_matrix(A, dtype=float),
            np.full(rows, -float(c), dtype=float),
        ),
        solver="highs",
        verbose=verbose,
        qp_iteration_limit=int(max_iter),
    )

    if not solution.found:
        # HiGHS' own infeasibility verdict is not the proof. `highs_feasibility`
        # decides this very region and confirms every negative result with
        # exact rational arithmetic in PPL, so it is the thing entitled to
        # conclude that there is no tip.
        if highs_feasibility(A, c, n) is None:
            return None
        raise SolverFailure(
            "HiGHS did not solve the stretched-tip QP, and the region is "
            "nonempty, so the absence of a tip is not a conclusion available "
            "here."
        )

    if solution.x is None or solution.z is None:
        raise SolverFailure(
            "HiGHS reported an optimal stretched-tip QP without returning "
            "both primal and dual solutions."
        )
    certificate = certify_tip(A, c, solution.x, solution.z)
    if not certificate.holds():
        raise SolverFailure(
            f"HiGHS returned an uncertified stretched-tip point: {certificate}."
        )
    return np.asarray(solution.x, dtype=float)


def osqp_tip(
    hyperplanes: Matrix,
    c: float = 1,
    max_iter: int = 10**6,
    verbose: bool = False,
) -> np.ndarray | None:
    """
    **Description:**
    Operator-splitting QP. Retained for differential testing and for
    reproducing historical runs; `highs_tip` is the default.

    The tolerances and `scaling` below are the historical CYTools values. This
    adapter intentionally preserves them for reproduction; callers comparing
    solvers should evaluate its result with `certify_tip` rather than treating
    solver convergence as an optimality guarantee.

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
    Interior-point QP on Mosek. Closed source and licence-gated; retained as an
    independent explicit engine when it is activated.

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
