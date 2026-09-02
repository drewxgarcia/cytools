# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""Feasibility engines: find x with A x >= c, or prove none exists.

Every engine returns a point, or `None` **only** when infeasibility was
actually proved. An engine that runs out of iterations, hits a numerical
problem, or returns a status it cannot interpret raises `SolverFailure`
instead.

That distinction is the whole point of this module. `Cone.is_solid` reads a
missing interior point as "the cone is not full-dimensional", which is a
mathematical conclusion. Before this split, four different solvers funnelled
"proved infeasible" and "I gave up" into the same `None`, so a numerical
failure at high dimension silently became a claim about the geometry.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from cytools._backends.arith import gcd_int
from cytools._typing import Matrix

__all__ = [
    "SolverFailure",
    "cpsat_feasibility",
    "glop_feasibility",
    "highs_feasibility",
    "scip_feasibility",
]


class SolverFailure(RuntimeError):
    """A solver failed to reach a conclusion.

    Distinct from a `None` return, which asserts that the problem is
    infeasible. Callers must not read this as a statement about the geometry.
    """


def _rows(hyperplanes):
    """Iterate (index, value) pairs per row, dense or sparse-dict."""
    if isinstance(hyperplanes, (list, np.ndarray)):
        return enumerate
    return lambda hp: hp.items()


def highs_feasibility(
    hyperplanes: Matrix | Sequence[Mapping[int, float]],
    c: float,
    ambient_dim: int,
    lower_bound: float | None = None,
    verbose: bool = False,
) -> np.ndarray | None:
    """
    **Description:**
    LP feasibility on HiGHS. The default engine: an exact simplex basis
    status, so infeasibility is proved rather than inferred from a failure.

    **Arguments:**
    - `hyperplanes`: The constraint matrix A, dense or as sparse rows.
    - `c`: The required stretching, i.e. the right-hand side.
    - `ambient_dim`: Number of columns of A.
    - `lower_bound`: Optional lower bound on every variable.
    - `verbose`: Whether to let the solver print.

    **Returns:**
    A feasible point, or None if the problem was proved infeasible.
    """
    import highspy

    if len(hyperplanes) == 0:
        return np.ones(ambient_dim)

    hp_iter = _rows(hyperplanes)
    n = ambient_dim

    if isinstance(hyperplanes, np.ndarray):
        # dense: assemble the CSR structure with numpy instead of a Python
        # double loop over every (row, column) entry
        m = len(hyperplanes)
        grading = hyperplanes.sum(axis=0) / m
        starts = np.arange(m, dtype=np.int32) * n
        index = np.tile(np.arange(n, dtype=np.int32), m)
        value = hyperplanes.ravel().astype(float)
    else:
        # sparse rows (e.g. LIL): iterate only the stored entries
        starts, index, value = [], [], []
        grading = np.zeros(n)
        for v in hyperplanes:
            starts.append(len(index))
            for ind, val in hp_iter(v):
                index.append(int(ind))
                value.append(float(val))
                grading[int(ind)] += float(val)
        grading /= len(starts)
        starts = np.asarray(starts, dtype=np.int32)
        index = np.asarray(index, dtype=np.int32)
        value = np.asarray(value, dtype=float)

    inf = highspy.kHighsInf
    lb = -inf if lower_bound is None else float(lower_bound)
    h = highspy.Highs()
    if not verbose:
        h.silent()
    h.addVars(n, np.full(n, lb), np.full(n, inf))
    h.changeColsCost(n, np.arange(n, dtype=np.int32), grading)
    h.addRows(
        len(starts),
        np.full(len(starts), float(c)),
        np.full(len(starts), inf),
        len(index),
        starts,
        index,
        value,
    )
    h.run()

    status = h.getModelStatus()
    if status == highspy.HighsModelStatus.kOptimal:
        return np.asarray(h.getSolution().col_value, dtype=float)
    if status == highspy.HighsModelStatus.kInfeasible:
        return None
    raise SolverFailure(f"HiGHS returned status {status!r}, which is not a proof.")


def _ortools_feasibility(
    solver_name: str,
    hyperplanes,
    c: float,
    ambient_dim: int,
    lower_bound: float | None,
    verbose: bool,
) -> np.ndarray | None:
    """Shared body for the two OR-Tools continuous/MIP solvers."""
    from ortools.linear_solver import pywraplp

    if len(hyperplanes) == 0:
        return np.ones(ambient_dim)

    hp_iter = _rows(hyperplanes)
    solver = pywraplp.Solver.CreateSolver(solver_name.upper())
    if verbose:
        solver.EnableOutput()

    var_type = solver.NumVar if solver_name == "glop" else solver.IntVar
    lower = -solver.infinity() if lower_bound is None else lower_bound
    var = [var_type(lower, solver.infinity(), f"x_{i}") for i in range(ambient_dim)]

    for v in hyperplanes:
        cons = solver.Constraint(c, solver.infinity())
        for ind, val in hp_iter(v):
            cons.SetCoefficient(var[ind], float(val))

    obj = solver.Objective()
    obj.SetMinimization()
    obj_vec = np.asarray(hyperplanes).sum(axis=0) / len(hyperplanes)
    for i in range(ambient_dim):
        obj.SetCoefficient(var[i], obj_vec[i])

    status = solver.Solve()
    if status in (solver.FEASIBLE, solver.OPTIMAL):
        return np.array([x.solution_value() for x in var])
    if status == solver.INFEASIBLE:
        return None
    names = [
        "OPTIMAL",
        "FEASIBLE",
        "INFEASIBLE",
        "UNBOUNDED",
        "ABNORMAL",
        "MODEL_INVALID",
        "NOT_SOLVED",
    ]
    raise SolverFailure(
        f"{solver_name} returned status {names[status]}, which is not a proof."
    )


def glop_feasibility(
    hyperplanes, c, ambient_dim, lower_bound=None, verbose=False
) -> np.ndarray | None:
    """LP feasibility on OR-Tools' GLOP. See `highs_feasibility`."""
    return _ortools_feasibility(
        "glop", hyperplanes, c, ambient_dim, lower_bound, verbose
    )


def scip_feasibility(
    hyperplanes, c, ambient_dim, lower_bound=None, verbose=False
) -> np.ndarray | None:
    """Mixed-integer feasibility on SCIP. See `cpsat_feasibility` on scope."""
    return _ortools_feasibility(
        "scip", hyperplanes, c, ambient_dim, lower_bound, verbose
    )


def cpsat_feasibility(
    hyperplanes, c, ambient_dim, lower_bound=None, verbose=False
) -> np.ndarray | None:
    """
    **Description:**
    Integer feasibility on CP-SAT.

    :::caution
    This engine searches the **integer** points of the region, over a bounded
    box (`INT32_MIN` to `INT32_MAX` per coordinate). A solid rational cone can
    therefore be reported infeasible when its integer points all lie outside
    that box, which a narrow cone at high dimension can easily do. CP-SAT is
    consequently registered *without* the `CERTIFIES_INFEASIBLE` guarantee and
    is never selected where a missing point is read as "not full-dimensional".
    :::

    **Arguments:**
    See `highs_feasibility`.

    **Returns:**
    An integer feasible point, or None when CP-SAT proved no integer point
    exists in its search box.
    """
    from ortools.sat.python import cp_model

    if len(hyperplanes) == 0:
        return np.ones(ambient_dim)

    A = np.asarray(hyperplanes)
    if not np.all(A == np.rint(A)):
        raise SolverFailure(
            "CP-SAT is an integer solver and the constraint matrix is not "
            "integral. Truncating it would silently change the problem."
        )
    A = np.rint(A).astype(int)

    # A and x are integral, so A@x is an integer and `A@x >= c` is equivalent
    # to `A@x >= ceil(c)`. Taking the ceiling is exact, and it is also
    # necessary: CP-SAT's `>=` rejects a float bound outright, so a caller
    # passing c=1.0 rather than c=1 used to fail with a pybind TypeError.
    bound = math.ceil(c)

    solver = cp_model.CpSolver()
    model = cp_model.CpModel()

    lower = cp_model.INT32_MIN if lower_bound is None else int(math.ceil(lower_bound))
    var = [
        model.new_int_var(lower, cp_model.INT32_MAX, f"x_{i}")
        for i in range(ambient_dim)
    ]

    for row in A:
        model.add(sum(int(row[i]) * var[i] for i in range(ambient_dim)) >= bound)

    obj_vec = A.sum(axis=0)
    obj_vec //= gcd_int(obj_vec)
    model.minimize(sum(var[i] * int(obj_vec[i]) for i in range(ambient_dim)))

    status = solver.solve(model)
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return np.array([solver.value(x) for x in var])
    if status == cp_model.INFEASIBLE:
        return None
    raise SolverFailure(
        f"CP-SAT returned status {solver.status_name(status)}, which is not a proof."
    )
