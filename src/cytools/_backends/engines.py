# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""The engine registries, one per mathematical task.

Order within a registry is by measured cost for a typical problem, cheapest
first. Order is *not* a correctness mechanism: an engine that does not provide
a guarantee a call site requires is excluded outright, wherever it sits.

Availability predicates use `importlib.util.find_spec`, which resolves a
package without executing its `__init__`. Importing an optional engine merely
to discover whether it exists would defeat the point, and in one case
(PyTorch, via the `gnn` extra) can abort the interpreter -- see
`cytools._backends.openmp`.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import Any

from cytools._backends import hull, linsolve, lp, qp, triangulate
from cytools._backends.crossovers import CROSSOVERS
from cytools._backends.registry import (
    CERTIFIES_INFEASIBLE,
    DETERMINISTIC,
    EXACT,
    RECOVERABLE,
    REGULAR,
    Engine,
    Registry,
)

__all__ = [
    "CONVEX_HULL",
    "INTERIOR_POINT",
    "LINEAR_SOLVE",
    "STRETCHED_TIP",
    "TRIANGULATE",
    "all_registries",
]


# availability
# ------------
def _installed(module: str):
    """A predicate reporting whether `module` can be imported."""

    def check() -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    return check


def _always() -> bool:
    return True


def _mosek_activated() -> bool:
    """Whether Mosek is both installed and holding a valid licence.

    Deferred import: `cytools.config` is domain-adjacent, and this is only
    reached at resolution time.
    """
    if importlib.util.find_spec("mosek") is None:
        return False
    import cytools.config as config

    return bool(config.mosek_is_activated())


def _qpsolver(name: str):
    """Whether qpsolvers can actually dispatch to ``name``.

    Looking only for the solver's import package is insufficient: a package
    can be installed while its qpsolvers adapter is unavailable or broken.
    Resolution is the right time to import the lightweight dispatcher and ask
    it for the engines it can use.
    """

    def check() -> bool:
        try:
            import qpsolvers
        except (ImportError, OSError):
            return False
        return name in qpsolvers.available_solvers

    return check


# applicability and preference
# ----------------------------
def _dim_equals(key: str, wanted: int):
    def applies(problem: Mapping[str, Any]) -> bool:
        return problem.get(key) == wanted

    return applies


def _dim_at_least_2(problem: Mapping[str, Any]) -> bool:
    """QHull rejects configurations below two dimensions outright."""
    dim = problem.get("dim")
    return dim is None or dim >= 2


_TIP_X = CROSSOVERS["stretched_tip.osqp_to_mosek"]


def _tip_preference(engine: str, *, fallback: float = 0):
    """Rank one side of the tip crossover without restricting support."""

    def preference(problem: Mapping[str, Any]) -> float:
        value = problem.get(_TIP_X.metric)
        if value is None:
            return fallback
        preferred = _TIP_X.above if value >= _TIP_X.value else _TIP_X.below
        return 0 if engine == preferred else 1

    return preference


# convex hull
# ===========
# PPL and PALP are both exact; QHull is not, and so is never selected for
# lattice work no matter how fast it is. The one-dimensional closed form goes
# first because it applies to exactly one problem shape and is free there.
#
# PALP does not provide RECOVERABLE: it is a C program with compile-time array
# bounds and calls abort() when a configuration exceeds them, which kills the
# interpreter outright. The isolated calibration found no stable dimensional
# crossover: PPL won every nontrivial cell through dimension five, PALP won one
# dense six-dimensional cell, then PALP began aborting at dimensions eight and
# above. PPL therefore stays ahead in the automatic order. PALP remains
# registered for explicit reproduction of historical runs.
CONVEX_HULL = Registry(
    task="convex_hull",
    engines=(
        Engine(
            name="interval",
            run=hull.interval_hull,
            provides=frozenset({EXACT, DETERMINISTIC, RECOVERABLE}),
            applies=_dim_equals("dim", 1),
        ),
        Engine(
            name="ppl",
            run=hull.ppl_hull,
            provides=frozenset({EXACT, DETERMINISTIC, RECOVERABLE}),
            available=_installed("ppl"),
            why_unavailable="pplpy is not installed",
        ),
        Engine(
            name="palp",
            run=hull.palp_hull,
            # no RECOVERABLE: aborts the process past its compiled-in limits
            provides=frozenset({EXACT, DETERMINISTIC}),
            available=_installed("pypalp"),
            why_unavailable="pypalp is not installed",
        ),
        Engine(
            name="qhull",
            run=hull.qhull_hull,
            # deliberately no EXACT: floating point, with a rounding step
            provides=frozenset({DETERMINISTIC, RECOVERABLE}),
            available=_installed("scipy"),
            applies=_dim_at_least_2,
            why_unavailable="scipy is not installed",
        ),
    ),
)


# interior point / LP feasibility
# ===============================
# HiGHS, GLOP, and SCIP confirm every negative result with PPL before returning
# ``None``. CP-SAT searches integer points in a fixed-width box, so it cannot
# certify that a rational region is empty; it keeps EXACT integer arithmetic
# but not CERTIFIES_INFEASIBLE.
INTERIOR_POINT = Registry(
    task="interior_point",
    engines=(
        Engine(
            name="highs",
            run=lp.highs_feasibility,
            provides=frozenset({CERTIFIES_INFEASIBLE, DETERMINISTIC, RECOVERABLE}),
            available=_installed("highspy"),
            why_unavailable="highspy is not installed",
        ),
        Engine(
            name="glop",
            run=lp.glop_feasibility,
            provides=frozenset({CERTIFIES_INFEASIBLE, DETERMINISTIC, RECOVERABLE}),
            available=_installed("ortools"),
            why_unavailable="ortools is not installed",
        ),
        Engine(
            name="scip",
            run=lp.scip_feasibility,
            provides=frozenset({CERTIFIES_INFEASIBLE, DETERMINISTIC, RECOVERABLE}),
            available=_installed("ortools"),
            why_unavailable="ortools is not installed",
        ),
        Engine(
            name="cpsat",
            run=lp.cpsat_feasibility,
            provides=frozenset({EXACT, DETERMINISTIC, RECOVERABLE}),
            available=_installed("ortools"),
            why_unavailable="ortools is not installed",
        ),
    ),
)


# tip of the stretched cone
# =========================
# HiGHS is first because it is a hard dependency and the only adapter here
# whose negative result is confirmed exactly. Its primal-dual result is also
# validated independently before it is returned. The other engines remain
# useful explicit compatibility and differential-test choices, but ``None``
# from them can mean that the solver gave up.
#
# The osqp/mosek crossover consequently no longer decides anything automatic,
# which is the right outcome for a threshold that was inherited unmeasured.
# It still records their relative order in problem-specific candidate lists.
#
# The LP engines are not registered here because they minimise a linear
# functional and return *a* point of the stretched cone rather than its
# minimum-norm point.
STRETCHED_TIP = Registry(
    task="stretched_tip",
    engines=(
        Engine(
            name="highs",
            run=qp.highs_tip,
            provides=frozenset({CERTIFIES_INFEASIBLE, DETERMINISTIC, RECOVERABLE}),
            available=_always,  # highspy is a hard dependency
        ),
        Engine(
            name="mosek",
            run=qp.mosek_tip,
            provides=frozenset({DETERMINISTIC, RECOVERABLE}),
            available=lambda: _mosek_activated() and _qpsolver("mosek")(),
            preference=_tip_preference("mosek"),
            why_unavailable="Mosek is not installed or its licence is not active",
        ),
        Engine(
            name="osqp",
            run=qp.osqp_tip,
            provides=frozenset({DETERMINISTIC, RECOVERABLE}),
            available=_qpsolver("osqp"),
            preference=_tip_preference("osqp"),
            why_unavailable="osqp is not installed",
        ),
        Engine(
            name="cvxopt",
            run=qp.cvxopt_tip,
            provides=frozenset({DETERMINISTIC, RECOVERABLE}),
            available=_qpsolver("cvxopt"),
            preference=lambda problem: 2,
            why_unavailable=(
                "CVXOPT is unavailable; install the "
                "`cytools-workbench[cvxopt]` optional extra"
            ),
        ),
    ),
)


# triangulation
# =============
# ``triangulumancer`` is the Python boundary for both native engines: its
# height-induced triangulation uses CGAL and its fine triangulation uses
# TOPCOM. Only ``heights`` carries a height certificate here. QHull performs
# the lifted-hull construction in floating point and remains for explicit
# compatibility and differential testing.
TRIANGULATE = Registry(
    task="triangulate",
    engines=(
        Engine(
            name="heights",
            run=triangulate.heights_triangulate,
            provides=frozenset({REGULAR, DETERMINISTIC, RECOVERABLE}),
            available=_installed("triangulumancer"),
            applies=lambda problem: problem.get("heights", True) is not None,
            why_unavailable="triangulumancer is not installed",
        ),
        Engine(
            name="fine",
            run=triangulate.fine_triangulate,
            provides=frozenset({DETERMINISTIC, RECOVERABLE}),
            available=_installed("triangulumancer"),
            why_unavailable="triangulumancer is not installed",
        ),
        Engine(
            name="qhull",
            run=triangulate.qhull_triangulate,
            provides=frozenset({RECOVERABLE}),
            available=_installed("scipy"),
            applies=lambda problem: problem.get("heights", True) is not None,
            why_unavailable="scipy is not installed",
        ),
    ),
)


# sparse linear solve
# ===================
LINEAR_SOLVE = Registry(
    task="linear_solve",
    engines=(
        Engine(
            name="cholmod",
            run=linsolve.cholmod_solve,
            provides=frozenset({DETERMINISTIC, RECOVERABLE}),
            available=_installed("sksparse"),
            why_unavailable=(
                "scikit-sparse is not installed; it needs SuiteSparse headers "
                "and ships in the `cytools-workbench[performance]` extra"
            ),
        ),
        Engine(
            name="superlu",
            run=linsolve.superlu_solve,
            provides=frozenset({DETERMINISTIC, RECOVERABLE}),
            available=_always,
        ),
    ),
)


def all_registries() -> tuple[Registry, ...]:
    """Every registry, for introspection and for tests that sweep them."""
    return (CONVEX_HULL, INTERIOR_POINT, STRETCHED_TIP, TRIANGULATE, LINEAR_SOLVE)
