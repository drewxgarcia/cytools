"""The stretched-cone tip, checked against closed forms rather than a solver.

Every expected value here comes from an argument, not from a run: the two
fixtures are cone families whose minimum-norm point can be written down. That
matters because the failure this module pins was invisible to differential
testing against the previous engine -- both engines agreed, and both were
wrong, until the ambient dimension grew.
"""

import numpy as np
import pytest

from cytools import Cone
from cytools._backends.engines import STRETCHED_TIP
from cytools._backends.lp import SolverFailure
from cytools._backends.qp import certify_tip, highs_tip, osqp_tip
from cytools._backends.registry import CERTIFIES_INFEASIBLE, RECOVERABLE


def doubling_chain(n: int) -> np.ndarray:
    """Hyperplanes of a cone whose tip has norm ~2^n.

    Row 0 is `x_0 >= 1`; row `i+1` is `x_{i+1} - 2 x_i >= 1`. Each pairing
    forces the next coordinate past twice the previous one, so the tip runs
    away exponentially in `n` while every coefficient stays a small integer --
    the same regime a real Kahler cone reaches at large h11, reproduced
    without needing the database.
    """
    A = np.zeros((n, n), dtype=int)
    A[0, 0] = 1
    for i in range(n - 1):
        A[i + 1, i + 1] = 1
        A[i + 1, i] = -2
    return A


def doubling_chain_tip(n: int) -> np.ndarray:
    """The exact tip of `doubling_chain(n)`, at stretching 1.

    `x_0 >= 1` forces `x_0 = 1` at best, and `x_{i+1} >= 1 + 2 x_i` then
    forces `x_i >= 2^(i+1) - 1` by induction, with equality attainable for
    every coordinate at once. A point that minimises every coordinate
    simultaneously minimises the norm, so this is the tip -- and it is unique,
    because the objective is strictly convex.
    """
    return np.array([2.0 ** (i + 1) - 1 for i in range(n)])


def polygon_cone(m: int, tilt: float = 0.4) -> np.ndarray:
    """Hyperplanes of the cone over a regular `m`-gon, normals tilted by `tilt`.

    The tip is `(0, 0, 1 / tilt)`: the region is invariant under the m-fold
    rotation about the z-axis, a strictly convex objective has a unique
    minimiser, and a unique minimiser of a symmetric problem must be fixed by
    the symmetry -- so it lies on the axis, where the constraints reduce to
    `tilt * z >= 1`.
    """
    angles = 2 * np.pi * np.arange(m) / m
    return np.column_stack([np.cos(angles), np.sin(angles), np.full(m, float(tilt))])


# the certificate
# ---------------
def test_dual_bound_is_valid_for_every_nonnegative_multiplier():
    """The lower bound must hold for multipliers no solver would produce.

    This is what makes the certificate independent of the solver: the dual
    objective bounds the optimum from below at *any* `lam >= 0`, so a solver
    that stopped early still hands over usable evidence. If this property
    failed, a passing certificate would prove nothing.
    """
    rng = np.random.default_rng(20260902)
    for A, optimum in (
        (np.eye(5), 5.0),
        (doubling_chain(8), float(doubling_chain_tip(8) @ doubling_chain_tip(8))),
    ):
        x = np.zeros(A.shape[1])  # the bound may not reference a good primal
        for scale in (1e-3, 1.0, 1e3):
            for _ in range(50):
                lam = rng.exponential(scale, size=A.shape[0])
                bound = certify_tip(A, 1, x, lam).dual_bound
                assert bound <= optimum * (1 + 1e-9) + 1e-9


def test_certificate_separates_the_tip_from_its_near_misses():
    """A feasible non-tip and an infeasible low-norm point both fail."""
    A, c = np.eye(4), 3.0
    tip, multipliers = np.full(4, c), np.full(4, 2 * c)

    exact = certify_tip(A, c, tip, multipliers)
    assert exact.holds()
    assert exact.objective == pytest.approx(4 * c**2)
    assert exact.dual_bound == pytest.approx(4 * c**2)
    assert exact.worst_violation >= 0

    # Feasible, but twice as far out as it needs to be: the gap opens upward.
    too_far = certify_tip(A, c, 2 * tip, multipliers)
    assert not too_far.holds()
    assert too_far.relative_gap > 0

    # Infeasible, and therefore *below* the true minimum -- the direction a
    # first-order method stopped early actually errs in, which is why a norm
    # comparison alone cannot catch it.
    too_close = certify_tip(A, c, 0.5 * tip, multipliers)
    assert not too_close.holds()
    assert too_close.relative_gap < 0
    assert too_close.worst_violation < 0


def test_an_unconstrained_region_is_vacuously_feasible():
    """`min` of no residuals is not zero, and must not raise."""
    certificate = certify_tip(np.zeros((0, 3)), 1, np.zeros(3), np.zeros(0))
    assert certificate.worst_violation == float("inf")
    assert certificate.holds()


# closed-form tips
# ----------------
@pytest.mark.parametrize("n", [4, 12, 20, 24])
def test_highs_finds_the_analytic_tip_of_the_doubling_chain(n):
    """Exact agreement with the closed form, over four orders of tip norm."""
    tip = highs_tip(doubling_chain(n), 1)
    assert tip is not None
    assert np.allclose(tip, doubling_chain_tip(n), rtol=1e-9)


@pytest.mark.parametrize("m", [3, 6, 12, 40])
def test_highs_finds_the_symmetric_tip_of_a_polygon_cone(m):
    tip = highs_tip(polygon_cone(m), 1)
    assert tip is not None
    assert np.allclose(tip, [0, 0, 2.5], atol=1e-9)


def test_the_public_tip_survives_a_norm_that_defeats_a_first_order_method():
    """The regression, at the public API.

    At n = 20 the tip has norm 1.2e6, so an absolute residual tolerance of
    1e-4 is nine orders of magnitude finer than the answer and the previous
    default engine ran to its iteration cap without converging. This test
    fails against that engine and passes against a certified one, which is the
    whole reason the registry order changed.
    """
    cone = Cone(hyperplanes=doubling_chain(20), check=False)
    assert np.allclose(cone.tip_of_stretched_cone(1), doubling_chain_tip(20), rtol=1e-9)


def test_highs_and_osqp_agree_where_the_first_order_method_converges():
    """Differential check in the regime both engines can handle.

    Kept small deliberately: the point is that the new default reproduces the
    old one exactly where the old one worked, so the change is an extension
    rather than a different answer.
    """
    for A in (np.eye(2), np.array([[3, 2], [5, 3]]), doubling_chain(6)):
        certified = highs_tip(A, 1)
        first_order = osqp_tip(A, 1)
        assert first_order is not None, "fixture chosen to be within OSQP's reach"
        assert np.allclose(certified, first_order, rtol=1e-4, atol=1e-6)


# the contract
# ------------
def test_an_empty_stretched_region_is_the_only_reason_for_none():
    """`x >= 1` and `-x >= 1` cannot both hold, and nothing else returns None."""
    assert highs_tip(np.array([[1], [-1]]), 1) is None
    assert highs_tip(np.array([[1, 0], [-1, 0], [0, 1]]), 1) is None
    assert highs_tip(doubling_chain(6), 1) is not None


def test_a_solver_failure_on_a_nonempty_region_raises(monkeypatch):
    """A solver giving up must not masquerade as an empty region.

    `Cone.is_solid` reads a missing point as a statement about dimension, so
    an engine that conflated the two would turn a numerical failure into a
    false claim about the geometry.
    """
    import qpsolvers

    class Unsolved:
        found, x, z = False, None, None

    monkeypatch.setattr(qpsolvers, "solve_problem", lambda *args, **kwargs: Unsolved())
    with pytest.raises(SolverFailure, match="nonempty"):
        highs_tip(np.eye(3), 1)


def test_an_uncertified_point_raises_rather_than_being_returned(monkeypatch):
    import qpsolvers

    class Wrong:
        found = True
        x = np.full(3, 0.5)  # infeasible for `x >= 1`, and below the true norm
        z = np.zeros(3)

    monkeypatch.setattr(qpsolvers, "solve_problem", lambda *args, **kwargs: Wrong())
    with pytest.raises(SolverFailure, match="uncertified"):
        highs_tip(np.eye(3), 1)


def test_max_iter_reaches_highs_as_an_option_it_recognises():
    """`max_iter` is not a HiGHS option; `qp_iteration_limit` is.

    Passing the former sets nothing at all, which is how an iteration cap came
    to be silently ignored. A cap of one is enough to stop the active-set
    method on a cone whose optimal active set takes several steps to find, so
    this test would fail if the mapping were dropped.
    """
    import highspy

    solver = highspy.Highs()
    solver.silent()
    assert solver.getOptionType("qp_iteration_limit")[0] == highspy.HighsStatus.kOk
    assert solver.getOptionType("max_iter")[0] != highspy.HighsStatus.kOk

    A = polygon_cone(12)
    assert highs_tip(A, 1, max_iter=10**6) is not None
    with pytest.raises(SolverFailure, match="nonempty"):
        highs_tip(A, 1, max_iter=1)


def test_the_certifying_engine_is_the_default_at_every_size():
    """No dimensional threshold stands between a caller and a certificate."""
    for dim, rows in ((2, 2), (24, 200), (300, 2320)):
        problem = {"dim": dim, "rows": rows}
        assert STRETCHED_TIP.resolve(need=(RECOVERABLE,), problem=problem).name == (
            "highs"
        )
        assert (
            STRETCHED_TIP.resolve(need=(CERTIFIES_INFEASIBLE,), problem=problem).name
            == "highs"
        )

    certifying = [
        engine.name
        for engine in STRETCHED_TIP.engines
        if CERTIFIES_INFEASIBLE in engine.provides
    ]
    assert certifying == ["highs"]
