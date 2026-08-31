"""
Tests for local Cartier data: `Cartier_index` and `is_Cartier`.

The fan is stubbed. Both functions need only `cones()` and `vectors(cone)`,
and the interesting cases are small cones whose exact answer can be written
down by hand -- an A_1 singularity, a singular cone admitting no local data --
which is far clearer than steering a real `Fan` into those shapes.

Each test that pins a fix records what the previous implementation returned,
so the regression is legible without running it.
"""

from fractions import Fraction

import numpy as np
import pytest

from cytools.f_theory.Uplift_functions import (
    Cartier_index,
    _cartier_index_of,
    _local_cartier_data,
    is_Cartier,
)


class StubFan:
    """The two methods `Cartier_index` / `is_Cartier` ask a fan for."""

    def __init__(self, cones_to_rays):
        self._cones_to_rays = cones_to_rays

    def cones(self):
        return list(self._cones_to_rays)

    def vectors(self, cone):
        return np.array(self._cones_to_rays[cone])


SMOOTH = StubFan({(1, 2): [[1, 0], [0, 1]]})
#: The A_1 singularity: rays (1,0) and (1,2), determinant 2.
A1 = StubFan({(1, 2): [[1, 0], [1, 2]]})
#: Square and singular, so `lstsq` reports no residuals at all.
SINGULAR = StubFan({(1, 2): [[1, 1], [1, 1]]})


# --------------------------------------------------------- _cartier_index_of
@pytest.mark.parametrize(
    "fractions, expected",
    [
        ([Fraction(1)], 1),
        ([Fraction(1, 3), Fraction(2, 3)], 3),
        ([Fraction(1, 2), Fraction(1, 3)], 6),
        ([Fraction(3, 4), Fraction(5, 6)], 12),
    ],
)
def test_cartier_index_of_clears_denominators(fractions, expected):
    assert _cartier_index_of(fractions) == expected


def test_cartier_index_of_is_exact_for_large_denominators():
    """The scanning loop this replaced returned 999973 here, not 999983.

    `np.isclose` is a *relative* test, so scanning `n` until `n * y` looks
    integral raises the tolerance in step with `n` and eventually accepts a
    near-miss. Reading the denominator off the rational cannot drift.
    """
    assert _cartier_index_of([Fraction(1, 999983)]) == 999983


def test_cartier_index_of_result_actually_clears():
    """Defining property: multiplying by the index yields integers."""
    for denominators in [(2, 3), (4, 6), (7, 11), (999983,)]:
        data = [Fraction(1, d) for d in denominators]
        n = _cartier_index_of(data)
        assert all((n * f).denominator == 1 for f in data)


# -------------------------------------------------------- _local_cartier_data
def test_local_data_on_a_smooth_cone_is_integral():
    data = _local_cartier_data(SMOOTH, np.array([0, 1]), (1, 2))
    assert data is not None
    assert _cartier_index_of(data) == 1


def test_local_data_on_the_a1_singularity_is_half_integral():
    data = _local_cartier_data(A1, np.array([0, 1]), (1, 2))
    assert data is not None
    assert _cartier_index_of(data) == 2


def test_local_data_is_none_when_the_system_is_inconsistent():
    """A singular cone with an unsolvable system has no local data.

    `lstsq` returns an *empty* residuals array for square and rank-deficient
    systems, so the `sum(residuals) < 1e-10` test this replaced was satisfied
    unconditionally and accepted exactly this input.
    """
    assert _local_cartier_data(SINGULAR, np.array([2, 6]), (1, 2)) is None


# --------------------------------------------------------------- Cartier_index
def test_cartier_index_smooth_cone_is_one():
    assert Cartier_index(SMOOTH, [0, 1]) == 1


def test_cartier_index_a1_singularity_is_two():
    assert Cartier_index(A1, [0, 1]) == 2
    # doubling the weights clears the denominator
    assert Cartier_index(A1, [0, 2]) == 1


def test_cartier_index_is_none_without_local_data():
    """Previously returned 70 -- an index for data that does not exist."""
    assert Cartier_index(SINGULAR, [2, 6]) is None


def test_cartier_index_is_the_lcm_across_cones():
    fan = StubFan(
        {
            (1, 2): [[1, 0], [1, 2]],  # index 2
            (3, 4): [[1, 0], [1, 3]],  # index 3
        }
    )
    assert Cartier_index(fan, [0, 1, 0, 1]) == 6


# ------------------------------------------------------------------ is_Cartier
def test_is_cartier_true_on_a_smooth_cone():
    cartier, data = is_Cartier(SMOOTH, [0, 1])
    assert cartier is True
    assert all(np.asarray(d).dtype.kind == "i" for d in data)


def test_is_cartier_false_on_the_a1_singularity():
    assert is_Cartier(A1, [0, 1])[0] is False


def test_is_cartier_false_when_no_local_data_exists():
    """Regression: this returned True.

    The least-squares solution is integral, so the integrality half of the old
    test passed, and the residual half was vacuous -- yet `arr @ y` is `[4, 4]`
    against a right-hand side of `[2, 6]`, so no local data exists at all.
    """
    assert is_Cartier(SINGULAR, [2, 6])[0] is False


def test_is_cartier_reports_none_for_cones_without_local_data():
    cartier, data = is_Cartier(SINGULAR, [2, 6], return_Q_Cartier_data=True)
    assert cartier is False
    assert data == [None]


def test_is_cartier_returns_q_cartier_data_when_asked():
    cartier, data = is_Cartier(A1, [0, 1], return_Q_Cartier_data=True)
    assert cartier is False
    assert len(data) == 1 and data[0] is not None


def test_is_cartier_agrees_with_cartier_index():
    """`is_Cartier` is exactly the index-one case."""
    for fan, weights in [
        (SMOOTH, [0, 1]),
        (A1, [0, 1]),
        (A1, [0, 2]),
        (SINGULAR, [2, 6]),
    ]:
        assert is_Cartier(fan, weights)[0] == (Cartier_index(fan, weights) == 1)
