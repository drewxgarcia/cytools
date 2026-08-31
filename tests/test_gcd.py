"""Tests for gcd_float / gcd_list.

`gcd_float` is a float Euclidean gcd with a tolerance, and its own docstring
warns that it can be wrong. These tests pin the precondition precisely, because
the vague version ("buggy if b starts tiny") invites both panic and
complacency, and pin the behaviour the callers actually rely on.
"""

import functools
import math

import numpy as np
import pytest

from cytools.utils import gcd_float, gcd_list


def float_path(arr):
    """The pre-fast-path implementation, for differential comparison."""
    return functools.reduce(gcd_float, arr)


# ---------------------------------------------------------------------------
# What the precondition actually is
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exponent", [1, 2, 3, 4, 5, 6, 7])
def test_a_large_ratio_alone_is_harmless(exponent):
    """A wide spread of *integers* is fine; only small magnitudes break it.

    Corrects the natural misreading of the docstring. What matters is the
    smallest nonzero magnitude against `tol`, not max/min.
    """
    values = [1.0, float(10**exponent)]
    assert gcd_float(*values) == 1.0
    assert float_path(values) == 1.0


def test_the_failure_needs_a_genuinely_sub_tolerance_component():
    """Pins the actual failure, so it is discoverable rather than folklore.

    A unit-normalized [1, 100000] has a component at 1e-5, at the tolerance, and
    the small entry is silently dropped: the ray becomes [0, 1].
    """
    v = np.array([1.0, 100000.0])
    v = v / np.linalg.norm(v)
    g = gcd_list(v)
    recovered = [int(round(x / g)) for x in v]

    assert recovered == [0, 1], (
        "expected the sub-tolerance component to be dropped; if this now gives "
        "[1, 100000] the tolerance behaviour changed and the docstring needs "
        "updating"
    )


def test_svd_noise_is_collapsed_to_zero():
    """The tolerance is wanted here: 1e-17 is a structural zero, not a divisor.

    This is why the float path is kept for float input rather than being
    replaced by exact arithmetic.
    """
    v = np.array([0.5, 0.5, -0.5, -0.5, -3.2e-17])
    g = gcd_list(v)
    assert [int(round(x / g)) for x in v] == [1, 1, -1, -1, 0]


# ---------------------------------------------------------------------------
# The exact fast path must agree with the float path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        [4, 6, 10],
        [-4, 6, -10],
        [0, 4, 6],
        [7],
        [1, 10**7],
        [2**40, 2**41],
        [12, 18, 24, 30],
    ],
    ids=str,
)
def test_integer_input_matches_the_float_path(values):
    exact = gcd_list(np.asarray(values, dtype=np.int64))
    floaty = float_path([float(v) for v in values])
    assert exact == pytest.approx(floaty)

    nz = [abs(v) for v in values if v]
    if nz:
        assert exact == math.gcd(*nz)


def test_integer_lists_take_the_exact_path():
    assert gcd_list([4, 6, 10]) == 2
    assert isinstance(gcd_list([4, 6, 10]), int)


def test_integer_arrays_of_any_shape():
    assert gcd_list(np.array([[4, 6], [10, 8]])) == 2


def test_all_zero_and_empty():
    assert gcd_list(np.array([0, 0])) == 0
    assert gcd_list([0, 0]) == 0
    assert gcd_list([]) == 0
    assert gcd_list(np.array([], dtype=float)) == 0


def test_float_input_keeps_returning_a_float():
    """Callers divide by the result; the return type should not shift under them."""
    assert isinstance(gcd_list(np.array([0.5, 0.5])), float)


# ---------------------------------------------------------------------------
# Randomized differential check
# ---------------------------------------------------------------------------


def test_randomized_agreement_between_paths():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        v = rng.integers(-40, 41, size=int(rng.integers(2, 9)))
        if not v.any():
            continue
        exact = gcd_list(v)
        floaty = float_path(v.astype(float))
        assert exact == pytest.approx(floaty), f"{v.tolist()}"

        # and the integers each implies must match
        assert [int(round(x / exact)) for x in v] == [int(round(x / floaty)) for x in v]
