"""
Differential tests for `cytools.helpers.arithmetic`.

Each test pins the new helper against the *exact* expression it replaced,
spelled out inline, rather than against recorded values. A refactor that
changes a result therefore fails here regardless of what the result is.
"""

import math

import numpy as np
import pytest

from cytools.helpers.arithmetic import gcd_reduce, primitive


# The spellings that existed at the call sites before this module. They are
# reproduced verbatim -- that is the whole point of a differential test -- so
# the bound `np.gcd.reduce`, whose numpy stub lacks a matching overload, is
# quarantined in this one function rather than repeated at each use.
def _old_reduce(a, axis=None):
    return np.gcd.reduce(np.abs(a), axis=axis)


def _old_flat(a):
    return int(_old_reduce(a))


def _old_axis(a, axis):
    return _old_reduce(a, axis)


def _old_primitive_floordiv(a, axis):
    """utils / ntfe / triangulation: exact, with an explicit zero guard."""
    gcds = _old_reduce(a, axis)
    gcds[gcds == 0] = 1
    return a // np.expand_dims(gcds, axis)


def _old_primitive_float(v):
    """f_theory: float division then rint. Exact only when gcd != 0."""
    return np.rint(v / _old_reduce(v)).astype(int)


@pytest.fixture
def rng():
    return np.random.default_rng(20260831)


# ----------------------------------------------------------------- gcd_reduce
def test_gcd_reduce_flat_matches_old(rng):
    for _ in range(200):
        a = rng.integers(-1000, 1000, size=rng.integers(1, 50))
        assert int(gcd_reduce(a)) == _old_flat(a)


@pytest.mark.parametrize("axis", [0, 1])
def test_gcd_reduce_axis_matches_old(rng, axis):
    for _ in range(100):
        m = rng.integers(-500, 500, size=(rng.integers(1, 30), rng.integers(1, 30)))
        assert np.array_equal(gcd_reduce(m, axis=axis), _old_axis(m, axis))


def test_gcd_reduce_is_sign_insensitive(rng):
    m = rng.integers(-500, 500, size=(40, 12))
    assert np.array_equal(gcd_reduce(m, axis=1), gcd_reduce(np.abs(m), axis=1))


def test_gcd_reduce_agrees_with_math_gcd(rng):
    """Independent oracle: stdlib math.gcd, exact and arbitrary precision."""
    for _ in range(200):
        a = rng.integers(-(10**6), 10**6, size=rng.integers(1, 20))
        assert int(gcd_reduce(a)) == math.gcd(*a.tolist())


def test_gcd_reduce_zero_vector_is_zero():
    assert int(gcd_reduce(np.zeros(5, dtype=np.int64))) == 0
    assert np.array_equal(
        gcd_reduce(np.zeros((3, 4), dtype=np.int64), axis=1), np.zeros(3)
    )


def test_gcd_reduce_single_element():
    assert int(gcd_reduce(np.array([-7]))) == 7


# ------------------------------------------------------------------ primitive
@pytest.mark.parametrize("axis", [0, 1])
def test_primitive_matches_old_floordiv(rng, axis):
    for _ in range(100):
        m = rng.integers(-200, 200, size=(rng.integers(1, 25), rng.integers(1, 25)))
        assert np.array_equal(primitive(m, axis=axis), _old_primitive_floordiv(m, axis))


def test_primitive_matches_old_float_spelling_on_nonzero(rng):
    """The f_theory spelling agrees wherever it was well defined (gcd != 0)."""
    for _ in range(300):
        v = rng.integers(-500, 500, size=rng.integers(2, 15))
        if _old_flat(v) == 0:
            continue
        assert np.array_equal(primitive(v), _old_primitive_float(v))


def test_primitive_result_is_primitive(rng):
    """The defining property: the gcd of the result is 1."""
    for _ in range(200):
        v = rng.integers(-500, 500, size=rng.integers(2, 15))
        if int(gcd_reduce(v)) == 0:
            continue
        assert int(gcd_reduce(primitive(v))) == 1


def test_primitive_rows_are_primitive(rng):
    m = rng.integers(-300, 300, size=(60, 9))
    m[0] = 0  # keep a zero row in the mix
    out = primitive(m, axis=1)
    gcds = gcd_reduce(out, axis=1)
    assert np.array_equal(gcds[1:], np.ones(len(m) - 1))
    assert np.array_equal(out[0], np.zeros(9))


def test_primitive_preserves_direction(rng):
    """Primitivization scales by a positive factor; signs must not flip."""
    for _ in range(200):
        v = rng.integers(-500, 500, size=rng.integers(2, 12))
        if int(gcd_reduce(v)) == 0:
            continue
        assert np.array_equal(np.sign(primitive(v)), np.sign(v))


def test_primitive_zero_vector_is_defined():
    """The old float spelling hit nan -> int here, which numpy leaves undefined."""
    assert np.array_equal(primitive(np.zeros(4, dtype=np.int64)), np.zeros(4))


def test_primitive_is_exact_beyond_float64_mantissa():
    """The float spelling this replaced is wrong once a quotient exceeds 2**53.

    Both entries below come back wrong under `np.rint(v / gcd).astype(int)`:
    the quotients are not representable in float64. Reaching this needs
    coordinates far larger than anything in the Kreuzer-Skarke database, so
    this pins exactness rather than fixing an observed failure.
    """
    v = np.array([2 * (2**54 + 1), 2 * (2**54 + 3)], dtype=np.int64)
    assert np.array_equal(primitive(v), [2**54 + 1, 2**54 + 3])
    assert not np.array_equal(primitive(v), _old_primitive_float(v))


def test_math_lcm_replaces_np_lcm_reduce_because_int64_overflows():
    """Why polytope/Uplift call `math.lcm`: `np.lcm.reduce` wraps, silently.

    The GLSM row/column scalings are lcms of `fmpq` denominators. On int64
    the reduction overflows to a *negative* scaling with no error raised;
    `math.lcm` is arbitrary precision.
    """
    dens = [982451653, 961748941, 899809343, 878492759]  # pairwise coprime
    assert math.lcm(*dens) == math.prod(dens)
    assert int(np.lcm.reduce(np.array(dens))) < 0


def test_primitive_is_idempotent(rng):
    m = rng.integers(-300, 300, size=(40, 7))
    once = primitive(m, axis=1)
    assert np.array_equal(primitive(once, axis=1), once)
