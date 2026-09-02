"""Predicates that state exact mathematics must be decided exactly.

A cone is smooth iff its ray matrix is unimodular. That is a question about
integers with a yes/no answer, so a floating-point answer to it is not a
slower answer, it is sometimes a different one.
"""

import numpy as np
import pytest
from flint import fmpz_mat

from cytools import Cone


def unimodular(n: int, mixes: int = 3, seed: int = 0) -> np.ndarray:
    """An integer matrix of determinant exactly 1, with large entries.

    Built as a product of unit-triangular integer matrices, each of which has
    determinant 1, so the product does too however big its entries grow. The
    growth is the point: it is what separates the exact answer from the
    floating-point one.
    """
    rng = np.random.default_rng(seed)
    matrix = np.eye(n, dtype=object)
    for _ in range(mixes):
        for indices in (np.tril_indices(n, -1), np.triu_indices(n, 1)):
            factor = np.eye(n, dtype=object)
            factor[indices] = rng.integers(-2, 3, size=len(indices[0]))
            matrix = matrix @ factor
    return matrix


@pytest.mark.parametrize("n", [4, 8, 16, 24, 32])
def test_smoothness_survives_a_determinant_float64_cannot_hold(n):
    """The regression, over the range where float64 gives up.

    At n = 4 and 8 both arithmetics agree. From n = 16 the float determinant
    of these matrices reads 1.19, and at n = 32 it reads -4.5e29, so this test
    fails at every size from 16 up against a float implementation while the
    cone is smooth at every size.
    """
    rays = unimodular(n)
    assert int(fmpz_mat(rays.tolist()).det()) == 1, "fixture must be unimodular"

    cone = Cone(rays=rays.astype(np.int64))
    assert cone.is_smooth() is True


@pytest.mark.parametrize("n", [16, 24, 32])
def test_dimension_of_an_ill_conditioned_cone_is_exact(n):
    """Rank fails before the determinant does, and takes the geometry with it.

    `numpy.linalg.matrix_rank` compares singular values against
    `S.max() * max(M, N) * eps`, so on these matrices -- condition number
    ~2e17 -- it reports 15, 22 and 29 for what are exactly rank 16, 24 and 32.
    A cone then calls itself lower-dimensional than its ambient space, which
    makes `is_solid` false, `dim` wrong, and `is_smooth` never reach the
    determinant branch at all.
    """
    rays = unimodular(n)
    cone = Cone(rays=rays.astype(np.int64))

    assert np.linalg.matrix_rank(rays.astype(float)) < n, (
        "fixture must be ill-conditioned"
    )
    assert cone.dim() == n
    assert cone.is_solid() is True


def test_a_non_unimodular_cone_is_still_reported_singular():
    """Exactness must not turn into a rubber stamp."""
    rays = unimodular(16)
    rays[:, 0] *= 2  # determinant 2: simplicial and solid, but not smooth
    assert abs(int(fmpz_mat(rays.tolist()).det())) == 2

    assert Cone(rays=rays.astype(np.int64)).is_smooth() is False


def test_the_documented_small_examples_are_unchanged():
    """The docstring's own examples, which the float path also got right."""
    assert Cone([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).is_smooth() is True
    assert Cone([[2, 0, 1], [0, 1, 0], [1, 0, 2]]).is_smooth() is False


def test_a_non_simplicial_cone_is_rejected_before_any_determinant():
    """Four rays in three dimensions have no square ray matrix at all."""
    cone = Cone([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, -1]])
    assert not cone.is_simplicial()
    assert cone.is_smooth() is False
