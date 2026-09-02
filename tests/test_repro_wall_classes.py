"""Small algebraic checks for the Wall-class reproduction experiment."""

import numpy as np
from benchmarks.repro.wall_classes import (
    equivalent,
    gl_matrices,
    invariants,
    transform,
)
from benchmarks.repro.wall_refine import deep_equivalent


def test_gl_search_contains_exactly_the_unimodular_matrices():
    matrices = gl_matrices(2, 1)
    determinants = np.rint(np.linalg.det(np.asarray(matrices))).astype(int)

    assert len(matrices) == 40
    assert set(np.abs(determinants)) == {1}


def test_wall_data_transform_and_equivalence_agree():
    kappa = np.array([[[2]]], dtype=np.int64)
    c2 = np.array([4], dtype=np.int64)
    reflection = np.array([[-1]], dtype=np.int64)
    transformed_kappa, transformed_c2 = transform(kappa, c2, reflection)

    original = (1, 101, kappa, c2)
    transformed = (1, 101, transformed_kappa, transformed_c2)
    assert invariants(*original) == invariants(*transformed)
    assert equivalent(original, transformed, gl_matrices(1, 1))
    assert deep_equivalent(original, transformed, bound=1)
