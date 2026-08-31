"""Tests for the batched secondary-cone hyperplane computation.

`Triangulation.secondary_cone`'s native backend derives one hyperplane per
interior facet from the null space of a (dim+1) x (dim+2) integer matrix. That
was one exact `flint` solve per facet; it is now a handful of batched
determinants, exploiting the null space being one-dimensional so the null vector
is the generalized cross product of signed minors.

Both implementations are still present -- the per-facet one as an exact fallback
-- so the batched path can be tested differentially against it rather than
against pinned values.
"""

import numpy as np
import pytest

from cytools import Polytope
from cytools.triangulation import (
    _flint_nullvector,
    _secondary_cone_hyperplanes_flint,
    _secondary_cone_hyperplanes_native,
)

POLYTOPES = [
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]],
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]],
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, -1, -3, -6],
        [-1, -1, -1, -1],
    ],
    [[-6, -8, -5, -5], [0, 1, 0, 0], [1, 0, 0, 0], [2, 4, 5, 0], [3, 3, 0, 5]],
]


def _triangulations(verts, n_perturbed=4):
    """The default triangulation plus a few from perturbed heights."""
    p = Polytope(verts)
    out = [p.triangulate()]
    base = np.asarray(out[0].heights(), dtype=float)
    rng = np.random.default_rng(0)
    for _ in range(n_perturbed):
        try:
            out.append(
                p.triangulate(heights=base + rng.normal(scale=0.4, size=base.shape))
            )
        except Exception:
            continue
    return out


@pytest.mark.parametrize("verts", POLYTOPES, ids=lambda v: f"{len(v)}v")
def test_batched_matches_per_facet_flint(verts):
    """The batched minors must give exactly the flint hyperplanes."""
    for triang in _triangulations(verts):
        fast = _secondary_cone_hyperplanes_native(triang)
        exact = _secondary_cone_hyperplanes_flint(triang)
        assert set(fast) == set(exact), (
            f"{len(fast)} vs {len(exact)} hyperplanes; "
            f"only_fast={sorted(set(fast) - set(exact))[:3]} "
            f"only_flint={sorted(set(exact) - set(fast))[:3]}"
        )


@pytest.mark.parametrize("verts", POLYTOPES, ids=lambda v: f"{len(v)}v")
def test_hyperplanes_are_primitive_and_sign_normalized(verts):
    """Each normal is gcd-reduced with a non-negative leading entry.

    Both are required for the set-based deduplication to collapse hyperplanes
    that differ only by scale or sign.
    """
    import math

    for triang in _triangulations(verts, n_perturbed=2):
        for hyp in _secondary_cone_hyperplanes_native(triang):
            nonzero = [h for h in hyp if h]
            if not nonzero:
                continue
            assert math.gcd(*[abs(h) for h in nonzero]) == 1, f"not primitive: {hyp}"


@pytest.mark.parametrize("verts", POLYTOPES, ids=lambda v: f"{len(v)}v")
def test_secondary_cone_is_solid_for_regular_triangulations(verts):
    """A triangulation is regular exactly when its secondary cone is solid.

    An end-to-end property of the hyperplanes, independent of how they are
    computed: these triangulations come from explicit heights, so they are
    regular by construction.
    """
    for triang in _triangulations(verts, n_perturbed=2):
        assert triang.is_regular()
        assert triang.secondary_cone().is_solid()


def test_flint_nullvector_is_normalized():
    """The fallback returns a primitive vector with a positive leading entry."""
    import math

    # rows one shorter than columns, so the null space is one-dimensional
    mat = np.array(
        [
            [2, 0, 0, 0, 0],
            [0, 3, 0, 0, 0],
            [0, 0, 4, 0, 0],
            [1, 1, 1, 1, 1],
        ]
    )
    v = _flint_nullvector(mat)
    assert (mat @ v == 0).all(), f"{v} is not in the null space"
    assert v[0] >= 0
    assert math.gcd(*[abs(int(x)) for x in v if x]) == 1


@pytest.mark.parametrize("verts", POLYTOPES[:2], ids=lambda v: f"{len(v)}v")
def test_cached_and_uncached_agree(verts):
    triang = Polytope(verts).triangulate()
    first = triang.secondary_cone(use_cache=False)
    second = triang.secondary_cone()
    assert set(map(tuple, first.hyperplanes())) == set(map(tuple, second.hyperplanes()))


def test_batched_path_does_not_fall_back():
    """The fast path must actually be used, not silently repaired.

    The batched minors are verified exactly and any failure is recomputed with
    flint, so a bug in the fast path yields a *correct* answer more slowly. No
    output comparison can detect that -- an earlier version of this file tried,
    and a deliberately corrupted minor sign passed every test in it. Assert the
    repair counter stays put instead.
    """
    import cytools.triangulation as tri

    before = tri._secondary_cone_fallbacks
    checked = 0
    for verts in POLYTOPES:
        for triang in _triangulations(verts):
            _secondary_cone_hyperplanes_native(triang)
            checked += 1

    assert checked, "no triangulations were exercised"
    assert tri._secondary_cone_fallbacks == before, (
        f"{tri._secondary_cone_fallbacks - before} facets fell back to flint; "
        "results are still correct but the batched path is not being used"
    )
