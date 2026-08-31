"""Differential tests for Triangulation.sr_ideal.

`sr_ideal` computes the generators of the Stanley-Reisner ideal, i.e. the
minimal non-faces of the triangulation. The implementation was rewritten from a
frozenset-based incremental search to packed integer bitmasks plus a
neighbourhood-restricted candidate enumeration, so these tests check it against
a direct, deliberately naive reference rather than against pinned values.
"""

import itertools

import numpy as np
import pytest

from cytools import Polytope

# A spread of reflexive 4D polytopes: the two standard small ones, plus larger
# vertex counts where the two algorithms diverge most.
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
    [
        [-6, -8, -5, -5],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [2, 4, 5, 0],
        [3, 3, 0, 5],
    ],
]


def sr_ideal_reference(triang):
    """Minimal non-faces, computed directly from the definition.

    A set is a face when it is contained in some simplex. A minimal non-face is
    a non-face all of whose proper subsets are faces. Enumerated exhaustively
    over subset sizes 2..dim, matching the range the implementation covers.
    """
    labels = [ll for ll in triang.labels if ll != triang.poly._label_origin]
    dim = triang.dim()

    simplices = [set(s) for s in triang.simplices()]

    def is_face(subset):
        return any(subset <= s for s in simplices)

    generators = []
    for size in range(2, dim + 1):
        for combo in itertools.combinations(labels, size):
            subset = set(combo)
            if is_face(subset):
                continue
            # minimal: every proper subset obtained by dropping one point must
            # be a face (which inductively makes all proper subsets faces)
            if all(is_face(subset - {x}) for x in subset):
                generators.append(tuple(sorted(combo)))

    return tuple(sorted(generators, key=lambda x: (len(x), x)))


def _usable(t):
    """sr_ideal's stated precondition."""
    return t.is_star() and t._is_fulldim


@pytest.mark.parametrize("verts", POLYTOPES, ids=lambda v: f"{len(v)}v")
def test_sr_ideal_matches_definition(verts):
    t = Polytope(verts).triangulate(make_star=True)
    if not _usable(t):
        pytest.skip("sr_ideal only applies to full-dimensional star triangulations")
    if not t.is_fine():
        pytest.skip("see test_non_fine_triangulations_include_unused_points")
    assert tuple(t.sr_ideal()) == sr_ideal_reference(t)


@pytest.mark.parametrize("verts", POLYTOPES, ids=lambda v: f"{len(v)}v")
def test_sr_ideal_matches_definition_for_random_triangulations(verts):
    """Vary the heights so the simplicial complex differs between runs."""
    p = Polytope(verts)
    base = p.triangulate(make_star=True)
    base_heights = np.asarray(base.heights(), dtype=float)

    rng = np.random.default_rng(0)
    n_checked = 0

    # perturb around the default heights: staying near a fine chamber keeps the
    # sampled triangulations fine, which uniform random heights do not for every
    # polytope here
    for scale in (0.0, 0.05, 0.2, 0.5, 1.0, 2.0):
        heights = base_heights + scale * rng.normal(size=base_heights.shape)
        try:
            t = p.triangulate(heights=heights, make_star=True)
        except Exception:
            continue
        if not _usable(t) or not t.is_fine():
            continue
        assert tuple(t.sr_ideal()) == sr_ideal_reference(t)
        n_checked += 1

    assert n_checked, "no fine, full-dimensional star triangulation was produced"


def test_non_fine_triangulations_include_unused_points():
    """Pins a long-standing quirk, so that changing it has to be deliberate.

    When a point of the triangulation belongs to no simplex, `sr_ideal` emits
    the pairs joining it to every point that *is* used, even though those pairs
    are not minimal non-faces: the unused point is a non-face by itself, so
    strictly the size-1 set generates them. Pairs of two unused points are not
    emitted.

    This only arises for non-fine triangulations. Every fine triangulation uses
    all of its points, so the output agrees with the textbook definition there,
    which is what `test_sr_ideal_matches_definition` checks.
    """
    p = Polytope(POLYTOPES[0])
    rng = np.random.default_rng(0)

    for _ in range(12):
        heights = rng.integers(1, 40, size=len(p.points())).astype(float)
        try:
            t = p.triangulate(heights=heights, make_star=True)
        except Exception:
            continue
        if not _usable(t) or t.is_fine():
            continue

        used = {ll for s in t.simplices() for ll in s}
        unused = [
            ll for ll in t.labels if ll != t.poly._label_origin and ll not in used
        ]
        if not unused:
            continue

        generators = {frozenset(g) for g in t.sr_ideal()}
        textbook = {frozenset(g) for g in sr_ideal_reference(t)}

        # the extra generators are exactly used-point/unused-point pairs
        extra = generators - textbook
        assert extra, "expected the quirk to show on a non-fine triangulation"
        for g in extra:
            assert len(g) == 2
            assert len(g & set(unused)) == 1, f"{sorted(g)} is not a used/unused pair"

        # ...and no pair of two unused points is emitted
        for a, b in itertools.combinations(unused, 2):
            assert frozenset((a, b)) not in generators

        return

    pytest.skip("no non-fine star triangulation with unused points was produced")


def test_sr_ideal_generators_are_minimal_and_are_non_faces():
    """Property check: every generator is a non-face, and is minimally so."""
    p = Polytope(POLYTOPES[0])
    t = p.triangulate()
    simplices = [set(s) for s in t.simplices()]

    def is_face(subset):
        return any(subset <= s for s in simplices)

    generators = [set(g) for g in t.sr_ideal()]
    assert generators, "expected a non-empty SR ideal"

    for g in generators:
        assert not is_face(g), f"{sorted(g)} is a face, so not a non-face"
        for x in g:
            assert is_face(g - {x}), f"{sorted(g)} is not a minimal non-face"

    # no generator is a superset of another (that would be a multiple)
    for a, b in itertools.permutations(generators, 2):
        assert not (a < b), f"{sorted(b)} is a multiple of {sorted(a)}"


def test_sr_ideal_is_cached():
    t = Polytope(POLYTOPES[0]).triangulate()
    assert t.sr_ideal() is t.sr_ideal()


def test_sr_ideal_rejects_non_star_triangulations():
    p = Polytope(POLYTOPES[0])
    t = p.triangulate(include_points_interior_to_facets=True, make_star=False)
    if t.is_star():
        pytest.skip("triangulation came out star anyway")
    with pytest.raises(NotImplementedError):
        t.sr_ideal()
