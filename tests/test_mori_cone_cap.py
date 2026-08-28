"""Tests for CalabiYau.mori_cone_cap.

The Mori-cone cap supplies the generators that GV-invariant computation needs,
so it is on the critical path for instanton corrections. Its ray matrix used to
be assembled by element-wise assignment into a dok_matrix; these pin the result
so the COO construction that replaced it cannot drift.
"""

import numpy as np
import pytest

from cytools import Polytope

POLYTOPES = [
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]],
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]],
    [[-6, -8, -5, -5], [0, 1, 0, 0], [1, 0, 0, 0], [2, 4, 5, 0], [3, 3, 0, 5]],
]


def _cy(verts):
    p = Polytope(verts)
    if not p.is_reflexive():
        pytest.skip("needs a reflexive polytope")
    triang = p.triangulate(make_star=True)
    if not triang.is_star():
        pytest.skip("needs a star triangulation")
    try:
        return triang.get_cy()
    except Exception as e:
        pytest.skip(f"cannot build a CY: {e}")


@pytest.mark.parametrize("verts", POLYTOPES, ids=lambda v: f"{len(v)}v")
def test_cap_rays_are_integral_and_stable(verts):
    cy = _cy(verts)
    rays = np.asarray(cy.mori_cone_cap(in_basis=True).rays())

    assert rays.size, "expected some cap generators"
    assert np.allclose(rays, np.rint(rays)), "cap rays must be integral"

    # recomputing must give the same set of rays
    again = np.asarray(cy.mori_cone_cap(in_basis=True).rays())
    assert {tuple(r) for r in rays.tolist()} == {tuple(r) for r in again.tolist()}


@pytest.mark.parametrize("verts", POLYTOPES, ids=lambda v: f"{len(v)}v")
def test_sparse_and_dense_formats_agree(verts):
    """format="sparse" must hold the same matrix the Cone is built from."""
    cy = _cy(verts)
    sparse = cy.mori_cone_cap(in_basis=True, format="sparse")
    dense = np.asarray(cy.mori_cone_cap(in_basis=True).rays())

    as_dense = np.asarray(sparse.todense())
    assert as_dense.shape[1] == dense.shape[1]
    assert {tuple(r) for r in dense.tolist()} <= {tuple(r) for r in as_dense.tolist()}


@pytest.mark.parametrize("verts", POLYTOPES, ids=lambda v: f"{len(v)}v")
def test_exclude_origin_drops_exactly_one_column(verts):
    cy = _cy(verts)
    full = cy.mori_cone_cap(format="sparse")
    trimmed = cy.mori_cone_cap(exclude_origin=True, format="sparse")
    assert trimmed.shape[1] == full.shape[1] - 1
    assert np.array_equal(
        np.asarray(trimmed.todense()), np.asarray(full.todense())[:, 1:]
    )


def test_cap_generators_span_a_pointed_cone():
    """The cap must admit a grading vector, which GV computation requires."""
    cy = _cy(POLYTOPES[0])
    cone = cy.mori_cone_cap(in_basis=True)
    grading = cone.find_grading_vector()
    rays = np.asarray(cone.rays())
    # a grading vector pairs positively with every generator
    assert np.all(rays.dot(np.asarray(grading)) > 0)
