"""Tests for the toric-variety intersection-number linear system.

`_construct_intnum_equations_4d` builds the sparse system M x + C = 0 whose
solution supplies the intersection numbers that are not fixed directly by the
simplices. It was rewritten from dict-of-lists accumulation plus a
triple-nested Python loop to vectorised COO assembly, so these tests check the
*defining property* of that system rather than pinning its internals: the
resulting intersection numbers must annihilate the GLSM linear relations.

That property is what the assembly encodes, so a mis-set row, column or
coefficient breaks it, while any correct assembly satisfies it.
"""

import itertools

import numpy as np
import pytest

from cytools import Polytope

POLYTOPES = [
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]],
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]],
    [
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        [-1, -1, -3, -6], [-1, -1, -1, -1],
    ],
    [[-6, -8, -5, -5], [0, 1, 0, 0], [1, 0, 0, 0], [2, 4, 5, 0], [3, 3, 0, 5]],
]


def _toric_variety(verts):
    p = Polytope(verts)
    if not p.is_reflexive():
        pytest.skip("toric varieties from non-reflexive polytopes need exp features")
    triang = p.triangulate(make_star=True)
    if not triang.is_star():
        pytest.skip("toric varieties require a star triangulation")
    return triang.get_toric_variety()


@pytest.mark.parametrize("verts", POLYTOPES, ids=lambda v: f"{len(v)}v")
def test_intersection_numbers_annihilate_the_linear_relations(verts):
    r"""For every linear relation l and every divisor triple (j, k, m):

        sum_i l_i * kappa_{i j k m} = 0

    This is the system that `_construct_intnum_equations_4d` assembles, so it
    holds only if rows, columns and coefficients all line up.
    """
    tv = _toric_variety(verts)
    intnums = tv.intersection_numbers(in_basis=False)
    relations = np.asarray(tv.glsm_linear_relations(include_origin=False), dtype=float)

    # symmetric lookup: keys are sorted index tuples
    def kappa(indices):
        return intnums.get(tuple(sorted(indices)), 0.0)

    # the divisors actually appearing, excluding the canonical one
    divisors = sorted({i for key in intnums for i in key if i != 0})
    assert divisors, "expected some prime toric divisors"

    n_rel = len(relations)
    worst = 0.0
    checked = 0
    for triple in itertools.combinations_with_replacement(divisors, 3):
        for r in range(n_rel):
            total = 0.0
            for i in divisors:
                coeff = relations[r][i - 1]
                if coeff:
                    total += coeff * kappa((i,) + triple)
            worst = max(worst, abs(total))
            checked += 1

    assert checked, "no equations were checked"
    assert worst < 1e-6, f"linear relations violated by {worst:.3e}"


@pytest.mark.parametrize("verts", POLYTOPES, ids=lambda v: f"{len(v)}v")
def test_in_basis_is_the_filtered_full_tensor(verts):
    """The basis-restricted tensor must equal filtering the full one.

    Cross-checks the two consumers of the assembled solution against each
    other, which catches an assembly that is self-consistent but indexed wrong.
    """
    from cytools.utils import filter_tensor_indices

    tv = _toric_variety(verts)
    full = tv.intersection_numbers(in_basis=False)
    in_basis = tv.intersection_numbers(in_basis=True)

    expected = filter_tensor_indices(full, tv.divisor_basis())
    assert set(in_basis) == set(expected)
    for key in in_basis:
        assert in_basis[key] == pytest.approx(expected[key], abs=1e-8)


@pytest.mark.parametrize("verts", POLYTOPES, ids=lambda v: f"{len(v)}v")
def test_intersection_numbers_are_integral_when_smooth(verts):
    """A smooth toric variety has integral intersection numbers."""
    tv = _toric_variety(verts)
    if not tv.is_smooth():
        pytest.skip("toric variety is not smooth")

    for key, val in tv.intersection_numbers(in_basis=False).items():
        assert abs(round(float(val)) - float(val)) < 1e-6, f"{key} -> {val}"


def test_projective_space_and_quintic_pinned_values():
    """Pinned values, so an assembly that preserves the linear relations but
    scales the solution still fails.

    The polytope with vertices e_i and -(1,1,1,1) gives P^4, whose five prime
    divisors are all linearly equivalent with D^4 = 1 (the class of a point).
    Its anticanonical hypersurface is the quintic threefold, with the single
    independent intersection number kappa = 5. Note the distinction: the 5 is a
    property of the Calabi-Yau, not of the ambient toric variety.
    """
    tv = _toric_variety(POLYTOPES[1])

    in_basis = tv.intersection_numbers(in_basis=True)
    assert len(in_basis) == 1
    ((key, value),) = in_basis.items()
    assert key == (0, 0, 0, 0)
    assert float(value) == pytest.approx(1.0), "P^4 should have D^4 = 1"

    cy = tv.get_cy()
    cy_in_basis = cy.intersection_numbers(in_basis=True)
    assert len(cy_in_basis) == 1
    ((cy_key, cy_value),) = cy_in_basis.items()
    assert float(cy_value) == pytest.approx(5.0), "the quintic should have kappa = 5"


def test_distinct_intersection_numbers_match_simplex_volumes():
    """The directly-determined entries are 1/|det| of their simplex.

    Guards the batched-determinant path and the row ordering feeding it.
    """
    verts = POLYTOPES[0]
    p = Polytope(verts)
    triang = p.triangulate()
    tv = triang.get_toric_variety()

    points = triang.points()
    pts_ext = np.empty((points.shape[0], points.shape[1] + 1), dtype=int)
    pts_ext[:, :-1] = points
    pts_ext[:, -1] = 1

    intnums = tv.intersection_numbers(in_basis=False)

    for simplex in triang.simplices(as_indices=True):
        labels = tuple(sorted(int(c) for c in simplex if c != 0))
        if len(labels) != 4:
            continue
        expected = 1.0 / abs(np.linalg.det(pts_ext[np.asarray(simplex)]))
        assert float(intnums[labels]) == pytest.approx(expected, rel=1e-9)
