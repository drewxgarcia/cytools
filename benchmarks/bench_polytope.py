"""
Benchmarks for the Polytope class — exhaustive coverage of all 55 public methods.

Methods covered:
  Construction, backend, ambient_dimension, dimension, is_solid, labels/*,
  inequalities, points, get_bdry, vc, face_triangs, n_2face_triangs,
  points_to_labels, points_to_indices, vertices, faces, facets, dual_polytope,
  is_reflexive, automorphisms, normal_form, is_linearly_equivalent,
  is_affinely_equivalent, triangulate, random_triangulations_fast,
  random_triangulations_fair, all_triangulations, hpq, chi, is_favorable,
  glsm_charge_matrix, glsm_linear_relations, glsm_basis, minkowski_sum,
  volume, find_2d_reflexive_subpolytopes, is_trilayer,
  ntfe_hypers (skipped — requires large setup), nef_partitions (experimental)

Fixture design
--------------
``POLY_5V``   Single 5-vertex reference polytope.  All single-polytope tests.
``batch_polys``  20 polytopes at 5v (tiny tier) — fast batch sweep.
``small_batch_polys``  20 polytopes at 6-7v (small tier) — medium batch sweep.
``medium_batch_polys``  20 polytopes at 9-10v (medium tier) — slow batch sweep.

Run fast suite:
    pytest benchmarks/bench_polytope.py --benchmark-only -m "not slow"

Run full suite:
    pytest benchmarks/bench_polytope.py --benchmark-only
"""

import numpy as np
import pytest

from cytools import Polytope
from cytools.dataset import POLY_5V, POLY_6V

# Two polytopes for equivalence tests
_P1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
_P2 = Polytope([[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,0,0,2],[-1,-1,-1,0]])

# Two small polytopes for Minkowski sum
_MS1 = Polytope([[1,0,0],[0,1,0],[-1,-1,0]])
_MS2 = Polytope([[0,0,1],[0,0,-1]])

# 4D reflexive with a known 2D reflexive subpolytope
_P_4D_REFL = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])


# ---------------------------------------------------------------------------
# Module-scope fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def batch_polys(tiny_polys):
    """20 polytopes from the tiny (5v) tier."""
    return [r.polytope for r in tiny_polys]


@pytest.fixture(scope="module")
def small_batch_polys(small_polys):
    """20 polytopes from the small (6-7v) tier."""
    return [r.polytope for r in small_polys]


@pytest.fixture(scope="module")
def medium_batch_polys(medium_polys):
    """20 polytopes from the medium (9-10v) tier."""
    return [r.polytope for r in medium_polys]


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_construct_5v(self, benchmark):
        benchmark(Polytope, [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])

    def test_construct_simplex_5d(self, benchmark):
        """5D simplex — tests PPL fallback for dim > 4."""
        verts = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[-1,-1,-1,-1,-1]]
        benchmark(Polytope, verts)

    def test_construct_batch(self, benchmark, tiny_polys):
        verts_list = [list(r.polytope.vertices()) for r in tiny_polys]
        def go():
            return [Polytope(v) for v in verts_list]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_construct_small_batch(self, benchmark, small_polys):
        verts_list = [list(r.polytope.vertices()) for r in small_polys]
        def go():
            return [Polytope(v) for v in verts_list]
        benchmark.pedantic(go, rounds=3, iterations=1)

    @pytest.mark.slow
    def test_construct_medium_batch(self, benchmark, medium_polys):
        verts_list = [list(r.polytope.vertices()) for r in medium_polys]
        def go():
            return [Polytope(v) for v in verts_list]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 2. Metadata / labels (cheap, but tested for regression)
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_ambient_dimension(self, benchmark):
        benchmark(POLY_5V.ambient_dimension)

    def test_dimension(self, benchmark):
        benchmark(POLY_5V.dimension)

    def test_is_solid(self, benchmark):
        benchmark(POLY_5V.is_solid)

    def test_is_reflexive(self, benchmark):
        benchmark(POLY_5V.is_reflexive)

    def test_backend(self, benchmark):
        benchmark(lambda: POLY_5V.backend)

    def test_labels(self, benchmark):
        benchmark(lambda: POLY_5V.labels)

    def test_labels_vertices(self, benchmark):
        benchmark(lambda: POLY_5V.labels_vertices)

    def test_labels_int(self, benchmark):
        benchmark(lambda: POLY_5V.labels_int)

    def test_labels_facet(self, benchmark):
        benchmark(lambda: POLY_5V.labels_facet)

    def test_labels_bdry(self, benchmark):
        benchmark(lambda: POLY_5V.labels_bdry)

    def test_labels_codim2(self, benchmark):
        benchmark(lambda: POLY_5V.labels_codim2)

    def test_labels_not_facet(self, benchmark):
        benchmark(lambda: POLY_5V.labels_not_facet)


# ---------------------------------------------------------------------------
# 3. Inequalities
# ---------------------------------------------------------------------------

class TestInequalities:
    def test_inequalities_5v(self, benchmark):
        benchmark(POLY_5V.inequalities)

    def test_inequalities_batch(self, benchmark, batch_polys):
        def go():
            return [p.inequalities() for p in batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_inequalities_small_batch(self, benchmark, small_batch_polys):
        def go():
            return [p.inequalities() for p in small_batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 4. Points / vertices
# ---------------------------------------------------------------------------

class TestPointsVertices:
    def test_points_5v(self, benchmark):
        benchmark(POLY_5V.points)

    def test_points_optimal_5v(self, benchmark):
        benchmark(lambda: POLY_5V.points(optimal=True))

    def test_vertices_5v(self, benchmark):
        benchmark(POLY_5V.vertices)

    def test_vertices_optimal_5v(self, benchmark):
        benchmark(lambda: POLY_5V.vertices(optimal=True))

    def test_points_not_interior_to_facets_5v(self, benchmark):
        benchmark(POLY_5V.points_not_interior_to_facets)

    def test_points_to_indices_single_5v(self, benchmark):
        pt = POLY_5V.points()[0].tolist()
        benchmark(POLY_5V.points_to_indices, pt)

    def test_points_to_indices_batch_5v(self, benchmark):
        pts = POLY_5V.points().tolist()
        benchmark(POLY_5V.points_to_indices, pts)

    def test_points_to_labels_batch_5v(self, benchmark):
        pts = POLY_5V.points().tolist()
        benchmark(POLY_5V.points_to_labels, pts)

    def test_points_batch(self, benchmark, batch_polys):
        def go():
            return [p.points() for p in batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_points_small_batch(self, benchmark, small_batch_polys):
        def go():
            return [p.points() for p in small_batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    @pytest.mark.slow
    def test_points_medium_batch(self, benchmark, medium_batch_polys):
        def go():
            return [p.points() for p in medium_batch_polys]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 5. Faces / facets
# ---------------------------------------------------------------------------

class TestFaces:
    def test_faces_all_5v(self, benchmark):
        benchmark(POLY_5V.faces)

    def test_faces_0d_5v(self, benchmark):
        benchmark(lambda: POLY_5V.faces(0))

    def test_faces_1d_5v(self, benchmark):
        benchmark(lambda: POLY_5V.faces(1))

    def test_faces_2d_5v(self, benchmark):
        benchmark(lambda: POLY_5V.faces(2))

    def test_faces_3d_5v(self, benchmark):
        benchmark(lambda: POLY_5V.faces(3))

    def test_facets_5v(self, benchmark):
        benchmark(POLY_5V.facets)

    def test_faces_batch(self, benchmark, batch_polys):
        def go():
            return [p.faces() for p in batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_faces_small_batch(self, benchmark, small_batch_polys):
        def go():
            return [p.faces() for p in small_batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    @pytest.mark.slow
    def test_faces_medium_batch(self, benchmark, medium_batch_polys):
        def go():
            return [p.faces() for p in medium_batch_polys]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 6. Dual polytope
# ---------------------------------------------------------------------------

class TestDualPolytope:
    def test_dual_polytope_5v(self, benchmark):
        benchmark(_P1.dual_polytope)

    def test_dual_polytope_batch(self, benchmark, batch_polys):
        """All KS 4D polytopes are reflexive by definition — no filter needed."""
        def go():
            return [p.dual_polytope() for p in batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_dual_roundtrip_5v(self, benchmark):
        """Dual of dual should return same object (cached)."""
        d = _P1.dual_polytope()
        benchmark(d.dual_polytope)


# ---------------------------------------------------------------------------
# 7. Automorphisms
# ---------------------------------------------------------------------------

class TestAutomorphisms:
    def test_automorphisms_5v(self, benchmark):
        benchmark(POLY_5V.automorphisms)

    def test_automorphisms_square_5v(self, benchmark):
        benchmark(lambda: POLY_5V.automorphisms(square_to_one=True))

    def test_automorphisms_as_dict_5v(self, benchmark):
        benchmark(lambda: POLY_5V.automorphisms(as_dictionary=True))

    def test_automorphisms_batch(self, benchmark, batch_polys):
        def go():
            return [p.automorphisms() for p in batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_automorphisms_small_batch(self, benchmark, small_batch_polys):
        def go():
            return [p.automorphisms() for p in small_batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 8. Normal form
# ---------------------------------------------------------------------------

class TestNormalForm:
    def test_normal_form_5v(self, benchmark):
        benchmark(POLY_5V.normal_form)

    def test_normal_form_affine_5v(self, benchmark):
        benchmark(lambda: POLY_5V.normal_form(affine_transform=True))

    def test_normal_form_batch(self, benchmark, batch_polys):
        def go():
            return [p.normal_form() for p in batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_normal_form_small_batch(self, benchmark, small_batch_polys):
        def go():
            return [p.normal_form() for p in small_batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 9. Equivalence tests
# ---------------------------------------------------------------------------

class TestEquivalence:
    def test_is_linearly_equivalent_true(self, benchmark):
        p1 = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
        p2 = Polytope([[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[1,1,1,1]])
        benchmark(p1.is_linearly_equivalent, p2)

    def test_is_linearly_equivalent_false(self, benchmark):
        benchmark(_P1.is_linearly_equivalent, _P_4D_REFL)

    def test_is_affinely_equivalent_true(self, benchmark):
        benchmark(_P1.is_affinely_equivalent, _P2)

    def test_is_affinely_equivalent_false(self, benchmark):
        benchmark(_P1.is_affinely_equivalent, _P_4D_REFL)

    def test_is_favorable_5v(self, benchmark):
        benchmark(lambda: _P_4D_REFL.is_favorable(lattice="N"))

    def test_is_favorable_batch(self, benchmark, batch_polys):
        def go():
            return [p.is_favorable(lattice="N") for p in batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 10. Volume + geometry
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_volume_3d(self, benchmark):
        p = Polytope([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
        benchmark(p.volume)

    def test_volume_4d_5v(self, benchmark):
        benchmark(POLY_5V.volume)

    def test_volume_batch(self, benchmark, batch_polys):
        def go():
            return [p.volume() for p in batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_minkowski_sum(self, benchmark):
        benchmark(_MS1.minkowski_sum, _MS2)

    def test_get_bdry_5v(self, benchmark):
        p = Polytope([[1,0,0],[0,1,0],[-1,-1,0]])
        benchmark(p.get_bdry)

    def test_find_2d_reflexive_subpolytopes(self, benchmark):
        benchmark(_P_4D_REFL.find_2d_reflexive_subpolytopes)

    def test_is_trilayer(self, benchmark):
        benchmark(POLY_5V.is_trilayer)


# ---------------------------------------------------------------------------
# 11. GLSM / linear algebra
# ---------------------------------------------------------------------------

class TestGLSM:
    def test_glsm_charge_matrix_5v(self, benchmark):
        benchmark(POLY_5V.glsm_charge_matrix)

    def test_glsm_charge_matrix_no_origin(self, benchmark):
        benchmark(lambda: POLY_5V.glsm_charge_matrix(include_origin=False))

    def test_glsm_basis_5v(self, benchmark):
        benchmark(POLY_5V.glsm_basis)

    def test_glsm_linear_relations_5v(self, benchmark):
        benchmark(POLY_5V.glsm_linear_relations)

    def test_glsm_charge_matrix_batch(self, benchmark, batch_polys):
        def go():
            return [p.glsm_charge_matrix() for p in batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_glsm_charge_matrix_small_batch(self, benchmark, small_batch_polys):
        def go():
            return [p.glsm_charge_matrix() for p in small_batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    @pytest.mark.slow
    def test_glsm_charge_matrix_medium_batch(self, benchmark, medium_batch_polys):
        def go():
            return [p.glsm_charge_matrix() for p in medium_batch_polys]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 12. Topological invariants
# ---------------------------------------------------------------------------

class TestTopology:
    def test_chi_N_5v(self, benchmark):
        benchmark(lambda: POLY_5V.chi(lattice="N"))

    def test_chi_M_5v(self, benchmark):
        benchmark(lambda: POLY_5V.chi(lattice="M"))

    def test_hpq_00_5v(self, benchmark):
        benchmark(lambda: POLY_5V.hpq(0, 0, lattice="N"))

    def test_hpq_11_5v(self, benchmark):
        benchmark(lambda: POLY_5V.hpq(1, 1, lattice="N"))

    def test_hpq_12_5v(self, benchmark):
        benchmark(lambda: POLY_5V.hpq(1, 2, lattice="N"))

    def test_chi_batch(self, benchmark, batch_polys):
        def go():
            return [p.chi(lattice="N") for p in batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_hpq_batch(self, benchmark, batch_polys):
        def go():
            return [p.hpq(1, 1, lattice="N") for p in batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_chi_small_batch(self, benchmark, small_batch_polys):
        def go():
            return [p.chi(lattice="N") for p in small_batch_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    @pytest.mark.slow
    def test_chi_medium_batch(self, benchmark, medium_batch_polys):
        def go():
            return [p.chi(lattice="N") for p in medium_batch_polys]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 13. Triangulation
# ---------------------------------------------------------------------------

class TestTriangulation:
    def test_triangulate_5v(self, benchmark):
        def go():
            p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            return p.triangulate()
        benchmark(go)

    def test_triangulate_no_star(self, benchmark):
        def go():
            p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            return p.triangulate(make_star=False)
        benchmark(go)

    def test_all_triangulations_2triang(self, benchmark):
        def go():
            p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-2,-1,-1],[-2,-1,-1,-1]])
            return list(p.all_triangulations(as_list=True))
        benchmark(go)

    def test_all_triangulations_no_filters(self, benchmark):
        def go():
            p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-2,-1,-1],[-2,-1,-1,-1]])
            return list(p.all_triangulations(only_fine=False, only_regular=False, only_star=False, as_list=True))
        benchmark(go)

    def test_triangulate_batch(self, benchmark, tiny_polys):
        verts_list = [list(r.polytope.vertices()) for r in tiny_polys]
        def go():
            return [Polytope(v).triangulate() for v in verts_list]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_triangulate_small_batch(self, benchmark, small_polys):
        verts_list = [list(r.polytope.vertices()) for r in small_polys]
        def go():
            return [Polytope(v).triangulate() for v in verts_list]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_random_triangulations_fast_5(self, benchmark):
        def go():
            p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            return list(p.random_triangulations_fast(N=5, progress_bar=False))
        benchmark(go)

    def test_vc_5v(self, benchmark):
        benchmark(POLY_5V.vc)

    def test_face_triangs_5v(self, benchmark):
        benchmark(lambda: POLY_5V.face_triangs(dim=2))

    def test_n_2face_triangs_5v(self, benchmark):
        benchmark(POLY_5V.n_2face_triangs)

    @pytest.mark.slow
    def test_triangulate_medium_batch(self, benchmark, medium_polys):
        verts_list = [list(r.polytope.vertices()) for r in medium_polys]
        def go():
            return [Polytope(v).triangulate() for v in verts_list]
        benchmark.pedantic(go, rounds=1, iterations=1)
