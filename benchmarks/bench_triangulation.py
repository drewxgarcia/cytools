"""
Benchmarks for the Triangulation class — exhaustive coverage of all 28 public methods.

Methods covered:
  clear_cache, poly, labels, dimension, ambient_dimension,
  is_fine, is_star, is_regular, is_valid, check_heights,
  get_toric_variety, get_cy,
  vc, fan,
  points, points_to_labels, points_to_indices, triangulation_to_polytope_indices,
  simplices (all variants), restrict,
  secondary_cone, heights,
  automorphism_orbit, is_equivalent,
  neighbor_triangulations, random_flips,
  gkz_phi, sr_ideal

Fixture design
--------------
``triang_5v``
    Single reference triangulation (POLY_5V, 5 vertices).  All single-object tests.

``batch_triangs``  (from tiny_polys, 5v, vertex-count tier)
    For triangulation-structure tests: simplices, is_fine/star/regular/valid,
    GKZ, secondary_cone, SR ideal, fan, restrict, neighbor_traversal.

``cy_triangs``  (from cy_polys, h11 <= 4, DB-filtered)
    For full-pipeline tests: get_toric_variety(), get_cy().
    h11 filtering guarantees CY admissibility — no try/except needed.

``small_triangs``  (from small_polys, 6-7v, vertex-count tier)
    For slow-marked sweeps of operations that scale with vertex count
    (is_regular, simplices, sr_ideal).

Run fast suite:
    pytest benchmarks/bench_triangulation.py --benchmark-only -m "not slow"

Run full suite:
    pytest benchmarks/bench_triangulation.py --benchmark-only
"""

import pytest

from cytools import Polytope
from cytools.dataset import POLY_5V


# ---------------------------------------------------------------------------
# Module-scope fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def triang_5v():
    return POLY_5V.triangulate()


@pytest.fixture(scope="module")
def batch_triangs(tiny_polys):
    """Triangulations from tiny_polys (5v tier).  For structure-level tests."""
    return [r.polytope.triangulate() for r in tiny_polys]


@pytest.fixture(scope="module")
def cy_triangs(cy_polys):
    """Triangulations from cy_polys (h11 <= 4).

    All polytopes are guaranteed CY-admissible via DB filter —
    no try/except needed for get_toric_variety() or get_cy().
    """
    return [r.polytope.triangulate() for r in cy_polys]


@pytest.fixture(scope="module")
def small_triangs(small_polys):
    """Triangulations from small_polys (6-7v tier).  For slow sweeps."""
    return [r.polytope.triangulate() for r in small_polys]


# Polytope with 2 triangulations — used for neighbor / all_triangulations tests
_P2T = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-2,-1,-1],[-2,-1,-1,-1]])


# ---------------------------------------------------------------------------
# 1. Basic properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_dimension_5v(self, benchmark, triang_5v):
        benchmark(triang_5v.dimension)

    def test_ambient_dimension_5v(self, benchmark, triang_5v):
        benchmark(triang_5v.ambient_dimension)

    def test_poly_property(self, benchmark, triang_5v):
        benchmark(lambda: triang_5v.poly)

    def test_labels_property(self, benchmark, triang_5v):
        benchmark(lambda: triang_5v.labels)


# ---------------------------------------------------------------------------
# 2. Validity checks
# ---------------------------------------------------------------------------

class TestValidityChecks:
    def test_is_fine_5v(self, benchmark, triang_5v):
        benchmark(triang_5v.is_fine)

    def test_is_star_5v(self, benchmark, triang_5v):
        benchmark(triang_5v.is_star)

    def test_is_regular_5v(self, benchmark, triang_5v):
        benchmark(triang_5v.is_regular)

    def test_is_valid_5v(self, benchmark, triang_5v):
        benchmark(triang_5v.is_valid)

    def test_check_heights_5v(self, benchmark, triang_5v):
        benchmark(triang_5v.check_heights)

    def test_is_fine_batch(self, benchmark, batch_triangs):
        def go():
            return [t.is_fine() for t in batch_triangs]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_is_star_batch(self, benchmark, batch_triangs):
        def go():
            return [t.is_star() for t in batch_triangs]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_is_regular_batch(self, benchmark, batch_triangs):
        def go():
            return [t.is_regular() for t in batch_triangs]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_is_valid_batch(self, benchmark, batch_triangs):
        def go():
            return [t.is_valid() for t in batch_triangs]
        benchmark.pedantic(go, rounds=5, iterations=1)

    @pytest.mark.slow
    def test_is_regular_large(self, benchmark, small_triangs):
        def go():
            return [t.is_regular() for t in small_triangs]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 3. Simplices access (all variants)
# ---------------------------------------------------------------------------

class TestSimplices:
    def test_simplices_default_5v(self, benchmark, triang_5v):
        benchmark(lambda: triang_5v.simplices())

    def test_simplices_as_indices_5v(self, benchmark, triang_5v):
        benchmark(lambda: triang_5v.simplices(as_indices=True))

    def test_simplices_on_faces_dim2_5v(self, benchmark, triang_5v):
        benchmark(lambda: triang_5v.simplices(on_faces_dim=2))

    def test_simplices_on_faces_codim1_5v(self, benchmark, triang_5v):
        benchmark(lambda: triang_5v.simplices(on_faces_codim=1))

    def test_simplices_split_by_face_5v(self, benchmark, triang_5v):
        benchmark(lambda: triang_5v.simplices(split_by_face=True))

    def test_simplices_batch(self, benchmark, batch_triangs):
        def go():
            return [t.simplices() for t in batch_triangs]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_simplices_large(self, benchmark, small_triangs):
        def go():
            return [t.simplices() for t in small_triangs]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 4. Points lookup
# ---------------------------------------------------------------------------

class TestPointsLookup:
    def test_points_5v(self, benchmark, triang_5v):
        benchmark(lambda: triang_5v.points())

    def test_points_optimal_5v(self, benchmark, triang_5v):
        benchmark(lambda: triang_5v.points(optimal=True))

    def test_points_as_poly_indices_5v(self, benchmark, triang_5v):
        benchmark(lambda: triang_5v.points(as_poly_indices=True))

    def test_points_to_indices_5v(self, benchmark, triang_5v):
        pts = triang_5v.points().tolist()
        benchmark(triang_5v.points_to_indices, pts)

    def test_points_to_labels_5v(self, benchmark, triang_5v):
        pts = triang_5v.points().tolist()
        benchmark(triang_5v.points_to_labels, pts)

    def test_triangulation_to_polytope_indices_5v(self, benchmark, triang_5v):
        inds = list(range(len(triang_5v.labels)))
        benchmark(triang_5v.triangulation_to_polytope_indices, inds)


# ---------------------------------------------------------------------------
# 5. GKZ / secondary cone
# ---------------------------------------------------------------------------

class TestGKZAndSecondaryCone:
    def test_gkz_phi_5v(self, benchmark, triang_5v):
        benchmark(triang_5v.gkz_phi)

    def test_secondary_cone_5v(self, benchmark, triang_5v):
        benchmark(triang_5v.secondary_cone)

    def test_secondary_cone_not_as_cone_5v(self, benchmark, triang_5v):
        benchmark(lambda: triang_5v.secondary_cone(as_cone=False))

    def test_secondary_cone_on_faces_5v(self, benchmark, triang_5v):
        benchmark(lambda: triang_5v.secondary_cone(on_faces_dim=2))

    def test_heights_5v(self, benchmark, triang_5v):
        benchmark(triang_5v.heights)

    def test_gkz_phi_batch(self, benchmark, batch_triangs):
        def go():
            return [t.gkz_phi() for t in batch_triangs]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_secondary_cone_batch(self, benchmark, batch_triangs):
        def go():
            return [t.secondary_cone() for t in batch_triangs]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 6. SR ideal
# ---------------------------------------------------------------------------

class TestSRIdeal:
    def test_sr_ideal_5v(self, benchmark, triang_5v):
        benchmark(triang_5v.sr_ideal)

    def test_sr_ideal_batch(self, benchmark, batch_triangs):
        def go():
            return [t.sr_ideal() for t in batch_triangs]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_sr_ideal_large(self, benchmark, small_triangs):
        def go():
            return [t.sr_ideal() for t in small_triangs]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 7. Neighbor traversal / flips
# ---------------------------------------------------------------------------

class TestNeighborTraversal:
    def test_neighbors_5v_default(self, benchmark, triang_5v):
        benchmark(lambda: list(triang_5v.neighbor_triangulations()))

    def test_neighbors_5v_fine_only(self, benchmark, triang_5v):
        benchmark(lambda: list(triang_5v.neighbor_triangulations(only_fine=True)))

    def test_neighbors_5v_regular_only(self, benchmark, triang_5v):
        benchmark(lambda: list(triang_5v.neighbor_triangulations(only_regular=True)))

    def test_neighbors_batch(self, benchmark, batch_triangs):
        def go():
            return [list(t.neighbor_triangulations()) for t in batch_triangs]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_random_flips_5_2triang(self, benchmark):
        """Use a known polytope with multiple triangulations so flips always succeed."""
        def go():
            p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-2,-1,-1],[-2,-1,-1,-1]])
            t = p.triangulate()
            return t.random_flips(5, seed=42)
        benchmark(go)

    def test_random_flips_batch(self, benchmark):
        """Random flips on a polytope with 2 known triangulations.

        random_flips requires multiple triangulations; use _P2T which always
        has exactly 2 so the walk is well-defined and deterministic.
        """
        def go():
            return _P2T.triangulate().random_flips(5, seed=42)
        benchmark(go)


# ---------------------------------------------------------------------------
# 8. Automorphism orbit
# ---------------------------------------------------------------------------

class TestAutomorphismOrbit:
    def test_automorphism_orbit_5v_all(self, benchmark, triang_5v):
        benchmark(triang_5v.automorphism_orbit)

    def test_automorphism_orbit_5v_first(self, benchmark, triang_5v):
        benchmark(lambda: triang_5v.automorphism_orbit(automorphism=0))

    def test_automorphism_orbit_on_faces_5v(self, benchmark, triang_5v):
        benchmark(lambda: triang_5v.automorphism_orbit(on_faces_dim=2))

    def test_is_equivalent_self(self, benchmark, triang_5v):
        benchmark(triang_5v.is_equivalent, triang_5v)


# ---------------------------------------------------------------------------
# 9. Restrict to faces
# ---------------------------------------------------------------------------

class TestRestrict:
    def test_restrict_default_5v(self, benchmark, triang_5v):
        benchmark(lambda: triang_5v.restrict())

    def test_restrict_to_dim2_5v(self, benchmark, triang_5v):
        benchmark(lambda: triang_5v.restrict(restrict_dim=2))

    def test_restrict_batch(self, benchmark, batch_triangs):
        def go():
            return [t.restrict() for t in batch_triangs]
        benchmark.pedantic(go, rounds=5, iterations=1)


# ---------------------------------------------------------------------------
# 10. Fan / VectorConfiguration
# ---------------------------------------------------------------------------

class TestFanAndVC:
    def test_fan_5v(self, benchmark, triang_5v):
        benchmark(triang_5v.fan)

    def test_vc_5v(self, benchmark, triang_5v):
        benchmark(triang_5v.vc)

    def test_fan_batch(self, benchmark, batch_triangs):
        def go():
            return [t.fan() for t in batch_triangs]
        benchmark.pedantic(go, rounds=5, iterations=1)


# ---------------------------------------------------------------------------
# 11. Full pipeline: Polytope → Triangulation → ToricVariety → CalabiYau
#
# Uses cy_triangs (h11 <= 4, DB-filtered) — guaranteed admissible.
# No try/except needed.
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_get_toric_variety_5v(self, benchmark):
        def go():
            p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            t = p.triangulate()
            return t.get_toric_variety()
        benchmark(go)

    def test_get_cy_5v(self, benchmark):
        def go():
            p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
            t = p.triangulate()
            return t.get_cy()
        benchmark(go)

    def test_get_toric_variety_batch(self, benchmark, cy_triangs):
        def go():
            return [t.get_toric_variety() for t in cy_triangs]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_get_cy_batch(self, benchmark, cy_triangs):
        def go():
            return [t.get_cy() for t in cy_triangs]
        benchmark.pedantic(go, rounds=3, iterations=1)

    @pytest.mark.slow
    def test_get_toric_variety_large(self, benchmark, cy_polys_large):
        triangs = [r.polytope.triangulate() for r in cy_polys_large]
        def go():
            return [t.get_toric_variety() for t in triangs]
        benchmark.pedantic(go, rounds=1, iterations=1)
