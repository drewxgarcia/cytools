"""
Benchmarks for the Cone class.

Methods covered:
  ambient_dimension, dimension,
  rays, hyperplanes,
  contains,
  dual_cone,
  extremal_rays, extremal_hyperplanes,
  facets,
  tip_of_stretched_cone,
  find_grading_vector, find_interior_point,
  find_lattice_points,
  is_solid, is_pointed, is_simplicial, is_degenerate, is_smooth,
  hilbert_basis,
  intersection

Fixture design
--------------
Kähler cone ambient dimension equals the polytope's h12.  Two populations:

``cone_5v``
    Single reference cone (POLY_5V, h12=1, cone dim=1).  All operations.

``any_dim_cones``  (from tiny_polys, unbounded dim)
    For operations that never call rays()/hyperplanes() as a cold path:
    is_solid, is_pointed, contains, find_interior_point, find_grading_vector,
    dual_cone, construction.

``cone_polys``  (h12 <= 4, from DB, cone dim <= 4)
    For dualize()-dependent operations: rays(), hyperplanes(), extremal_rays(),
    extremal_hyperplanes(), facets(), intersection(), hilbert_basis(),
    is_simplicial(), is_smooth().

``cone_polys_large``  (h12 <= 8, slow sweep)
    Wider population for slow-marked KS sweeps of dualize()-dependent ops.

Run fast suite:
    pytest benchmarks/bench_cone.py --benchmark-only -m "not slow"

Run full suite:
    pytest benchmarks/bench_cone.py --benchmark-only
"""

import pytest

from cytools.dataset import POLY_5V


# ---------------------------------------------------------------------------
# Module-scope fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cone_5v():
    """Kähler cone from POLY_5V (h12=1, ambient dim=1).  All operations."""
    return POLY_5V.triangulate().get_toric_variety().kahler_cone()


@pytest.fixture(scope="module")
def any_dim_cones(tiny_polys):
    """Kähler cones from tiny_polys.  Unbounded ambient dim.

    Only use for operations that never call rays()/hyperplanes() cold.
    """
    cones = []
    for r in tiny_polys:
        try:
            cones.append(r.polytope.triangulate().get_toric_variety().kahler_cone())
        except Exception:
            pass
    return cones


@pytest.fixture(scope="module")
def dualize_safe_cones(cone_polys):
    """Kähler cones from cone_polys (h12<=4, cone dim<=4).

    Safe for all dualize()-dependent operations.
    """
    cones = []
    for r in cone_polys:
        try:
            cones.append(r.polytope.triangulate().get_toric_variety().kahler_cone())
        except Exception:
            pass
    return cones


@pytest.fixture(scope="module")
def dualize_safe_cones_large(cone_polys_large):
    """Kähler cones from cone_polys_large (h12<=8, cone dim<=8).

    For slow KS sweeps of dualize()-dependent operations.
    """
    cones = []
    for r in cone_polys_large:
        try:
            cones.append(r.polytope.triangulate().get_toric_variety().kahler_cone())
        except Exception:
            pass
    return cones


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_construct_5v(self, benchmark):
        tv = POLY_5V.triangulate().get_toric_variety()
        benchmark(tv.kahler_cone)

    def test_construct_batch(self, benchmark, tiny_polys):
        def go():
            return [r.polytope.triangulate().get_toric_variety().kahler_cone()
                    for r in tiny_polys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    @pytest.mark.slow
    def test_construct_large(self, benchmark, small_polys):
        def go():
            return [r.polytope.triangulate().get_toric_variety().kahler_cone()
                    for r in small_polys]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 2. Basic properties  (no dualize() in hot path — use any_dim_cones)
# ---------------------------------------------------------------------------

class TestBasicProperties:
    def test_ambient_dimension_5v(self, benchmark, cone_5v):
        benchmark(cone_5v.ambient_dimension)

    def test_dimension_5v(self, benchmark, cone_5v):
        benchmark(cone_5v.dimension)

    def test_is_solid_5v(self, benchmark, cone_5v):
        benchmark(cone_5v.is_solid)

    def test_is_pointed_5v(self, benchmark, cone_5v):
        benchmark(cone_5v.is_pointed)

    def test_is_degenerate_5v(self, benchmark, cone_5v):
        benchmark(cone_5v.is_degenerate)

    def test_is_solid_batch(self, benchmark, any_dim_cones):
        def go(): return [c.is_solid() for c in any_dim_cones]
        benchmark.pedantic(go, rounds=5, iterations=1)

    @pytest.mark.slow
    def test_is_solid_large(self, benchmark, dualize_safe_cones_large):
        def go(): return [c.is_solid() for c in dualize_safe_cones_large]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 3. is_simplicial / is_smooth
# (call extremal_hyperplanes() → dual.extremal_rays() — LP cost per ray)
# ---------------------------------------------------------------------------

class TestSimplicialSmooth:
    def test_is_simplicial_5v(self, benchmark, cone_5v):
        benchmark(cone_5v.is_simplicial)

    def test_is_smooth_5v(self, benchmark, cone_5v):
        benchmark(cone_5v.is_smooth)

    def test_is_simplicial_batch(self, benchmark, dualize_safe_cones):
        def go(): return [c.is_simplicial() for c in dualize_safe_cones]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_is_smooth_batch(self, benchmark, dualize_safe_cones):
        def go(): return [c.is_smooth() for c in dualize_safe_cones]
        benchmark.pedantic(go, rounds=5, iterations=1)

    @pytest.mark.slow
    def test_is_simplicial_large(self, benchmark, dualize_safe_cones_large):
        def go(): return [c.is_simplicial() for c in dualize_safe_cones_large]
        benchmark.pedantic(go, rounds=3, iterations=1)

    @pytest.mark.slow
    def test_is_smooth_large(self, benchmark, dualize_safe_cones_large):
        def go(): return [c.is_smooth() for c in dualize_safe_cones_large]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 4. Rays / hyperplanes  (dualize()-dependent — use dualize_safe_cones)
# ---------------------------------------------------------------------------

class TestRaysHyperplanes:
    def test_rays_5v(self, benchmark, cone_5v):
        benchmark(cone_5v.rays)

    def test_hyperplanes_5v(self, benchmark, cone_5v):
        benchmark(cone_5v.hyperplanes)

    def test_extremal_rays_5v(self, benchmark, cone_5v):
        benchmark(cone_5v.extremal_rays)

    def test_extremal_hyperplanes_5v(self, benchmark, cone_5v):
        benchmark(cone_5v.extremal_hyperplanes)

    def test_facets_5v(self, benchmark, cone_5v):
        benchmark(cone_5v.facets)

    def test_rays_batch(self, benchmark, dualize_safe_cones):
        def go(): return [c.rays() for c in dualize_safe_cones]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_extremal_rays_batch(self, benchmark, dualize_safe_cones):
        def go(): return [c.extremal_rays() for c in dualize_safe_cones]
        benchmark.pedantic(go, rounds=5, iterations=1)

    @pytest.mark.slow
    def test_rays_large(self, benchmark, dualize_safe_cones_large):
        def go(): return [c.rays() for c in dualize_safe_cones_large]
        benchmark.pedantic(go, rounds=3, iterations=1)

    @pytest.mark.slow
    def test_extremal_rays_large(self, benchmark, dualize_safe_cones_large):
        def go(): return [c.extremal_rays() for c in dualize_safe_cones_large]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 5. Dual cone  (just swaps _rays/_hyperplanes — safe at any dim)
# ---------------------------------------------------------------------------

class TestDualCone:
    def test_dual_cone_5v(self, benchmark, cone_5v):
        benchmark(cone_5v.dual_cone)

    def test_dual_roundtrip_5v(self, benchmark, cone_5v):
        d = cone_5v.dual_cone()
        benchmark(d.dual_cone)

    def test_dual_cone_batch(self, benchmark, any_dim_cones):
        def go(): return [c.dual_cone() for c in any_dim_cones]
        benchmark.pedantic(go, rounds=5, iterations=1)

    @pytest.mark.slow
    def test_dual_cone_large(self, benchmark, dualize_safe_cones_large):
        def go(): return [c.dual_cone() for c in dualize_safe_cones_large]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 6. Contains  (uses cached hyperplanes — safe at any dim)
# ---------------------------------------------------------------------------

class TestContains:
    def test_contains_5v(self, benchmark, cone_5v):
        pt = [1.0] * cone_5v.ambient_dimension()
        benchmark(cone_5v.contains, pt)

    def test_contains_batch(self, benchmark, any_dim_cones):
        def go():
            return [c.contains([1.0] * c.ambient_dimension()) for c in any_dim_cones]
        benchmark.pedantic(go, rounds=5, iterations=1)

    @pytest.mark.slow
    def test_contains_large(self, benchmark, dualize_safe_cones_large):
        def go():
            return [c.contains([1.0] * c.ambient_dimension())
                    for c in dualize_safe_cones_large]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 7. Interior point / grading vector  (LP on hyperplanes — safe at any dim)
# ---------------------------------------------------------------------------

class TestInteriorPoint:
    def test_find_interior_point_5v(self, benchmark, cone_5v):
        benchmark(cone_5v.find_interior_point)

    def test_find_grading_vector_5v(self, benchmark, cone_5v):
        benchmark(cone_5v.find_grading_vector)

    def test_find_interior_point_batch(self, benchmark, any_dim_cones):
        def go(): return [c.find_interior_point() for c in any_dim_cones]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_find_grading_vector_batch(self, benchmark, any_dim_cones):
        def go(): return [c.find_grading_vector() for c in any_dim_cones]
        benchmark.pedantic(go, rounds=3, iterations=1)

    @pytest.mark.slow
    def test_find_interior_point_large(self, benchmark, dualize_safe_cones_large):
        def go(): return [c.find_interior_point() for c in dualize_safe_cones_large]
        benchmark.pedantic(go, rounds=3, iterations=1)

    @pytest.mark.slow
    def test_find_grading_vector_large(self, benchmark, dualize_safe_cones_large):
        def go(): return [c.find_grading_vector() for c in dualize_safe_cones_large]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 8. Lattice points  (uses cached hyperplanes — safe at any dim)
# ---------------------------------------------------------------------------

class TestLatticePoints:
    @pytest.mark.parametrize("max_deg", [3, 5, 8])
    def test_find_lattice_points_5v(self, benchmark, cone_5v, max_deg):
        gv = cone_5v.find_grading_vector()
        benchmark(lambda: cone_5v.find_lattice_points(max_deg=max_deg, grading_vector=gv))

    @pytest.mark.slow
    def test_find_lattice_points_batch(self, benchmark, any_dim_cones):
        def go():
            return [c.find_lattice_points(max_deg=3, grading_vector=c.find_grading_vector())
                    for c in any_dim_cones]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 9. Hilbert basis  (calls rays() — dualize()-dependent)
# ---------------------------------------------------------------------------

class TestHilbertBasis:
    def test_hilbert_basis_5v(self, benchmark, cone_5v):
        benchmark(cone_5v.hilbert_basis)

    @pytest.mark.slow
    def test_hilbert_basis_batch(self, benchmark, dualize_safe_cones):
        def go(): return [c.hilbert_basis() for c in dualize_safe_cones]
        benchmark.pedantic(go, rounds=1, iterations=1)

    @pytest.mark.slow
    def test_hilbert_basis_large(self, benchmark, dualize_safe_cones_large):
        def go(): return [c.hilbert_basis() for c in dualize_safe_cones_large]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 10. Intersection  (calls hyperplanes() on both operands — dualize()-dependent)
# ---------------------------------------------------------------------------

class TestIntersection:
    def test_intersection_5v_with_dual(self, benchmark, cone_5v):
        dual = cone_5v.dual_cone()
        benchmark(cone_5v.intersection, dual)

    def test_intersection_batch(self, benchmark, dualize_safe_cones):
        def go(): return [c.intersection(c.dual_cone()) for c in dualize_safe_cones]
        benchmark.pedantic(go, rounds=5, iterations=1)

    @pytest.mark.slow
    def test_intersection_large(self, benchmark, dualize_safe_cones_large):
        def go(): return [c.intersection(c.dual_cone()) for c in dualize_safe_cones_large]
        benchmark.pedantic(go, rounds=3, iterations=1)
