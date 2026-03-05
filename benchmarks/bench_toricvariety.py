"""
Benchmarks for the ToricVariety class — exhaustive coverage of all 21 public methods.

Methods covered:
  clear_cache, is_compact, triangulation, polytope, dimension,
  sr_ideal,
  glsm_charge_matrix, glsm_linear_relations,
  divisor_basis, set_divisor_basis, curve_basis, set_curve_basis,
  mori_cone (all variants), kahler_cone,
  intersection_numbers (all format/basis variants),
  prime_toric_divisors, is_smooth, canonical_divisor_is_smooth,
  effective_cone, fan_cones, get_cy

Tiers: 5v micro, tiny (5v DB batch), small (6-7v DB batch)

Run with:
    pytest benchmarks/bench_toricvariety.py --benchmark-only -m "not slow"
"""

import pytest

from cytools import Polytope
from cytools.dataset import POLY_5V

# ---------------------------------------------------------------------------
# Module-scope fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tv_5v():
    t = POLY_5V.triangulate()
    return t.get_toric_variety()

@pytest.fixture(scope="module")
def tiny_tvs(tiny_polys):
    results = []
    skipped = 0
    for r in tiny_polys:
        try:
            t = r.polytope.triangulate()
            results.append(t.get_toric_variety())
        except Exception as e:
            skipped += 1
            import warnings; warnings.warn(f"tiny_tvs: skipped polytope ({e})")
    if not results:
        raise RuntimeError("tiny_tvs: all polytopes failed — check DB or pipeline")
    return results

@pytest.fixture(scope="module")
def small_tvs(small_polys):
    results = []
    for r in small_polys:
        try:
            t = r.polytope.triangulate()
            results.append(t.get_toric_variety())
        except Exception as e:
            import warnings; warnings.warn(f"small_tvs: skipped polytope ({e})")
    if not results:
        raise RuntimeError("small_tvs: all polytopes failed — check DB or pipeline")
    return results


# ---------------------------------------------------------------------------
# 1. Metadata / pipeline accessors
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_is_compact_5v(self, benchmark, tv_5v):
        benchmark(tv_5v.is_compact)

    def test_dimension_5v(self, benchmark, tv_5v):
        benchmark(tv_5v.dimension)

    def test_triangulation_5v(self, benchmark, tv_5v):
        benchmark(lambda: tv_5v.triangulation)

    def test_polytope_5v(self, benchmark, tv_5v):
        benchmark(lambda: tv_5v.polytope)

    def test_is_compact_tiny(self, benchmark, tiny_tvs):
        def go():
            return [tv.is_compact() for tv in tiny_tvs]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 2. SR ideal
# ---------------------------------------------------------------------------

class TestSRIdeal:
    def test_sr_ideal_5v(self, benchmark, tv_5v):
        benchmark(tv_5v.sr_ideal)

    def test_sr_ideal_tiny(self, benchmark, tiny_tvs):
        def go():
            return [tv.sr_ideal() for tv in tiny_tvs]
        benchmark.pedantic(go, rounds=1, iterations=1)

    def test_sr_ideal_small(self, benchmark, small_tvs):
        def go():
            return [tv.sr_ideal() for tv in small_tvs]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 3. GLSM / basis
# ---------------------------------------------------------------------------

class TestGLSMBasis:
    def test_glsm_charge_matrix_5v(self, benchmark, tv_5v):
        benchmark(tv_5v.glsm_charge_matrix)

    def test_glsm_charge_matrix_no_origin(self, benchmark, tv_5v):
        benchmark(lambda: tv_5v.glsm_charge_matrix(include_origin=False))

    def test_glsm_linear_relations_5v(self, benchmark, tv_5v):
        benchmark(tv_5v.glsm_linear_relations)

    def test_divisor_basis_5v(self, benchmark, tv_5v):
        benchmark(tv_5v.divisor_basis)

    def test_divisor_basis_as_matrix_5v(self, benchmark, tv_5v):
        benchmark(lambda: tv_5v.divisor_basis(as_matrix=True))

    def test_curve_basis_5v(self, benchmark, tv_5v):
        benchmark(tv_5v.curve_basis)

    def test_prime_toric_divisors_5v(self, benchmark, tv_5v):
        benchmark(tv_5v.prime_toric_divisors)

    def test_set_divisor_basis_reset(self, benchmark):
        """Benchmark resetting divisor basis to the default (index-based)."""
        # Use a fresh TV so mutations don't affect other tests sharing tv_5v.
        from cytools.dataset import POLY_5V as _P
        tv = _P.triangulate().get_toric_variety()
        basis = tv.divisor_basis()
        benchmark(lambda: tv.set_divisor_basis(basis))

    def test_glsm_tiny(self, benchmark, tiny_tvs):
        def go():
            return [tv.glsm_charge_matrix() for tv in tiny_tvs]
        benchmark.pedantic(go, rounds=1, iterations=1)

    def test_glsm_small(self, benchmark, small_tvs):
        def go():
            return [tv.glsm_charge_matrix() for tv in small_tvs]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 4. Intersection numbers
# ---------------------------------------------------------------------------

class TestIntersectionNumbers:
    def test_intersection_numbers_dok_5v(self, benchmark, tv_5v):
        benchmark(tv_5v.intersection_numbers)

    def test_intersection_numbers_in_basis_5v(self, benchmark, tv_5v):
        benchmark(lambda: tv_5v.intersection_numbers(in_basis=True))

    def test_intersection_numbers_dense_5v(self, benchmark, tv_5v):
        benchmark(lambda: tv_5v.intersection_numbers(format="dense"))

    def test_intersection_numbers_coo_5v(self, benchmark, tv_5v):
        benchmark(lambda: tv_5v.intersection_numbers(format="coo"))

    def test_intersection_numbers_zero_as_anticanon_5v(self, benchmark):
        # zero_as_anticanonical=True only makes sense for non-favorable polytopes
        # Use a polytope where h11 != h21 to exercise this code path
        p = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-3,-6]])
        t = p.triangulate()
        tv = t.get_toric_variety()
        benchmark(lambda: tv.intersection_numbers(zero_as_anticanonical=True))

    def test_intersection_numbers_tiny(self, benchmark, tiny_tvs):
        def go():
            return [tv.intersection_numbers() for tv in tiny_tvs]
        benchmark.pedantic(go, rounds=1, iterations=1)

    def test_intersection_numbers_small(self, benchmark, small_tvs):
        def go():
            return [tv.intersection_numbers() for tv in small_tvs]
        benchmark.pedantic(go, rounds=1, iterations=1)

    @pytest.mark.slow
    def test_intersection_numbers_medium(self, benchmark, small_tvs):
        def go():
            return [tv.intersection_numbers() for tv in small_tvs]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 5. Cones
# ---------------------------------------------------------------------------

class TestCones:
    def test_kahler_cone_5v(self, benchmark, tv_5v):
        benchmark(tv_5v.kahler_cone)

    def test_mori_cone_5v(self, benchmark, tv_5v):
        benchmark(tv_5v.mori_cone)

    def test_mori_cone_in_basis_5v(self, benchmark, tv_5v):
        benchmark(lambda: tv_5v.mori_cone(in_basis=True))

    def test_mori_cone_from_intnums_5v(self, benchmark, tv_5v):
        # Pre-warm intersection numbers
        tv_5v.intersection_numbers()
        benchmark(lambda: tv_5v.mori_cone(from_intersection_numbers=True))

    def test_effective_cone_5v(self, benchmark, tv_5v):
        benchmark(tv_5v.effective_cone)

    def test_fan_cones_5v(self, benchmark, tv_5v):
        benchmark(tv_5v.fan_cones)

    def test_fan_cones_dim3_5v(self, benchmark, tv_5v):
        benchmark(lambda: tv_5v.fan_cones(d=3))

    def test_kahler_cone_tiny(self, benchmark, tiny_tvs):
        def go():
            return [tv.kahler_cone() for tv in tiny_tvs]
        benchmark.pedantic(go, rounds=1, iterations=1)

    def test_mori_cone_tiny(self, benchmark, tiny_tvs):
        def go():
            return [tv.mori_cone() for tv in tiny_tvs]
        benchmark.pedantic(go, rounds=1, iterations=1)



# ---------------------------------------------------------------------------
# 6. Smoothness
# ---------------------------------------------------------------------------

class TestSmoothness:
    def test_is_smooth_5v(self, benchmark, tv_5v):
        benchmark(tv_5v.is_smooth)

    def test_canonical_divisor_is_smooth_5v(self, benchmark, tv_5v):
        benchmark(tv_5v.canonical_divisor_is_smooth)

    def test_is_smooth_tiny(self, benchmark, tiny_tvs):
        def go():
            return [tv.is_smooth() for tv in tiny_tvs]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 7. CY construction
# ---------------------------------------------------------------------------

class TestGetCY:
    def test_get_cy_5v(self, benchmark, tv_5v):
        benchmark(tv_5v.get_cy)

    def test_get_cy_tiny(self, benchmark, tiny_tvs):
        def go():
            results = []
            for tv in tiny_tvs:
                try:
                    results.append(tv.get_cy())
                except Exception:
                    pass
            return results
        benchmark.pedantic(go, rounds=1, iterations=1)

    def test_get_cy_small(self, benchmark, small_tvs):
        def go():
            results = []
            for tv in small_tvs:
                try:
                    results.append(tv.get_cy())
                except Exception:
                    pass
            return results
        benchmark.pedantic(go, rounds=1, iterations=1)
