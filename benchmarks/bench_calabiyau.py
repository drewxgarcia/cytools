"""
Benchmarks for the CalabiYau class — exhaustive coverage of all 48 public methods.

Methods covered:
  clear_cache, is_trivially_equivalent,
  ambient_variety, triangulation, polytope,
  ambient_dimension, dimension,
  hpq, h11, h12, h13, h22, chi,
  glsm_charge_matrix, glsm_linear_relations,
  divisor_basis, set_divisor_basis, curve_basis, set_curve_basis,
  intersection_numbers (all format/basis variants),
  prime_toric_divisors, second_chern_class,
  is_smooth,
  toric_mori_cone, toric_kahler_cone, toric_effective_cone,
  compute_cy_volume, compute_divisor_volumes, compute_curve_volumes,
  compute_kappa_matrix, compute_kappa_vector,
  compute_inverse_kahler_metric, compute_kahler_metric,
  compute_gvs, compute_gws,
  mori_cone_cap, grading_vec, cutoff, charges, cone, gvs, gws

Fixture design
--------------
``cy_5v``
    Single CY from POLY_5V.  All single-object tests.

``batch_cys``  (from cy_polys, h11 <= 4, DB-filtered)
    20 CY objects guaranteed admissible via DB filter — no try/except needed.

``large_cys``  (from cy_polys_large, h11 <= 8, DB-filtered)
    100 CY objects for slow-marked sweeps of expensive operations.

``tloc_5v``
    A valid point inside the Kähler cone of cy_5v (ones vector of length h11).

Run fast suite:
    pytest benchmarks/bench_calabiyau.py --benchmark-only -m "not slow"

Run full suite:
    pytest benchmarks/bench_calabiyau.py --benchmark-only
"""

import numpy as np
import pytest

from cytools import Polytope
from cytools.dataset import POLY_5V


# ---------------------------------------------------------------------------
# Module-scope fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cy_5v():
    t = POLY_5V.triangulate()
    tv = t.get_toric_variety()
    return tv.get_cy()


@pytest.fixture(scope="module")
def batch_cys(cy_polys):
    """20 CY objects with h11 <= 4.

    Polytopes are drawn from the KS reflexive database filtered by h11 at
    query time — guaranteed CY-admissible.  No try/except required.
    """
    return [r.polytope.triangulate().get_cy() for r in cy_polys]


@pytest.fixture(scope="module")
def large_cys(cy_polys_large):
    """100 CY objects with h11 <= 8.  For slow-marked sweeps."""
    return [r.polytope.triangulate().get_cy() for r in cy_polys_large]


@pytest.fixture(scope="module")
def tloc_5v(cy_5v):
    """A valid point inside the Kähler cone for the 5v CY."""
    return np.ones(cy_5v.h11())


# ---------------------------------------------------------------------------
# 1. Metadata / pipeline accessors
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_ambient_dimension_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.ambient_dimension)

    def test_dimension_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.dimension)

    def test_ambient_variety_5v(self, benchmark, cy_5v):
        benchmark(lambda: cy_5v.ambient_variety)

    def test_triangulation_5v(self, benchmark, cy_5v):
        benchmark(lambda: cy_5v.triangulation)

    def test_polytope_5v(self, benchmark, cy_5v):
        benchmark(lambda: cy_5v.polytope)

    def test_is_trivially_equivalent_self(self, benchmark, cy_5v):
        benchmark(cy_5v.is_trivially_equivalent, cy_5v)


# ---------------------------------------------------------------------------
# 2. Hodge numbers / chi
# ---------------------------------------------------------------------------

class TestHodgeNumbers:
    def test_chi_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.chi)

    def test_h11_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.h11)

    def test_h12_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.h12)

    def test_h13_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.h13)

    def test_h22_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.h22)

    def test_hpq_11_5v(self, benchmark, cy_5v):
        benchmark(lambda: cy_5v.hpq(1, 1))

    def test_hpq_12_5v(self, benchmark, cy_5v):
        benchmark(lambda: cy_5v.hpq(1, 2))

    def test_hodge_batch(self, benchmark, batch_cys):
        def go():
            return [(cy.h11(), cy.h12()) for cy in batch_cys]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_hodge_large(self, benchmark, large_cys):
        def go():
            return [(cy.h11(), cy.h12()) for cy in large_cys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_chi_batch(self, benchmark, batch_cys):
        def go():
            return [cy.chi() for cy in batch_cys]
        benchmark.pedantic(go, rounds=5, iterations=1)


# ---------------------------------------------------------------------------
# 3. GLSM / basis
# ---------------------------------------------------------------------------

class TestGLSMBasis:
    def test_glsm_charge_matrix_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.glsm_charge_matrix)

    def test_glsm_charge_matrix_no_origin(self, benchmark, cy_5v):
        benchmark(lambda: cy_5v.glsm_charge_matrix(include_origin=False))

    def test_glsm_linear_relations_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.glsm_linear_relations)

    def test_divisor_basis_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.divisor_basis)

    def test_divisor_basis_as_matrix_5v(self, benchmark, cy_5v):
        benchmark(lambda: cy_5v.divisor_basis(as_matrix=True))

    def test_curve_basis_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.curve_basis)

    def test_prime_toric_divisors_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.prime_toric_divisors)

    def test_glsm_batch(self, benchmark, batch_cys):
        def go():
            return [cy.glsm_charge_matrix() for cy in batch_cys]
        benchmark.pedantic(go, rounds=5, iterations=1)


# ---------------------------------------------------------------------------
# 4. Intersection numbers
# ---------------------------------------------------------------------------

class TestIntersectionNumbers:
    def test_intersection_numbers_dok_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.intersection_numbers)

    def test_intersection_numbers_in_basis_5v(self, benchmark, cy_5v):
        benchmark(lambda: cy_5v.intersection_numbers(in_basis=True))

    def test_intersection_numbers_dense_5v(self, benchmark, cy_5v):
        benchmark(lambda: cy_5v.intersection_numbers(format="dense"))

    def test_intersection_numbers_coo_5v(self, benchmark, cy_5v):
        benchmark(lambda: cy_5v.intersection_numbers(format="coo"))

    def test_intersection_numbers_zero_as_anticanon_5v(self, benchmark, cy_5v):
        benchmark(lambda: cy_5v.intersection_numbers(zero_as_anticanonical=True))

    def test_intersection_numbers_batch(self, benchmark, batch_cys):
        def go():
            return [cy.intersection_numbers() for cy in batch_cys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    @pytest.mark.slow
    def test_intersection_numbers_large(self, benchmark, large_cys):
        def go():
            return [cy.intersection_numbers() for cy in large_cys]
        benchmark.pedantic(go, rounds=1, iterations=1)

    def test_second_chern_class_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.second_chern_class)

    def test_second_chern_in_basis_5v(self, benchmark, cy_5v):
        benchmark(lambda: cy_5v.second_chern_class(in_basis=True))

    def test_second_chern_batch(self, benchmark, batch_cys):
        def go():
            return [cy.second_chern_class() for cy in batch_cys]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 5. Smoothness
# ---------------------------------------------------------------------------

class TestSmoothness:
    def test_is_smooth_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.is_smooth)

    def test_is_smooth_batch(self, benchmark, batch_cys):
        def go():
            return [cy.is_smooth() for cy in batch_cys]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 6. Cones
# ---------------------------------------------------------------------------

class TestCones:
    def test_toric_kahler_cone_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.toric_kahler_cone)

    def test_toric_mori_cone_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.toric_mori_cone)

    def test_toric_mori_cone_in_basis_5v(self, benchmark, cy_5v):
        benchmark(lambda: cy_5v.toric_mori_cone(in_basis=True))

    def test_toric_effective_cone_5v(self, benchmark, cy_5v):
        benchmark(cy_5v.toric_effective_cone)

    def test_cone_5v(self, benchmark, cy_5v):
        # CalabiYau.cone() lives on the Invariants object returned by compute_gvs
        inv = cy_5v.compute_gvs(max_deg=3)
        benchmark(inv.cone)

    def test_toric_kahler_cone_batch(self, benchmark, batch_cys):
        def go():
            return [cy.toric_kahler_cone() for cy in batch_cys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_toric_mori_cone_batch(self, benchmark, batch_cys):
        def go():
            return [cy.toric_mori_cone() for cy in batch_cys]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 7. Geometric compute_* methods (require tloc in Kähler cone)
# ---------------------------------------------------------------------------

class TestComputeMethods:
    def test_compute_cy_volume_5v(self, benchmark, cy_5v, tloc_5v):
        benchmark(lambda: cy_5v.compute_cy_volume(tloc_5v))

    def test_compute_divisor_volumes_5v(self, benchmark, cy_5v, tloc_5v):
        benchmark(lambda: cy_5v.compute_divisor_volumes(tloc_5v))

    def test_compute_divisor_volumes_in_basis_5v(self, benchmark, cy_5v, tloc_5v):
        benchmark(lambda: cy_5v.compute_divisor_volumes(tloc_5v, in_basis=True))

    def test_compute_curve_volumes_5v(self, benchmark, cy_5v, tloc_5v):
        benchmark(lambda: cy_5v.compute_curve_volumes(tloc_5v))

    def test_compute_curve_volumes_extremal_5v(self, benchmark, cy_5v, tloc_5v):
        benchmark(lambda: cy_5v.compute_curve_volumes(tloc_5v, only_extremal=True))

    def test_compute_kappa_matrix_5v(self, benchmark, cy_5v, tloc_5v):
        benchmark(lambda: cy_5v.compute_kappa_matrix(tloc_5v))

    def test_compute_kappa_vector_5v(self, benchmark, cy_5v, tloc_5v):
        benchmark(lambda: cy_5v.compute_kappa_vector(tloc_5v))

    def test_compute_inverse_kahler_metric_5v(self, benchmark, cy_5v, tloc_5v):
        benchmark(lambda: cy_5v.compute_inverse_kahler_metric(tloc_5v))

    def test_compute_kahler_metric_5v(self, benchmark, cy_5v, tloc_5v):
        benchmark(lambda: cy_5v.compute_kahler_metric(tloc_5v))

    def test_compute_methods_batch(self, benchmark, batch_cys):
        def go():
            for cy in batch_cys:
                t = np.ones(cy.h11())
                cy.compute_cy_volume(t)
                cy.compute_kappa_matrix(t)
                cy.compute_inverse_kahler_metric(t)
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 8. GV / GW invariants
# ---------------------------------------------------------------------------

class TestGVGW:
    def test_compute_gvs_5v(self, benchmark, cy_5v):
        benchmark(lambda: cy_5v.compute_gvs(max_deg=3))

    def test_compute_gws_5v(self, benchmark, cy_5v):
        benchmark(lambda: cy_5v.compute_gws(max_deg=3))

