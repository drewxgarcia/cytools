"""
End-to-end pipeline benchmarks grounded in the real KS database distribution.

Background
----------
The KS reflexive 4D database contains 438M polytopes across 29 vertex-count
files.  The distribution is highly skewed:

  5-8v    (tiny/small tiers):   1M polytopes  =  0.2%  of the DB,  h11 median ~40-50
  9-12v   (medium tier):       56M polytopes  = 12.9%  of the DB,  h11 median ~40-45
  13-17v  (bulk tier):        289M polytopes  = 65.9%  of the DB,  h11 median ~30-40
  18-29v  (large tier):        92M polytopes  = 21.0%  of the DB,  h11 median ~20-30

All existing benchmarks defaulted to tiny/small/medium tiers and
cy_polys (h11 <= 4), which together represent the extreme low-complexity
tail — <1% of the database and 5-10× faster than a typical KS polytope.

This file fixes that by making the ``bulk`` tier (13-17v, 65.9% of DB mass)
the primary non-slow population.  The tiny tier is retained as a fast
calibration point.

Population hierarchy
--------------------
``tiny``    5v, 20 polytopes — fast calibration, always runs, h11 ~34 median
``bulk``    13-17v, 20 polytopes — the real-world representative population
``full``    all vertex-count files, ~2900 polytopes — marked slow
``cy_polys_median``  h11 in [20,35], 20 polytopes — realistic CY workload

Stages benchmarked
------------------
Stage 1  — triangulate()
Stage 2  — triangulate() → get_toric_variety()
Stage 3  — triangulate() → get_toric_variety() → intersection_numbers()
Stage 4  — triangulate() → get_cy() → intersection_numbers()
Stage 5  — Hodge number extraction (h11/h12) at scale

Run fast suite (calibration + bulk, no DB needed for tiny):
    CYTOOLS_DB_DIR=~/Downloads/polytopes-4d \\
        pytest benchmarks/bench_pipeline.py --benchmark-only -m "not slow"

Run full suite:
    CYTOOLS_DB_DIR=~/Downloads/polytopes-4d \\
        pytest benchmarks/bench_pipeline.py --benchmark-only
"""

import pytest


# ---------------------------------------------------------------------------
# Fixtures — thin wrappers so conftest fixtures get meaningful local names
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_records(tiny_polys):
    """5v polytopes — fast calibration, h11 median ~34."""
    return tiny_polys


@pytest.fixture(scope="module")
def bulk_records(bulk_polys):
    """13-17v polytopes — 65.9% of the KS database, h11 median ~30-40."""
    return bulk_polys


@pytest.fixture(scope="module")
def full_records(full_polys):
    """All vertex-count files, ~2900 polytopes — full complexity distribution."""
    return full_polys


# ---------------------------------------------------------------------------
# Stage 1: Polytope → Triangulation
# ---------------------------------------------------------------------------

class TestStage1Triangulate:
    """How fast can we triangulate polytopes across the complexity range?"""

    def test_triangulate_tiny(self, benchmark, tiny_records):
        """Calibration: 20 × 5v polytopes."""
        polys = [r.polytope for r in tiny_records]
        def go():
            return [p.triangulate() for p in polys]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_triangulate_bulk(self, benchmark, bulk_records):
        """Primary: 20 polytopes drawn from the DB bulk (13-17v)."""
        polys = [r.polytope for r in bulk_records]
        def go():
            return [p.triangulate() for p in polys]
        benchmark.pedantic(go, rounds=3, iterations=1)

    @pytest.mark.slow
    def test_triangulate_full(self, benchmark, full_records):
        """Full distribution: ~2900 polytopes across all vertex-count files."""
        polys = [r.polytope for r in full_records]
        def go():
            return [p.triangulate() for p in polys]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# Stage 2: Polytope → Triangulation → ToricVariety
# ---------------------------------------------------------------------------

class TestStage2ToricVariety:
    """Full pipeline up to toric variety construction."""

    def test_toric_variety_tiny(self, benchmark, tiny_records):
        """Calibration: 20 × 5v polytopes."""
        polys = [r.polytope for r in tiny_records]
        def go():
            results = []
            for p in polys:
                try:
                    results.append(p.triangulate().get_toric_variety())
                except Exception:
                    pass
            return results
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_toric_variety_bulk(self, benchmark, bulk_records):
        """Primary: 20 polytopes from the DB bulk (13-17v)."""
        polys = [r.polytope for r in bulk_records]
        def go():
            results = []
            for p in polys:
                try:
                    results.append(p.triangulate().get_toric_variety())
                except Exception:
                    pass
            return results
        benchmark.pedantic(go, rounds=3, iterations=1)

    @pytest.mark.slow
    def test_toric_variety_full(self, benchmark, full_records):
        """Full distribution: ~2900 polytopes across all vertex-count files."""
        polys = [r.polytope for r in full_records]
        def go():
            results = []
            for p in polys:
                try:
                    results.append(p.triangulate().get_toric_variety())
                except Exception:
                    pass
            return results
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# Stage 3: Polytope → ToricVariety → intersection_numbers
# ---------------------------------------------------------------------------

class TestStage3IntersectionNumbers:
    """Full pipeline through intersection numbers.

    cy_polys (h11<=4) gives a fast low-complexity calibration.
    cy_polys_median (h11 in [20,35]) gives the realistic workload — these
    polytopes are 5-10× slower and represent the true typical-complexity regime.
    """

    def test_intnum_calibration(self, benchmark, cy_polys):
        """Calibration: 20 polytopes with h11<=4 (extreme low tail, ~5ms each)."""
        def go():
            results = []
            for r in cy_polys:
                tv = r.polytope.triangulate().get_toric_variety()
                results.append(tv.intersection_numbers())
            return results
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_intnum_bulk(self, benchmark, bulk_records):
        """Primary: 20 polytopes from the DB bulk (13-17v, h11 ~25-40)."""
        def go():
            results = []
            for r in bulk_records:
                try:
                    tv = r.polytope.triangulate().get_toric_variety()
                    results.append(tv.intersection_numbers())
                except Exception:
                    pass
            return results
        benchmark.pedantic(go, rounds=1, iterations=1)

    def test_intnum_median_h11(self, benchmark, cy_polys_median):
        """Realistic: 20 polytopes with h11 in [20,35] — the DB median range."""
        def go():
            results = []
            for r in cy_polys_median:
                try:
                    tv = r.polytope.triangulate().get_toric_variety()
                    results.append(tv.intersection_numbers())
                except Exception:
                    pass
            return results
        benchmark.pedantic(go, rounds=1, iterations=1)

    @pytest.mark.slow
    def test_intnum_full(self, benchmark, full_records):
        """Full distribution: ~2900 polytopes across all vertex-count files."""
        def go():
            results = []
            for r in full_records:
                try:
                    tv = r.polytope.triangulate().get_toric_variety()
                    results.append(tv.intersection_numbers())
                except Exception:
                    pass
            return results
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# Stage 4: Full CY pipeline — Polytope → CalabiYau → geometry
# ---------------------------------------------------------------------------

class TestStage4CYPipeline:
    """End-to-end: from raw polytope to CY geometric data.

    cy_polys is h11-filtered so every polytope is CY-admissible — no
    try/except noise.  cy_polys_median is the realistic-complexity version.
    """

    def test_cy_pipeline_calibration(self, benchmark, cy_polys):
        """Calibration: 20 polytopes with h11<=4 (extreme low tail)."""
        def go():
            return [
                r.polytope.triangulate().get_cy().intersection_numbers()
                for r in cy_polys
            ]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_cy_pipeline_median_h11(self, benchmark, cy_polys_median):
        """Primary: 20 polytopes with h11 in [20,35] — the realistic CY workload."""
        def go():
            results = []
            for r in cy_polys_median:
                try:
                    results.append(
                        r.polytope.triangulate().get_cy().intersection_numbers()
                    )
                except Exception:
                    pass
            return results
        benchmark.pedantic(go, rounds=1, iterations=1)

    @pytest.mark.slow
    def test_cy_pipeline_large(self, benchmark, cy_polys_large):
        """100 polytopes with h11 <= 8 — broader Hodge range."""
        def go():
            return [
                r.polytope.triangulate().get_cy().intersection_numbers()
                for r in cy_polys_large
            ]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# Stage 5: Hodge / topology throughput
#
# The dominant use case for database scan workflows: extract h11/h12 from
# as many polytopes as possible.  The bulk population gives the realistic
# throughput number that matters for KS scan workloads.
# ---------------------------------------------------------------------------

class TestHodgeThroughput:
    """h11/h12 extraction throughput across the complexity spectrum."""

    def test_hodge_calibration(self, benchmark, cy_polys):
        """Calibration: 20 polytopes with h11<=4."""
        def go():
            out = []
            for r in cy_polys:
                cy = r.polytope.triangulate().get_cy()
                out.append((cy.h11(), cy.h12()))
            return out
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_hodge_bulk(self, benchmark, bulk_records):
        """Primary: 20 polytopes from the DB bulk (13-17v).

        This number reflects the actual throughput a user sees when scanning
        the KS database for phenomenologically interesting models.
        """
        def go():
            out = []
            for r in bulk_records:
                try:
                    cy = r.polytope.triangulate().get_cy()
                    out.append((cy.h11(), cy.h12()))
                except Exception:
                    pass
            return out
        benchmark.pedantic(go, rounds=3, iterations=1)

    @pytest.mark.slow
    def test_hodge_large(self, benchmark, cy_polys_large):
        """100 polytopes with h11 <= 8."""
        def go():
            out = []
            for r in cy_polys_large:
                cy = r.polytope.triangulate().get_cy()
                out.append((cy.h11(), cy.h12()))
            return out
        benchmark.pedantic(go, rounds=1, iterations=1)

    @pytest.mark.slow
    def test_hodge_full(self, benchmark, full_records):
        """Full distribution: ~2900 polytopes across all vertex-count files."""
        def go():
            out = []
            for r in full_records:
                try:
                    cy = r.polytope.triangulate().get_cy()
                    out.append((cy.h11(), cy.h12()))
                except Exception:
                    pass
            return out
        benchmark.pedantic(go, rounds=1, iterations=1)
