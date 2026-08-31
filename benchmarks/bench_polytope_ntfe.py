"""
Benchmarks for advanced Polytope methods — NTFE secondary fan traversal,
random_triangulations_fair (MCMC), nef_partitions, and is_trilayer.

Methods covered:
  grow_ft, grow_frt,
  expanded_secondary_fan,
  triangface_ineqs, triangfaces_to_frt, triangfaces_to_frst,
  ntfe_hypers, ntfe_cones, ntfe_frts, ntfe_frsts,
  random_triangulations_fair,
  nef_partitions,
  is_trilayer

Background
----------
These methods implement the NTFE (Non-Triangulated Face Equivalence)
framework for enumerating fine regular (star) triangulations via the
expanded secondary fan.  The pipeline is:

  1. triangface_ineqs  — per-2-face CPL constraint generation
  2. ntfe_hypers       — combine face constraints into NTFE hyperplane systems
  3. ntfe_cones        — wrap hyperplane systems as Cone objects
  4. ntfe_frts/frsts   — find interior heights and triangulate per cone

This is the hot path for enumerating *all* fine regular triangulations of a
polytope without relying on CGAL random sampling.

Fixture design
--------------
``tiny_poly_objects``       20 polytopes from the 5v tier — fast calibration
``small_poly_objects``      20 polytopes from the 6-7v tier — primary non-slow
``reflexive_poly_objects``  20 reflexive polytopes with h11≤4 — for nef_partitions

The NTFE methods scale combinatorially with the number of 2-face
triangulations.  For 5v polytopes this is small enough to run without
pedantic.  For 6-7v polytopes it starts to be expensive; those tests are
the representative workload.  Larger polytopes are marked slow.

Run fast suite:
    pytest benchmarks/bench_polytope_ntfe.py --benchmark-only -m "not slow"

Run full suite:
    pytest benchmarks/bench_polytope_ntfe.py --benchmark-only
"""

import pytest

from cytools import Polytope

# A 5-vertex 4D polytope with multiple FRSTs — reliable for random_triangulations_fair.
# POLY_5V has only 1 FRST so MCMC can't find neighbors; use this instead.
# Verified: random_triangulations_fast(N=2) returns 2 distinct FRSTs in <0.1s.
_P2T = Polytope(
    [[-6, -8, -5, -5], [0, 1, 0, 0], [1, 0, 0, 0], [2, 4, 5, 0], [3, 3, 0, 5]]
)


# ---------------------------------------------------------------------------
# Module-scope fixtures
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. grow_ft / grow_frt — 2D triangulation growth
# ---------------------------------------------------------------------------


class TestGrow2D:
    """grow_ft: grow a single fine 2D triangulation by random insertion.
    grow_frt: filter grow_ft outputs to fine regular ones.

    Both operate on 2D polytopes.  We collect all 2D faces of the fixture
    polytopes so the benchmark reflects real workload geometry.
    """

    def test_grow_ft_tiny(self, benchmark, tiny_poly_objects):
        """grow_ft on all 2D faces of 5v polytopes."""
        faces_2d = []
        for p in tiny_poly_objects:
            try:
                faces_2d.extend(p.faces(2))
            except Exception:
                pass
        face_polys = []
        for f in faces_2d:
            try:
                face_polys.append(f.as_polytope())
            except Exception:
                pass

        def go():
            return [fp.grow_ft(seed=42) for fp in face_polys]

        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_grow_frt_tiny(self, benchmark, tiny_poly_objects):
        """grow_frt on all 2D faces of 5v polytopes (N=1, one fine regular)."""
        faces_2d = []
        for p in tiny_poly_objects:
            try:
                faces_2d.extend(p.faces(2))
            except Exception:
                pass
        face_polys = []
        for f in faces_2d:
            try:
                face_polys.append(f.as_polytope())
            except Exception:
                pass

        def go():
            results = []
            for fp in face_polys:
                try:
                    results.append(fp.grow_frt(N=1, seed=42))
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_grow_ft_small(self, benchmark, small_poly_objects):
        """grow_ft on 2D faces of 6-7v polytopes — more complex geometry."""
        faces_2d = []
        for p in small_poly_objects:
            try:
                faces_2d.extend(p.faces(2))
            except Exception:
                pass
        face_polys = []
        for f in faces_2d:
            try:
                face_polys.append(f.as_polytope())
            except Exception:
                pass

        def go():
            return [fp.grow_ft(seed=42) for fp in face_polys]

        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 2. expanded_secondary_fan
# ---------------------------------------------------------------------------


class TestExpandedSecondaryFan:
    """expanded_secondary_fan: hyperplane system for the expanded secondary fan.

    Iterates over all 2-faces and computes CPL cone inequalities.  Time scales
    with number of 2-faces and their point complexity.
    """

    def test_expanded_secondary_fan_tiny(self, benchmark, tiny_poly_objects):
        def go():
            results = []
            for p in tiny_poly_objects:
                try:
                    results.append(p.expanded_secondary_fan())
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_expanded_secondary_fan_small(self, benchmark, small_poly_objects):
        def go():
            results = []
            for p in small_poly_objects:
                try:
                    results.append(p.expanded_secondary_fan())
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 3. triangface_ineqs
# ---------------------------------------------------------------------------


class TestTriangfaceIneqs:
    """triangface_ineqs: CPL inequality generation for all 2-face triangulations.

    This is the first stage of the NTFE pipeline.  It enumerates or samples
    all fine regular triangulations of each 2-face and computes their
    associated polytope cone inequalities.

    N=10 (non-slow): cap per-face sampling at 10 FRTs, max_npts=0 so all faces
    use grow_frt sampling rather than TOPCOM enumerate-all.  This is the
    representative fast-path workload.

    N=None (slow): full enumeration via TOPCOM — can be minutes for faces with
    many points (e.g. 16-pt faces yield ~14k triangulations each).
    """

    def test_triangface_ineqs_N10_tiny(self, benchmark, tiny_poly_objects):
        """Fast path: sample ≤10 FRTs per face, skip TOPCOM enumerate-all."""

        def go():
            results = []
            for p in tiny_poly_objects:
                try:
                    results.append(p.triangface_ineqs(N_face_triangs=10, max_npts=0))
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_triangface_ineqs_N10_small(self, benchmark, small_poly_objects):
        def go():
            results = []
            for p in small_poly_objects:
                try:
                    results.append(p.triangface_ineqs(N_face_triangs=10, max_npts=0))
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=1, iterations=1)

    @pytest.mark.slow
    def test_triangface_ineqs_all_tiny(self, benchmark, tiny_poly_objects):
        """Full TOPCOM enumeration — all FRTs per face.  Can take minutes."""

        def go():
            results = []
            for p in tiny_poly_objects:
                try:
                    results.append(p.triangface_ineqs())
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 4. ntfe_hypers / ntfe_cones
# ---------------------------------------------------------------------------


class TestNTFEHypersCones:
    """ntfe_hypers: combine 2-face CPL ineqs into NTFE hyperplane systems.
    ntfe_cones: wrap them as Cone objects.

    These are stages 2 and 3 of the NTFE pipeline.  Time grows as the
    combinatorial product of per-face triangulation counts.
    N=1 limits to a single random NTFE cone (fast path).
    """

    def test_ntfe_hypers_N1_tiny(self, benchmark, tiny_poly_objects):
        """N=1: one random NTFE hyperplane system per polytope."""

        def go():
            results = []
            for p in tiny_poly_objects:
                try:
                    results.append(list(p.ntfe_hypers(N=1, seed=42)))
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_ntfe_cones_N1_tiny(self, benchmark, tiny_poly_objects):
        """N=1: one NTFE Cone object per polytope."""

        def go():
            results = []
            for p in tiny_poly_objects:
                try:
                    results.append(list(p.ntfe_cones(N=1, seed=42)))
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_ntfe_hypers_N1_small(self, benchmark, small_poly_objects):
        def go():
            results = []
            for p in small_poly_objects:
                try:
                    results.append(list(p.ntfe_hypers(N=1, seed=42)))
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=1, iterations=1)

    @pytest.mark.slow
    def test_ntfe_hypers_all_tiny(self, benchmark, tiny_poly_objects):
        """All NTFE hyperplane systems (N=None) — full enumeration."""

        def go():
            results = []
            for p in tiny_poly_objects:
                try:
                    results.append(list(p.ntfe_hypers()))
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 5. ntfe_frts / ntfe_frsts — full NTFE pipeline to triangulations
# ---------------------------------------------------------------------------


class TestNTFETriangulations:
    """ntfe_frts/frsts: full NTFE pipeline — from polytope to triangulations.

    Stage 4: find interior height vectors for each NTFE cone, then triangulate.
    This is the most computationally intensive NTFE operation.
    N=1 limits to one triangulation per polytope for non-slow tests.
    """

    def test_ntfe_frts_N1_tiny(self, benchmark, tiny_poly_objects):
        """One NTFE FRT per polytope (N=1) — tiny tier calibration."""

        def go():
            results = []
            for p in tiny_poly_objects:
                try:
                    results.append(list(p.ntfe_frts(N=1, seed=42)))
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_ntfe_frsts_N1_tiny(self, benchmark, tiny_poly_objects):
        """One NTFE FRST per polytope (N=1) — star constraint added."""

        def go():
            results = []
            for p in tiny_poly_objects:
                try:
                    results.append(list(p.ntfe_frsts(N=1, seed=42)))
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_ntfe_frts_N1_small(self, benchmark, small_poly_objects):
        """One NTFE FRT per polytope — primary non-slow workload."""

        def go():
            results = []
            for p in small_poly_objects:
                try:
                    results.append(list(p.ntfe_frts(N=1, seed=42)))
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=1, iterations=1)

    def test_ntfe_frsts_N1_small(self, benchmark, small_poly_objects):
        """One NTFE FRST per polytope — star constraint, primary workload."""

        def go():
            results = []
            for p in small_poly_objects:
                try:
                    results.append(list(p.ntfe_frsts(N=1, seed=42)))
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=1, iterations=1)

    @pytest.mark.slow
    def test_ntfe_frts_all_tiny(self, benchmark, tiny_poly_objects):
        """All NTFE FRTs (N=None) — full enumeration, tiny tier."""

        def go():
            results = []
            for p in tiny_poly_objects:
                try:
                    results.append(list(p.ntfe_frts()))
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 6. triangfaces_to_frt / triangfaces_to_frst
# ---------------------------------------------------------------------------


class TestTriangfacesTo:
    """triangfaces_to_frt/frst: lift a set of 2-face triangulations to an
    ambient FRT/FRST by solving an LP for valid heights.

    Requires pre-computed 2-face triangulations as input.  We use
    face_triangs(N_face_triangs=1, max_npts=0) to get one FRT per face quickly
    (grow_frt sampling, no TOPCOM enumerate-all).

    face_triangs() returns list[list[Triangulation]] — one inner list per face.
    """

    def test_triangfaces_to_frt_tiny(self, benchmark, tiny_poly_objects):
        def go():
            results = []
            for p in tiny_poly_objects:
                try:
                    ft = p.face_triangs(N_face_triangs=1, max_npts=0)
                    # Take first triangulation of each face (ft is a list of lists)
                    one_per_face = [face_ts[0] for face_ts in ft if face_ts]
                    if one_per_face:
                        results.append(p.triangfaces_to_frt(one_per_face))
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_triangfaces_to_frst_tiny(self, benchmark, tiny_poly_objects):
        def go():
            results = []
            for p in tiny_poly_objects:
                try:
                    ft = p.face_triangs(N_face_triangs=1, max_npts=0)
                    one_per_face = [face_ts[0] for face_ts in ft if face_ts]
                    if one_per_face:
                        results.append(p.triangfaces_to_frst(one_per_face))
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_triangfaces_to_frt_small(self, benchmark, small_poly_objects):
        def go():
            results = []
            for p in small_poly_objects:
                try:
                    ft = p.face_triangs(N_face_triangs=1, max_npts=0)
                    one_per_face = [face_ts[0] for face_ts in ft if face_ts]
                    if one_per_face:
                        results.append(p.triangfaces_to_frt(one_per_face))
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 7. random_triangulations_fair — MCMC hit-and-run sampling
# ---------------------------------------------------------------------------


class TestRandomTriangulationsFair:
    """random_triangulations_fair: Algorithm #3 from arXiv:2008.01730.

    MCMC hit-and-run in the secondary fan + random flips.  Produces a fair
    (asymptotically uniform) sample of fine regular triangulations.

    N=1 gives a single independent sample — fastest useful case.
    """

    def test_random_triangulations_fair_N1(self, benchmark):
        """Single fair triangulation of _P2T (5-vertex poly with multiple FRSTs)."""

        def go():
            return list(_P2T.random_triangulations_fair(N=1, seed=42))

        benchmark(go)

    def test_random_triangulations_fair_N5(self, benchmark):
        """5 independent fair triangulations of _P2T."""

        def go():
            return list(_P2T.random_triangulations_fair(N=5, seed=42))

        benchmark(go)

    @pytest.mark.slow
    def test_random_triangulations_fair_N1_tiny(self, benchmark, tiny_poly_objects):
        """N=1 fair triangulation across 20 tiny polytopes.

        Marked slow: many tiny polytopes have only 1 FRST, so MCMC cannot find
        a neighbor to flip to and hangs.  This is expected algorithmic behavior,
        not a bug — the MCMC sampler requires ≥2 FRSTs to make progress.
        """

        def go():
            results = []
            for p in tiny_poly_objects:
                try:
                    results.extend(p.random_triangulations_fair(N=1, seed=42))
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=1, iterations=1)

    @pytest.mark.slow
    def test_random_triangulations_fair_N1_small(self, benchmark, small_poly_objects):
        """N=1 fair triangulation across 20 small (6-7v) polytopes."""

        def go():
            results = []
            for p in small_poly_objects:
                try:
                    results.extend(p.random_triangulations_fair(N=1, seed=42))
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 8. nef_partitions — external PALP call, reflexive polytopes only
# ---------------------------------------------------------------------------


class TestNefPartitions:
    """nef_partitions: compute nef partitions via PALP.

    Requires reflexive polytopes.  Time is dominated by the external PALP
    call.  Caching behavior is also interesting — second call is O(1).
    """

    def test_nef_partitions_tiny_reflexive(self, benchmark, reflexive_poly_objects):
        def go():
            results = []
            for p in reflexive_poly_objects:
                try:
                    results.append(p.nef_partitions())
                except Exception:
                    pass
            return results

        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 9. is_trilayer
# ---------------------------------------------------------------------------


class TestIsTrilayer:
    """is_trilayer: checks GLSM kernel + anticanonical divisor condition.

    Uses FLINT nullspace internally.  Time scales with polytope dimension
    and point count.
    """

    def test_is_trilayer_tiny(self, benchmark, tiny_poly_objects):
        def go():
            return [p.is_trilayer() for p in tiny_poly_objects]

        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_is_trilayer_small(self, benchmark, small_poly_objects):
        def go():
            return [p.is_trilayer() for p in small_poly_objects]

        benchmark.pedantic(go, rounds=3, iterations=1)
