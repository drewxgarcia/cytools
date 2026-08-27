"""
Micro-benchmarks for internal helpers and algorithmic anti-patterns
identified in the performance audit.

These are isolated, standalone tests — they don't require the KS database.
They measure:

  1. get_bdry  — the O(n²) list.index()+pop() edge-removal loop
                 (basic_geometry.get_bdry, called via a real Polytope)
  2. integral_nullspace / lll_reduce  (utils)
  3. np.linalg.matrix_rank baseline  (quantifies cache-miss cost)
  4. Double-sort pattern vs np.sort
  5. deepcopy vs np.copy
  6. list.index() vs dict lookup
  7. gcd_list — math.gcd dispatch vs float Euclidean (utils.gcd_list)
  8. solve_linear_system backends — cholesky vs scipy sparse across the dispatch range (n=100-2000)
"""

import copy
import math

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import pytest

from cytools import Polytope
from cytools.utils import integral_nullspace, lll_reduce, gcd_list


# ---------------------------------------------------------------------------
# 1. get_bdry — O(n²) edge-removal
# ---------------------------------------------------------------------------
#
# get_bdry lives on Polytope objects (delegates to a 2D triangulation).
# We time it on polytopes of increasing size so the scaling is visible.

_SMALL_POLYS_2D = [
    Polytope([[1, 0, 0], [0, 1, 0], [-1, -1, 0]]),                          # 3 verts
    Polytope([[2, 0, 0], [0, 2, 0], [-1, -1, 0], [1, -1, 0]]),              # 4 verts
    Polytope([[3, 0, 0], [0, 3, 0], [-1, -1, 0], [1, -1, 0], [-1, 2, 0]]), # 5 verts
]


class TestGetBdry:
    @pytest.mark.parametrize("poly", _SMALL_POLYS_2D, ids=["3v", "4v", "5v"])
    def test_get_bdry(self, benchmark, poly):
        benchmark(poly.get_bdry)


# ---------------------------------------------------------------------------
# 2. integral_nullspace / lll_reduce  (utils.py)
# ---------------------------------------------------------------------------

def _rand_int_matrix(rows: int, cols: int, seed: int = 0) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return rng.integers(-5, 6, size=(rows, cols)).tolist()


class TestLinearAlgebraHelpers:
    # Limit sizes: integral_nullspace internally uses flint and overflows on
    # matrices with very large intermediate values at n >= 60.
    @pytest.mark.parametrize("n", [10, 20, 30])
    def test_integral_nullspace(self, benchmark, n):
        m = _rand_int_matrix(n // 2, n)
        benchmark(integral_nullspace, m)

    @pytest.mark.parametrize("n", [10, 30, 60])
    def test_lll_reduce(self, benchmark, n):
        m = _rand_int_matrix(n, n)
        benchmark(lll_reduce, m)


# ---------------------------------------------------------------------------
# 3. np.linalg.matrix_rank baseline
#    Quantifies the cost of each uncached call in cone.py / utils.py
# ---------------------------------------------------------------------------

class TestNumpyRankBaseline:
    @pytest.mark.parametrize("n", [10, 30, 60, 100])
    def test_matrix_rank(self, benchmark, n):
        rng = np.random.default_rng(0)
        m = rng.integers(-3, 4, size=(n, n)).astype(float)
        benchmark(np.linalg.matrix_rank, m)


# ---------------------------------------------------------------------------
# 4. Double-sort pattern vs single np.sort
#    Reproduces triangulation.py: sorted([sorted(s) for s in simps])
# ---------------------------------------------------------------------------

class TestSortingOverhead:
    @pytest.mark.parametrize("n", [100, 500, 2000])
    def test_double_sorted_list(self, benchmark, n):
        rng = np.random.default_rng(0)
        simps = rng.integers(0, n, size=(n, 5)).tolist()
        benchmark(lambda: sorted([sorted(s) for s in simps]))

    @pytest.mark.parametrize("n", [100, 500, 2000])
    def test_numpy_sort_once(self, benchmark, n):
        """Proposed replacement: sort each row then lexsort the matrix."""
        rng = np.random.default_rng(0)
        simps = rng.integers(0, n, size=(n, 5))

        def go():
            s = np.sort(simps, axis=1)
            return s[np.lexsort(s.T[::-1])]

        benchmark(go)


# ---------------------------------------------------------------------------
# 5. deepcopy vs np.copy
#    Reproduces polytope.py: return copy.deepcopy(cached_array)
# ---------------------------------------------------------------------------

class TestCopyOverhead:
    @pytest.mark.parametrize("n", [100, 1000, 10_000])
    def test_deepcopy_array(self, benchmark, n):
        arr = np.random.default_rng(0).integers(0, 100, size=(n, 4))
        benchmark(copy.deepcopy, arr)

    @pytest.mark.parametrize("n", [100, 1000, 10_000])
    def test_np_copy_array(self, benchmark, n):
        arr = np.random.default_rng(0).integers(0, 100, size=(n, 4))
        benchmark(np.copy, arr)


# ---------------------------------------------------------------------------
# 6. list.index() vs pre-built dict lookup
#    Reproduces polytope.py: {i: pts.index(pt) for i, pt in enumerate(pts)}
# ---------------------------------------------------------------------------

class TestLookupOverhead:
    @pytest.mark.parametrize("n", [100, 500, 2000])
    def test_list_index_reverse_map(self, benchmark, n):
        lst = list(range(n))
        queries = lst[::10]
        benchmark(lambda: {q: lst.index(q) for q in queries})

    @pytest.mark.parametrize("n", [100, 500, 2000])
    def test_dict_reverse_map(self, benchmark, n):
        lst = list(range(n))
        d = {v: i for i, v in enumerate(lst)}
        queries = lst[::10]
        benchmark(lambda: {q: d[q] for q in queries})


# ---------------------------------------------------------------------------
# 7. gcd_list — math.gcd dispatch (integer fast path) vs gcd_float loop
#    Reproduces utils.py gcd_list, called from secondary_cone for every
#    hyperplane vector.  Small arrays of integers are the common case.
# ---------------------------------------------------------------------------

class TestGcdList:
    @pytest.mark.parametrize("n", [4, 16, 64])
    def test_gcd_list_integers(self, benchmark, n):
        """Integer inputs — exercises the math.gcd fast path."""
        rng = np.random.default_rng(0)
        arr = rng.integers(1, 100, size=n).tolist()
        benchmark(gcd_list, arr)

    @pytest.mark.parametrize("n", [4, 16, 64])
    def test_gcd_list_floats(self, benchmark, n):
        """Float inputs — exercises the gcd_float fallback path."""
        rng = np.random.default_rng(0)
        arr = (rng.integers(1, 100, size=n) * 1.0).tolist()
        benchmark(gcd_list, arr)

    @pytest.mark.parametrize("n", [4, 16, 64])
    def test_math_gcd_direct(self, benchmark, n):
        """Baseline: functools.reduce(math.gcd, ...) with no dispatch overhead."""
        import functools
        rng = np.random.default_rng(0)
        arr = rng.integers(1, 100, size=n).tolist()
        benchmark(lambda: functools.reduce(math.gcd, arr))


# ---------------------------------------------------------------------------
# 8. solve_linear_system backends — dense Cholesky vs scipy sparse
#
#    The dispatch threshold in utils.py was raised from n_vars<=500 to
#    n_vars<=2000 after profiling showed SuperLU consumed 41% of bulk
#    pipeline time.  KS bulk polytopes (h11~25-45) have n_vars~1100-1400
#    and cond(M^T M)~1e4-1e5; Cholesky residuals are ~1e-12 (safe).
#
#    Parameter sizes here mirror the actual dispatch range:
#      n=100   — old "small" regime (well below old 500 threshold)
#      n=500   — just below old threshold, just above new threshold overlap
#      n=1000  — bulk regime (h11~25-35, n_vars typically 1100-1400)
#      n=2000  — near the new threshold upper bound
# ---------------------------------------------------------------------------

def _make_spd_system(n: int, seed: int = 0):
    """
    Build a random sparse SPD matrix M (n x n) and RHS vector C (n,)
    suitable for solve_linear_system(M, C).

    M is constructed as A^T A + n*I so it is guaranteed positive definite.
    The sparse format matches what solve_linear_system expects.
    """
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.3, format="csc", dtype=float,
                  random_state=rng)
    M = A.T @ A + n * sp.eye(n, format="csc")
    C = rng.standard_normal(n)
    return M, C


class TestSolveLinearSystemBackends:
    @pytest.mark.parametrize("n", [100, 500, 1000, 2000])
    def test_scipy_sparse_backend(self, benchmark, n):
        """scipy spsolve — SuperLU sparse solver (used for n_vars > 2000)."""
        M, C = _make_spd_system(n)
        benchmark(lambda: sp.linalg.spsolve(M.T @ M, -(M.T @ C)).tolist())

    @pytest.mark.parametrize("n", [100, 500, 1000, 2000])
    def test_cholesky_backend(self, benchmark, n):
        """Dense Cholesky — used for n_vars <= 2000 (covers KS bulk, h11~25-45)."""
        M, C = _make_spd_system(n)
        MtM = (M.T @ M).toarray()
        MtC = -(M.T @ C)
        def go():
            c_fac, low = scipy.linalg.cho_factor(MtM)
            return scipy.linalg.cho_solve((c_fac, low), MtC).tolist()
        benchmark(go)
