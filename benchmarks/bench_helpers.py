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
"""

import copy

import numpy as np
import pytest

from cytools import Polytope
from cytools.utils import integral_nullspace, lll_reduce


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
