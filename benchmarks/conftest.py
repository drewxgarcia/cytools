"""
pytest-benchmark configuration for the CYTools benchmark suite.

Registers custom markers and sets sensible defaults so benchmarks
don't time out on the larger tiers.

Module-scope fixtures shared across bench_*.py files are defined here so
the DB is loaded only once per pytest session per tier.
"""

import os

import pytest

from cytools.dataset import load_polytopes, load_tier


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks a benchmark as slow (skipped with -m 'not slow')")


def pytest_collection_modifyitems(config, items):
    # Auto-mark scaling tests as slow.  Do NOT auto-mark by name suffix —
    # tests use explicit @pytest.mark.slow where warranted.
    for item in items:
        name = item.name.lower()
        if "scaling" in name:
            item.add_marker(pytest.mark.slow)


# ---------------------------------------------------------------------------
# Shared tier fixtures  (vertex-count based — unbounded cone dimension)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_polys():
    return load_tier("tiny")


@pytest.fixture(scope="module")
def small_polys():
    return load_tier("small")


@pytest.fixture(scope="module")
def medium_polys():
    return load_tier("medium")


@pytest.fixture(scope="module")
def full_polys():
    """
    100 polytopes sampled from each vertex-count file in the KS database
    (~2900 total), covering the full complexity distribution.
    Override sample size via CYTOOLS_BENCH_N_FULL env var.
    """
    return load_tier("full")


# ---------------------------------------------------------------------------
# Cone-benchmark fixtures  (h12-filtered — bounded Kähler cone dimension)
#
# For 4D KS reflexive polytopes, the Kähler cone ambient dimension equals
# the polytope's h12 (= dual polytope's h11).  Querying by h12 at the DB
# layer gives us exactly the population we want without any post-hoc
# filtering or hanging.
#
#   h12 <= 4  →  cone dim <= 4   (~1500 polytopes in the KS database)
#   h12 <= 8  →  cone dim <= 8   (~50k polytopes)
#
# Override sample sizes via env vars:
#   CYTOOLS_BENCH_N_CONE_SMALL  (default 20)
#   CYTOOLS_BENCH_N_CONE_LARGE  (default 100)
# ---------------------------------------------------------------------------

def _load_cone_polys(h12_max: int, n: int, seed: int = 42):
    """Return n polytopes with h12 <= h12_max, sampled from the full DB.

    Distributes the sample evenly across h12 values 1..h12_max so the
    result covers the full cone-dimension range without scanning the
    entire database.
    """
    import math
    n_per_h12 = max(1, math.ceil(n / h12_max))
    records = []
    for h12 in range(1, h12_max + 1):
        records.extend(load_polytopes(h12=h12, n=n_per_h12, seed=seed))
    # trim to exactly n (may have slightly more due to ceiling division)
    return records[:n]


@pytest.fixture(scope="module")
def cone_polys():
    """20 polytopes with h12 <= 4, giving Kähler cones of ambient dim <= 4.

    Safe for all dualize()-dependent cone operations: rays(), hyperplanes(),
    extremal_rays(), intersection(), hilbert_basis().
    """
    n = int(os.environ.get("CYTOOLS_BENCH_N_CONE_SMALL", "20"))
    return _load_cone_polys(h12_max=4, n=n)


@pytest.fixture(scope="module")
def cone_polys_large():
    """100 polytopes with h12 <= 8, giving Kähler cones of ambient dim <= 8.

    For slow-marked sweeps of dualize()-dependent operations.
    """
    n = int(os.environ.get("CYTOOLS_BENCH_N_CONE_LARGE", "100"))
    return _load_cone_polys(h12_max=8, n=n)


# ---------------------------------------------------------------------------
# CY-benchmark fixtures  (h11-filtered — bounded CY Hodge complexity)
#
# For 4D KS reflexive polytopes, h11 of the polytope equals the Hodge number
# h^{1,1} of the CY hypersurface.  Small h11 means:
#   - fewer divisors → simpler intersection number tensor
#   - simpler Kähler metric / GV computations
#   - get_cy() is always admissible for KS reflexive polytopes
#
# These polytopes are safe for the full CY pipeline without try/except guards.
#
#   h11 <= 4   →  ~5k polytopes in the KS database
#   h11 <= 8   →  ~50k polytopes
#
# Override sample sizes via env vars:
#   CYTOOLS_BENCH_N_CY        (default 20)
#   CYTOOLS_BENCH_N_CY_LARGE  (default 100)
# ---------------------------------------------------------------------------

def _load_cy_polys(h11_max: int, n: int, seed: int = 42):
    """Return n polytopes with h11 <= h11_max, sampled evenly across h11 values."""
    import math
    n_per_h11 = max(1, math.ceil(n / h11_max))
    records = []
    for h11 in range(1, h11_max + 1):
        records.extend(load_polytopes(h11=h11, n=n_per_h11, seed=seed))
    return records[:n]


@pytest.fixture(scope="module")
def cy_polys():
    """20 polytopes with h11 <= 4.

    Safe for the full CY pipeline (triangulate → get_toric_variety → get_cy)
    without any try/except guards.  h11 bounds the Hodge complexity.
    """
    n = int(os.environ.get("CYTOOLS_BENCH_N_CY", "20"))
    return _load_cy_polys(h11_max=4, n=n)


@pytest.fixture(scope="module")
def cy_polys_large():
    """100 polytopes with h11 <= 8.

    For slow-marked sweeps of CY operations.
    """
    n = int(os.environ.get("CYTOOLS_BENCH_N_CY_LARGE", "100"))
    return _load_cy_polys(h11_max=8, n=n)
