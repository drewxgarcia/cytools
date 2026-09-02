"""Shared fixtures for the CYTools benchmark suite."""

from __future__ import annotations

import os

import pytest

from benchmarks._data import load_h11_sample, load_tier


@pytest.fixture(scope="session", autouse=True)
def _gv_subprocess_start_method():
    """Measure the GV kernel rather than subprocess start-up.

    `cygv.compute_gv` runs in a fresh process per call, and under the macOS
    default (`spawn`) that costs ~137 ms of re-importing numpy/cygv -- which is
    *independent of max_deg*, so `test_compute_gvs_5v` at max_deg=3 was ~92%
    start-up. `configure_gv_subprocess()` switches to `forkserver`, measured
    here at 12.8x / 11.9x / 9.0x for max_deg 1 / 3 / 6 with identical
    invariants. Session-scoped because the start method is process-global.
    """
    from cytools.calabiyau import configure_gv_subprocess

    configure_gv_subprocess()


# Database records ---------------------------------------------------------


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
def bulk_polys():
    """The representative 13–17 vertex-count population."""
    return load_tier("bulk")


@pytest.fixture(scope="module")
def full_polys():
    """A sample from every available vertex-count file."""
    return load_tier("full")


# Polytope objects ---------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_poly_objects(tiny_polys):
    return [record.polytope for record in tiny_polys]


@pytest.fixture(scope="module")
def small_poly_objects(small_polys):
    return [record.polytope for record in small_polys]


@pytest.fixture(scope="module")
def medium_poly_objects(medium_polys):
    return [record.polytope for record in medium_polys]


@pytest.fixture(scope="module")
def bulk_poly_objects(bulk_polys):
    return [record.polytope for record in bulk_polys]


# Bounded-Hodge samples ----------------------------------------------------


@pytest.fixture(scope="module")
def cone_polys():
    """Records with CY ``h11 <= 4`` for inexpensive cone operations."""
    n = int(os.environ.get("CYTOOLS_BENCH_N_CONE_SMALL", "20"))
    return load_h11_sample(range(1, 5), n)


@pytest.fixture(scope="module")
def cone_polys_large():
    """Records with CY ``h11 <= 8`` for slow cone sweeps."""
    n = int(os.environ.get("CYTOOLS_BENCH_N_CONE_LARGE", "100"))
    return load_h11_sample(range(1, 9), n)


@pytest.fixture(scope="module")
def cy_polys():
    """Records with CY ``h11 <= 4`` for the standard CY pipeline.

    Favorable only: these feed ``get_cy()``, which rejects a non-favorable
    polytope outright, and a single such row errors the whole fixture.
    """
    n = int(os.environ.get("CYTOOLS_BENCH_N_CY", "20"))
    return load_h11_sample(range(1, 5), n, favorable=True)


@pytest.fixture(scope="module")
def cy_polys_large():
    """Records with CY ``h11 <= 8`` for slow CY sweeps.

    Favorable only: these feed ``get_cy()``, which rejects a non-favorable
    polytope outright, and a single such row errors the whole fixture.
    """
    n = int(os.environ.get("CYTOOLS_BENCH_N_CY_LARGE", "100"))
    return load_h11_sample(range(1, 9), n, favorable=True)


@pytest.fixture(scope="module")
def cy_polys_median():
    """Records in the representative ``20 <= h11 <= 35`` range.

    Favorable only: these feed ``get_cy()``, which rejects a non-favorable
    polytope outright, and a single such row errors the whole fixture.
    """
    n = int(os.environ.get("CYTOOLS_BENCH_N_CY_MEDIAN", "20"))
    return load_h11_sample(range(20, 36), n, favorable=True)


@pytest.fixture(scope="module")
def reflexive_poly_objects(cy_polys):
    return [record.polytope for record in cy_polys]
