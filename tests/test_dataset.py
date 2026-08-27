"""Tests for the Parquet-backed KS dataset loader.

These are skipped unless a local 4D KS database is configured (CYTOOLS_DB_DIR
or the default download location), since they read real data.
"""

import numpy as np
import pytest

ds = pytest.importorskip("cytools.dataset")


def _load(**kwargs):
    try:
        return ds.load_polytopes(**kwargs)
    except ValueError as e:
        pytest.skip(f"no local KS database configured: {e}")


def _safe_extract(table):
    """Reference implementation: per-row Python extraction, no fast path."""
    col = table.column("vertices").combine_chunks()
    return [np.array(col[i].as_py(), dtype=np.int32) for i in range(len(table))]


@pytest.mark.parametrize(
    "n_vertices",
    [[13], [13, 14], [13, 14, 15], list(range(10, 18))],
    ids=["single", "two-files", "three-files", "eight-files"],
)
def test_vertex_extraction_matches_reference(n_vertices, monkeypatch):
    """The zero-copy fast path must agree with per-row extraction.

    Regression test: the fast path used to validate stride uniformity by
    comparing only the first and last row, so a query merging several
    vertex-count files could mis-slice vertices into the wrong rows and
    silently yield a corrupted (e.g. non-reflexive) polytope.
    """
    seen = []
    orig = ds._extract_vertices_from_table

    def spy(table):
        fast = orig(table)
        ref = _safe_extract(table)
        seen.append(sorted({len(r) for r in ref}))
        assert len(fast) == len(ref)
        for i, (a, b) in enumerate(zip(fast, ref)):
            assert np.array_equal(a, b), f"row {i} mismatch"
        return fast

    monkeypatch.setattr(ds, "_extract_vertices_from_table", spy)
    records = _load(n_vertices=n_vertices, n=16)

    assert records, "expected at least one record"
    assert seen, "extraction was never exercised"
    if len(n_vertices) > 1:
        assert any(len(s) > 1 for s in seen), "ragged case was not exercised"


def test_records_are_wellformed():
    records = _load(n_vertices=[13, 14, 15], n=12)
    for r in records:
        p = r.polytope
        assert p.vertices().shape == (r.vertex_count, 4)
        assert p.dim() == 4
        assert p.is_reflexive(), "corrupted vertices produce non-reflexive polytopes"


def test_hodge_columns_use_m_lattice_convention():
    """The DB's h11/h12/chi are M-lattice, opposite to CYTools' N default.

    Pinned so a future "fix" that swaps the columns cannot land silently.
    """
    records = _load(n_vertices=[13], n=5)
    for r in records:
        p = r.polytope
        assert r.h11 == p.h11(lattice="M")
        assert r.h11 == p.h21(lattice="N")
        assert r.h12 == p.h11(lattice="N")
        assert r.euler_characteristic == p.chi(lattice="M")
        assert r.euler_characteristic == -p.chi(lattice="N")
