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


# ---------------------------------------------------------------------------
# Batch-native scan
# ---------------------------------------------------------------------------


def _batches(**kwargs):
    try:
        return list(ds.scan_batches(**kwargs))
    except ValueError as e:
        pytest.skip(f"no local KS database configured: {e}")


def test_scan_batches_constructs_no_polytopes(monkeypatch):
    """The whole point: a scan must not materialize a Polytope per row.

    Regression guard for the architectural boundary. load_polytopes builds one
    Polytope per row eagerly; scan_batches must build none until asked.
    """
    from cytools.polytope import Polytope

    calls = []
    original = Polytope.__init__

    def counting_init(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Polytope, "__init__", counting_init)

    batches = _batches(n_vertices=[13, 14], n=40, batch_size=16)
    rows = sum(len(b) for b in batches)
    assert rows == 40

    # touching the geometry data must still not build objects
    total = sum(len(v) for b in batches for v in b.iter_vertices())
    assert total > 0
    assert not calls, f"scan built {len(calls)} Polytope objects; expected 0"

    # ...and materializing explicitly must build exactly one
    batches[0].polytope(0)
    assert len(calls) == 1


def test_batch_buffers_are_consistent():
    for b in _batches(n_vertices=[13, 14, 15], n=50, batch_size=20):
        assert b.vertex_values.flags["C_CONTIGUOUS"]
        assert b.vertex_offsets[0] == 0
        assert b.vertex_offsets[-1] == len(b.vertex_values)
        assert len(b.vertex_offsets) == len(b) + 1
        for i in range(len(b)):
            v = b.vertices(i)
            assert v.shape == (int(b.vertex_count[i]), 4)
            # a view into the flat buffer, not a copy
            assert v.base is b.vertex_values


def test_batch_agrees_with_load_polytopes():
    records = _load(n_vertices=[13], n=12)
    batches = _batches(n_vertices=[13], n=12, batch_size=64)
    flat = [(b, i) for b in batches for i in range(len(b))]
    assert len(flat) == len(records)

    for r, (b, i) in zip(records, flat):
        # Polytope canonicalizes vertex order, so compare as sets against the
        # raw stored buffer, and elementwise once both go through Polytope
        assert {tuple(v) for v in b.vertices(i)} == {
            tuple(v) for v in r.polytope.vertices()
        }
        assert np.array_equal(r.polytope.vertices(), b.polytope(i).vertices())
        assert r.vertex_count == int(b.vertex_count[i])
        assert r.h11 == int(b.h11[i])
        assert r.h12 == int(b.h12[i])
        assert r.euler_characteristic == int(b.euler_characteristic[i])
        assert r.polytope.vertices().tolist() == b.record(i).polytope.vertices().tolist()


def test_ks_ids_are_stable_and_content_derived():
    """Same row -> same id across independent scans, regardless of batching."""
    a = _batches(n_vertices=[13], n=30, batch_size=30)
    b = _batches(n_vertices=[13], n=30, batch_size=7)

    ids_a = [i for batch in a for i in batch.ks_ids]
    ids_b = [i for batch in b for i in batch.ks_ids]
    assert ids_a == ids_b, "ids depend on batch size"

    # and the id really is a function of the vertices
    verts_a = [v.tobytes() for batch in a for v in batch.iter_vertices()]
    by_id = dict(zip(ids_a, verts_a))
    assert len(by_id) == len(set(verts_a)), "distinct geometries share an id"


def test_batch_take_repacks_correctly():
    batch = _batches(n_vertices=[13, 14], n=20, batch_size=64)[0]
    picked = [3, 0, 7, 1]
    sub = batch.take(picked)

    assert len(sub) == len(picked)
    assert sub.vertex_offsets[-1] == len(sub.vertex_values)
    for j, i in enumerate(picked):
        assert np.array_equal(sub.vertices(j), batch.vertices(i))
        assert sub.ks_ids[j] == batch.ks_ids[i]
        assert sub.h11[j] == batch.h11[i]


def test_scan_batches_respects_batch_size():
    batches = _batches(n_vertices=[13, 14, 15], n=50, batch_size=20)
    assert [len(b) for b in batches] == [20, 20, 10]
