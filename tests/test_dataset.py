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
    """No batch may exceed batch_size, and the totals must be exact.

    The precise sizes are not fixed: a capped scan apportions its budget across
    the requested vertex-count files, so batch shapes follow the per-file share
    rather than a flat slice of one big table.
    """
    for batch_size in (7, 20, 500):
        batches = _batches(n_vertices=[13, 14, 15], n=50, batch_size=batch_size)
        sizes = [len(b) for b in batches]
        assert all(0 < s <= batch_size for s in sizes), sizes
        assert sum(sizes) == 50


def test_capped_scan_is_spread_across_vertex_counts():
    """A small n must not be served entirely out of the first file.

    Regression test. The streaming scan yields one batch per row group, and an
    early version let the first file's batch absorb the whole budget, so
    `n_vertices=[13, 14, 15], n=16` returned sixteen 13-vertex polytopes. Every
    benchmark fixture that samples a vertex-count range would have been
    silently skewed to its lowest member.
    """
    from collections import Counter

    counts = Counter()
    rows = 0
    for batch in _batches(n_vertices=[13, 14, 15], n=30, batch_size=8):
        counts.update(batch.vertex_count.tolist())
        rows += len(batch)

    assert rows == 30
    assert set(counts) == {13, 14, 15}, f"only got {dict(counts)}"
    # even apportioning, within the rounding of n / n_files
    assert max(counts.values()) - min(counts.values()) <= 1, dict(counts)


def test_streaming_memory_does_not_grow_with_n():
    """Memory must be bounded by batch_size, not by how many rows match.

    Regression test. The scan used to read every matching row into one table
    and then slice it, at ~6.9 kB per row held at once -- about 6.9 GB at a
    million rows, so the batch API could not be used at the scale it exists
    for. Compares Arrow's own allocation high-water mark across two scans that
    differ by 50x in row count.
    """
    import pyarrow as pa

    def peak_pool(n):
        high = 0
        for batch in ds.scan_batches(n_vertices=[13, 14], n=n, batch_size=512):
            high = max(high, pa.total_allocated_bytes())
            del batch
        return high

    try:
        small = peak_pool(1000)
        large = peak_pool(50_000)
    except ValueError as e:
        pytest.skip(f"no local KS database configured: {e}")

    assert large < small * 4, (
        f"Arrow allocation grew from {small/1e6:.0f}MB to {large/1e6:.0f}MB "
        "for 50x the rows; the scan is not streaming"
    )
