"""Tests for the Parquet-backed KS dataset loader.

The suite configures the committed Kreuzer--Skarke slice in ``conftest.py``,
so these tests always exercise real Parquet data without network access or a
machine-specific database installation.
"""

import numpy as np
import pytest

import cytools.dataset as ds

SINGLE_VERTEX_COUNT = 7
PAIR_VERTEX_COUNTS = [7, 8]
MULTI_VERTEX_COUNTS = [7, 8, 9]


def _load(**kwargs):
    return ds.load_polytopes(**kwargs)


def _safe_extract(table):
    """Reference implementation: per-row Python extraction, no fast path."""
    col = table.column("vertices").combine_chunks()
    return [np.array(col[i].as_py(), dtype=np.int32) for i in range(len(table))]


def test_bounded_cache_evicts_the_least_recently_used_entry():
    cache = ds._BoundedCache(2)
    cache["a"] = 1
    cache["b"] = 2
    assert cache.get_lru("a") == 1
    cache["c"] = 3
    assert list(cache) == ["a", "c"]
    assert cache.get_lru("b") is None


def test_filtered_load_table_honors_the_row_cap(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"value": list(range(20))}), path, row_group_size=5)

    table = ds._load_table(
        path,
        [[("value", "in", [1, 3, 5, 7, 9, 11])]],
        n=3,
        rng=np.random.default_rng(7),
        columns=["value"],
    )

    assert table.num_rows == 3
    assert set(table.column("value").to_pylist()) <= {1, 3, 5, 7, 9, 11}


def test_batch_exposes_all_zero_construction_database_counts():
    """Landscape columns must reach the batch without constructing Polytope."""
    import pyarrow as pa

    table = pa.table(
        {
            "vertices": pa.array(
                [
                    [[1, 0, 0, 0], [0, 1, 0, 0]],
                    [[0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]],
                ],
                type=pa.list_(pa.list_(pa.int32())),
            ),
            "vertex_count": pa.array([2, 3]),
            "facet_count": pa.array([5, 8]),
            "point_count": pa.array([6, 14]),
            "dual_point_count": pa.array([126, 22]),
            "h11": pa.array([101, 11]),
            "h12": pa.array([3, 9]),
            "euler_characteristic": pa.array([196, 4]),
        }
    )

    batch = ds._table_to_batch(table)
    assert batch.vertex_count.tolist() == [2, 3]
    assert batch.facet_count.tolist() == [5, 8]
    assert batch.point_count.tolist() == [6, 14]
    assert batch.dual_point_count.tolist() == [126, 22]


@pytest.mark.parametrize(
    "n_vertices",
    [[7], [7, 8], [7, 8, 9], list(range(5, 10))],
    ids=["single", "two-files", "three-files", "five-files"],
)
def test_vertex_extraction_matches_reference(n_vertices, monkeypatch):
    """The zero-copy fast path must agree with per-row extraction.

    Regression test: the fast path used to validate stride uniformity by
    comparing only the first and last row, so a query spanning several
    vertex-count files could mis-slice vertices into the wrong rows and
    silently yield a corrupted (e.g. non-reflexive) polytope.
    """
    seen = []
    orig = ds._table_to_batch

    def spy(table):
        # `_table_to_batch` rather than `_extract_vertices_from_table`: the
        # latter is only the ragged fallback inside `_batch_buffers`, so a
        # uniform-stride query never reaches it and the comparison below would
        # silently never run.
        fast = orig(table)
        ref = _safe_extract(table)
        seen.append(sorted({len(r) for r in ref}))
        assert len(fast) == len(ref)
        for i, b in enumerate(ref):
            assert np.array_equal(fast.vertices(i), b), f"row {i} mismatch"
        return fast

    monkeypatch.setattr(ds, "_table_to_batch", spy)
    records = _load(n_vertices=n_vertices, n=16)

    assert records, "expected at least one record"
    assert seen, "extraction was never exercised"
    if len(n_vertices) > 1:
        observed = {count for batch_counts in seen for count in batch_counts}
        assert len(observed) > 1, "multiple vertex-count files were not exercised"


def test_records_are_wellformed():
    records = _load(n_vertices=MULTI_VERTEX_COUNTS, n=12)
    for r in records:
        p = r.polytope
        assert p.vertices().shape == (r.vertex_count, 4)
        assert p.dim() == 4
        assert p.is_reflexive(), "corrupted vertices produce non-reflexive polytopes"


def test_hodge_columns_use_m_lattice_convention():
    """The DB's h11/h12/chi are M-lattice, opposite to CYTools' N default.

    Pinned so a future "fix" that swaps the columns cannot land silently.
    """
    records = _load(n_vertices=[SINGLE_VERTEX_COUNT], n=5)
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
    return list(ds.scan_batches(**kwargs))


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

    batches = _batches(n_vertices=PAIR_VERTEX_COUNTS, n=40, batch_size=16)
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
    for b in _batches(n_vertices=MULTI_VERTEX_COUNTS, n=50, batch_size=20):
        assert b.vertex_values.flags["C_CONTIGUOUS"]
        assert b.vertex_offsets[0] == 0
        assert b.vertex_offsets[-1] == len(b.vertex_values)
        assert len(b.vertex_offsets) == len(b) + 1
        for i in range(len(b)):
            v = b.vertices(i)
            assert v.shape == (int(b.vertex_count[i]), 4)
            # a view into the flat buffer, not a copy. Checked by shared
            # memory rather than identity of .base: vertex_values is itself a
            # reshaped view of Arrow's child buffer, so a row's base is that
            # buffer, not vertex_values.
            assert np.shares_memory(v, b.vertex_values)


def test_batch_agrees_with_load_polytopes():
    records = _load(n_vertices=[SINGLE_VERTEX_COUNT], n=12)
    batches = _batches(n_vertices=[SINGLE_VERTEX_COUNT], n=12, batch_size=64)
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
        assert (
            r.polytope.vertices().tolist() == b.record(i).polytope.vertices().tolist()
        )


def test_ks_ids_are_stable_and_content_derived():
    """Same row -> same id across independent scans, regardless of batching."""
    a = _batches(n_vertices=[SINGLE_VERTEX_COUNT], n=30, batch_size=30)
    b = _batches(n_vertices=[SINGLE_VERTEX_COUNT], n=30, batch_size=7)

    ids_a = [i for batch in a for i in batch.ks_ids]
    ids_b = [i for batch in b for i in batch.ks_ids]
    assert ids_a == ids_b, "ids depend on batch size"

    # and the id really is a function of the vertices
    verts_a = [v.tobytes() for batch in a for v in batch.iter_vertices()]
    by_id = dict(zip(ids_a, verts_a))
    assert len(by_id) == len(set(verts_a)), "distinct geometries share an id"


def test_batch_take_repacks_correctly():
    batch = _batches(n_vertices=[SINGLE_VERTEX_COUNT], n=20, batch_size=64)[0]
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
        batches = _batches(n_vertices=MULTI_VERTEX_COUNTS, n=50, batch_size=batch_size)
        sizes = [len(b) for b in batches]
        assert all(0 < s <= batch_size for s in sizes), sizes
        assert sum(sizes) == 50


def test_capped_scan_is_spread_across_vertex_counts():
    """A small n must not be served entirely out of the first file.

    Regression test. The batched scan yields one batch per row group, and an
    early version let the first file's batch absorb the whole budget, so
    a three-file query returned every row from its first shard. Every
    benchmark fixture that samples a vertex-count range would have been
    silently skewed to its lowest member.
    """
    from collections import Counter

    counts = Counter()
    rows = 0
    for batch in _batches(n_vertices=MULTI_VERTEX_COUNTS, n=30, batch_size=8):
        counts.update(batch.vertex_count.tolist())
        rows += len(batch)

    assert rows == 30
    assert set(counts) == set(MULTI_VERTEX_COUNTS), f"only got {dict(counts)}"

    # Apportioned in proportion to the committed shards, not evenly. These
    # counts are pinned by the fixture generator and make the test identical
    # on a laptop and in CI.
    populations = {7: 1_622, 8: 2_767, 9: 1_556}
    total = sum(populations.values())
    for vc, pop in populations.items():
        expected = 30 * pop / total
        assert abs(counts[vc] - expected) <= 1.5, (
            f"{vc}v got {counts[vc]}, expected about {expected:.1f}"
        )


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
        for batch in ds.scan_batches(
            n_vertices=PAIR_VERTEX_COUNTS, n=n, batch_size=512
        ):
            high = max(high, pa.total_allocated_bytes())
            del batch
        return high

    small = peak_pool(100)
    large = peak_pool(5_000)

    assert large < small * 4, (
        f"Arrow allocation grew from {small / 1e6:.0f}MB to {large / 1e6:.0f}MB "
        "for 50x the rows; the scan is not streaming"
    )


def test_ks_ids_uniform_and_ragged_paths_agree():
    """The vectorised hash must not depend on how rows are batched.

    ks_ids are computed column-wise over a uniform-width block when a batch
    comes from one vertex-count file, and grouped by width when it is ragged.
    Both must give a row the same id, or a derived-results store stops joining
    the moment a query spans files.
    """
    single = _batches(n_vertices=[SINGLE_VERTEX_COUNT], n=60, batch_size=60)
    assert single, "expected a batch"
    batch = single[0]

    blocks = [batch.vertices(i) for i in range(len(batch))]
    expected = [int(x) for x in batch.ks_ids]

    # append a row of a different width to force the ragged path
    other = _batches(n_vertices=[PAIR_VERTEX_COUNTS[1]], n=1, batch_size=1)[0]
    blocks.append(other.vertices(0))

    counts = np.array([len(b) for b in blocks], dtype=np.int64)
    offsets = np.zeros(len(blocks) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    ragged = ds._ks_ids(np.concatenate(blocks).astype(np.int32), offsets)

    assert [int(x) for x in ragged[: len(expected)]] == expected


def test_ks_ids_do_not_collide_over_a_large_sample():
    """Distinct geometries must get distinct ids in practice."""
    seen = {}
    collisions = 0
    rows = 0
    for batch in _batches(n_vertices=MULTI_VERTEX_COUNTS, n=20_000, batch_size=8192):
        for i in range(len(batch)):
            rows += 1
            key = int(batch.ks_ids[i])
            data = batch.vertices(i).tobytes()
            if seen.setdefault(key, data) != data:
                collisions += 1

    assert rows, "no rows scanned"
    assert collisions == 0, f"{collisions} distinct geometries shared an id"


def test_row_set_is_invariant_to_batch_size():
    """A capped scan must select the same rows regardless of batch_size.

    The order is not stable -- the round-robin across vertex-count files
    interleaves differently -- but the set of selected rows must not move, or
    the same query returns different data run to run.
    """

    def id_set(batch_size):
        return {
            int(i)
            for b in _batches(
                n_vertices=MULTI_VERTEX_COUNTS, n=600, batch_size=batch_size
            )
            for i in b.ks_ids
        }

    small, medium, large = id_set(64), id_set(301), id_set(4096)
    assert small == medium == large
    assert len(small) == 600


def test_batch_values_survive_the_arrow_table():
    """The zero-copy buffer must keep the Arrow data alive.

    vertex_values is a reshaped view of Arrow's child buffer rather than a
    copy, so the batch has to hold that buffer alive after the table it came
    from is gone.
    """
    import gc

    batch = _batches(n_vertices=[SINGLE_VERTEX_COUNT], n=200, batch_size=200)[0]
    snapshot = np.array(batch.vertices(0), copy=True)
    for _ in range(3):
        gc.collect()
    assert np.array_equal(batch.vertices(0), snapshot)


def test_capped_scans_are_nested_in_n():
    """Growing n must add rows, not resample.

    A capped scan takes a prefix of each file's seeded row-group ordering, so a
    larger n is a superset of a smaller one. This is what makes growing a sweep
    cheap against a derived store: the previously computed rows are all still
    present and get reused, rather than a fresh random subset being drawn. The
    earlier read-everything-then-sample implementation did not have this
    property.
    """

    def id_set(n):
        return {
            int(i)
            for b in _batches(n_vertices=PAIR_VERTEX_COUNTS, n=n, batch_size=32)
            for i in b.ks_ids
        }

    small, medium, large = id_set(60), id_set(120), id_set(400)
    assert len(small) == 60 and len(medium) == 120 and len(large) == 400
    assert small <= medium <= large


def test_capped_scan_reads_only_the_files_it_budgets(monkeypatch):
    """A two-row sample must not open every requested file.

    Files are apportioned a share of `n`, and a file allotted nothing should
    never be read. This used to be about not *downloading* files in streaming
    mode; streaming is gone, but the property still matters -- opening a row
    group is the expensive part of a local scan too.
    """
    import pyarrow as pa

    read = []

    def resolve(vc, resolved_dir):
        return vc

    def one_file(path, dnf, expr, rng, batch_size):
        read.append(path)
        yield pa.record_batch({"marker": pa.array([path], type=pa.int64())})

    monkeypatch.setattr(ds, "_resolve_4d_path", resolve)
    monkeypatch.setattr(ds, "_stream_one_file", one_file)

    batches = list(
        ds._iter_record_batches(
            counts=[5, 6, 7, 8],
            h11=None,
            h12=None,
            chi=None,
            n_facets=None,
            n_points=None,
            n_dual_points=None,
            n=2,
            seed=42,
            resolved_dir="/nonexistent",
            batch_size=8,
        )
    )

    assert sum(batch.num_rows for batch in batches) == 2
    assert len(read) <= 2, f"read {read} for a two-row sample"


def test_empty_batch_scan_needs_no_database_configuration(monkeypatch):
    monkeypatch.delenv("CYTOOLS_DB_DIR", raising=False)
    monkeypatch.setattr(ds, "DB_DIR", None)
    assert list(ds.scan_batches(n=0)) == []
    assert ds.load_polytopes(n=0) == []


# ---------------------------------------------------------------------------
# A capped scan must sample in proportion to what each file holds
# ---------------------------------------------------------------------------


def test_apportion_is_proportional_and_exact():
    """Splitting n evenly across files skews the sample; weight it instead.

    Measured at h11(N)=200, where the population is 6v:5 ... 10v:174 ... 15v:5,
    an even split exhausted the small files (3.5x over-represented) while 10v
    returned 23 rows where 49 were due (0.47x).
    """
    from cytools.dataset import _apportion

    weights = [5, 38, 82, 107, 174, 132, 98, 51, 14, 5]
    budgets = _apportion(350, weights)

    assert sum(budgets) == 350, "apportionment must not lose or invent rows"
    total = sum(weights)
    for w, b in zip(weights, budgets):
        expected = 350 * w / total
        assert abs(b - expected) < 1.0, f"{b} vs {expected:.2f} for weight {w}"


def test_apportion_falls_back_when_weights_are_unknown():
    from cytools.dataset import _apportion

    budgets = _apportion(10, [0, 0, 0])
    assert sum(budgets) == 10
    assert max(budgets) - min(budgets) <= 1


def test_apportion_handles_n_smaller_than_file_count():
    from cytools.dataset import _apportion

    budgets = _apportion(3, [100, 50, 10, 1])
    assert sum(budgets) == 3
    # the largest population must be served first
    assert budgets[0] >= budgets[-1]


def test_capped_scan_matches_the_population_shape():
    """End-to-end: the returned vertex-count mix tracks the real population."""
    import collections

    truth = {5: 14, 6: 220, 7: 939, 8: 2_261, 9: 1_556}
    got = collections.Counter()
    for b in ds.scan_batches(h12=5, n=350, batch_size=64):
        for i in range(len(b)):
            got[len(b.vertices(i))] += 1

    assert got, "the committed slice must contain h11(N)=5 polytopes"

    drawn = sum(got.values())
    total = sum(truth.values())
    # Only check files with a big enough expectation that rounding is not the
    # dominant effect.
    for nv, pop in truth.items():
        expected = pop / total * drawn
        if expected < 5:
            continue
        ratio = got.get(nv, 0) / expected
        assert 0.8 < ratio < 1.25, f"{nv}v drawn at {ratio:.2f}x its share"
