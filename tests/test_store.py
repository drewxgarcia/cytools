"""Tests for the Parquet-backed derived-results store.

These use synthetic batches rather than the KS database, so they run anywhere.
"""

import numpy as np
import pyarrow.parquet as pq
import pytest

from cytools.store import DerivedStore, materialize


# ---------------------------------------------------------------------------
# A minimal stand-in for PolytopeBatch: the store only needs ks_ids +
# vertices(i), so testing against that contract keeps these tests fast and
# independent of the database.
# ---------------------------------------------------------------------------


class FakeBatch:
    def __init__(self, ks_ids, blocks):
        self.ks_ids = np.asarray(ks_ids, dtype=np.int64)
        self._blocks = blocks

    def __len__(self):
        return len(self.ks_ids)

    def vertices(self, i):
        return self._blocks[i]


def make_scan(ids, batch_size=4):
    ids = list(ids)
    blocks = {i: np.full((3, 4), i, dtype=np.int32) for i in ids}
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        yield FakeBatch(chunk, [blocks[i] for i in chunk])


CALLS = []


def payload(vertices):
    """Module-level so it is picklable for the parallel test."""
    CALLS.append(1)
    return {"value": int(vertices[0][0]) * 2, "n_verts": len(vertices)}


def failing_payload(vertices):
    v = int(vertices[0][0])
    if v % 3 == 0:
        raise ValueError(f"synthetic failure for {v}")
    return {"value": v}


@pytest.fixture
def store(tmp_path):
    return DerivedStore(tmp_path / "derived")


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS.clear()


def test_cold_run_computes_everything(store):
    summary = materialize("q", payload, store=store, scan=make_scan(range(10)))
    assert summary == {"requested": 10, "computed": 10, "skipped": 0, "failed": 0}
    assert len(CALLS) == 10
    assert len(store.known_ids("q")) == 10


def test_second_run_is_a_noop(store):
    materialize("q", payload, store=store, scan=make_scan(range(10)))
    CALLS.clear()

    summary = materialize("q", payload, store=store, scan=make_scan(range(10)))
    assert summary == {"requested": 10, "computed": 0, "skipped": 10, "failed": 0}
    assert not CALLS, "a completed quantity must not recompute anything"


def test_partial_run_resumes(store):
    """The property the store exists for: interrupt, rerun, finish the rest."""
    materialize("q", payload, store=store, scan=make_scan(range(6)))
    CALLS.clear()

    summary = materialize("q", payload, store=store, scan=make_scan(range(10)))
    assert summary == {"requested": 10, "computed": 4, "skipped": 6, "failed": 0}
    assert len(CALLS) == 4
    assert len(store.known_ids("q")) == 10


def test_recompute_ignores_what_is_stored(store):
    materialize("q", payload, store=store, scan=make_scan(range(5)))
    CALLS.clear()

    summary = materialize(
        "q", payload, store=store, scan=make_scan(range(5)), recompute=True
    )
    assert summary["computed"] == 5
    assert summary["skipped"] == 0
    assert len(CALLS) == 5
    # still one row per id after the duplicate write
    assert store.read("q").num_rows == 5


def test_versions_are_independent(store):
    materialize("q", payload, store=store, scan=make_scan(range(5)), version=1)
    CALLS.clear()

    summary = materialize(
        "q", payload, store=store, scan=make_scan(range(5)), version=2
    )
    assert summary["computed"] == 5, "a new version must recompute"
    assert len(store.known_ids("q", 1)) == 5
    assert len(store.known_ids("q", 2)) == 5
    assert store.versions("q") == [1, 2]


def test_results_round_trip(store):
    materialize("q", payload, store=store, scan=make_scan(range(4)))
    table = store.read("q")

    assert set(table.column_names) >= {"ks_id", "value", "n_verts"}
    rows = {
        r["ks_id"]: r for r in table.to_pylist()
    }
    for i in range(4):
        assert rows[i]["value"] == i * 2
        assert rows[i]["n_verts"] == 3


def test_read_can_be_restricted_to_ids(store):
    materialize("q", payload, store=store, scan=make_scan(range(10)))
    table = store.read("q", ks_ids=[2, 5, 7])
    assert sorted(table.column("ks_id").to_pylist()) == [2, 5, 7]


def test_missing_reports_only_absent_ids(store):
    materialize("q", payload, store=store, scan=make_scan(range(5)))
    missing = store.missing("q", [0, 1, 99, 100])
    assert missing.tolist() == [99, 100]


def test_failures_are_recorded_and_not_retried(store):
    summary = materialize(
        "q", failing_payload, store=store, scan=make_scan(range(9))
    )
    # 0, 3, 6 raise
    assert summary["failed"] == 3
    assert summary["computed"] == 6

    again = materialize(
        "q", failing_payload, store=store, scan=make_scan(range(9))
    )
    assert again["skipped"] == 9, "recorded failures should not be retried"

    table = store.read("q")
    errors = [r for r in table.to_pylist() if r.get("error")]
    assert len(errors) == 3
    assert all("synthetic failure" in r["error"] for r in errors)


def test_failures_can_be_left_for_retry(store):
    materialize(
        "q",
        failing_payload,
        store=store,
        scan=make_scan(range(9)),
        store_errors=False,
    )
    assert len(store.known_ids("q")) == 6

    missing = store.missing("q", list(range(9)))
    assert missing.tolist() == [0, 3, 6]


def test_compact_merges_parts_without_changing_contents(store):
    materialize("q", payload, store=store, scan=make_scan(range(12), batch_size=3))
    before = store.read("q")
    assert store.stats("q")["n_parts"] > 1

    merged = store.compact("q")
    assert merged is not None
    assert store.stats("q")["n_parts"] == 1

    after = store.read("q")
    assert after.num_rows == before.num_rows
    assert sorted(after.column("ks_id").to_pylist()) == sorted(
        before.column("ks_id").to_pylist()
    )
    assert store.compact("q") is None, "compacting once is enough"


def test_unreadable_part_does_not_break_reads(store, tmp_path):
    """A killed write must not make the whole quantity unreadable."""
    materialize("q", payload, store=store, scan=make_scan(range(5)))
    junk = store.quantity_dir("q") / "part-deadbeef.parquet"
    junk.write_bytes(b"not parquet")

    assert len(store.known_ids("q")) == 5
    assert store.read("q").num_rows == 5


def test_write_rejects_mismatched_lengths(store):
    with pytest.raises(ValueError, match="ks_ids"):
        store.write("q", [1, 2], [{"a": 1}])


def test_invalid_quantity_names_rejected(store):
    for bad in ("", "a/b", ".hidden"):
        with pytest.raises(ValueError):
            store.quantity_dir(bad)


def test_empty_store_reads_empty(store):
    assert store.quantities() == []
    assert store.versions("nope") == []
    assert len(store.known_ids("nope")) == 0
    assert store.read("nope").num_rows == 0


def test_parallel_workers_produce_the_same_results(store, tmp_path):
    serial = DerivedStore(tmp_path / "serial")
    parallel = DerivedStore(tmp_path / "parallel")

    materialize("q", payload, store=serial, scan=make_scan(range(8)))
    materialize("q", payload, store=parallel, scan=make_scan(range(8)), workers=2)

    a = {r["ks_id"]: r["value"] for r in serial.read("q").to_pylist()}
    b = {r["ks_id"]: r["value"] for r in parallel.read("q").to_pylist()}
    assert a == b


def test_progress_callback_is_called_per_batch(store):
    seen = []
    materialize(
        "q",
        payload,
        store=store,
        scan=make_scan(range(10), batch_size=4),
        on_progress=seen.append,
    )
    assert len(seen) == 3  # 4 + 4 + 2
    assert seen[-1]["computed"] == 10
    assert [s["requested"] for s in seen] == [4, 8, 10]


def test_unpicklable_payload_fails_clearly(store):
    """A lambda must produce an actionable error, not BrokenProcessPool.

    With spawn-based workers the payload is sent by reference, so an
    unpicklable one dies deep inside concurrent.futures with nothing naming the
    cause. Check it early instead.
    """
    with pytest.raises(ValueError, match="cannot be sent to worker processes"):
        materialize(
            "q",
            lambda vertices: {"a": 1},
            store=store,
            scan=make_scan(range(4)),
            workers=2,
        )


def test_unpicklable_payload_is_fine_serially(store):
    """The restriction only applies to parallel runs."""
    summary = materialize(
        "q",
        lambda vertices: {"a": int(vertices[0][0])},
        store=store,
        scan=make_scan(range(4)),
        workers=1,
    )
    assert summary["computed"] == 4


def test_non_dict_payload_result_is_rejected(store):
    with pytest.raises(TypeError, match="must return a dict"):
        materialize("q", lambda v: 42, store=store, scan=make_scan(range(2)))
