"""Stored numbers must say what produced them.

`DerivedStore` keys on a content hash, so two generations of results for the
same geometry collide by design and `read` concatenates them. These tests pin
the property that makes that safe: a mismatch in the conditions that computed
the rows is reported at the moment the rows are combined.
"""

import json
import warnings

import pyarrow.parquet as pq
import pytest

from cytools import provenance
from cytools.store import DerivedStore


@pytest.fixture
def store(tmp_path):
    return DerivedStore(tmp_path)


@pytest.fixture
def other_generation(monkeypatch):
    """Make this process claim it computed under different conditions.

    Patching the cache rather than the function keeps `with_fingerprint` and
    everything downstream on the real code path, so what is exercised is the
    store's handling of a difference rather than a stubbed-out stamp.
    """

    def apply(**overrides):
        record = provenance.fingerprint()
        record.update(overrides)
        record["digest"] = "0" * 16
        monkeypatch.setattr(
            provenance,
            "_CACHED",
            json.dumps(record, sort_keys=True, separators=(",", ":")),
        )

    return apply


def strip_stamp(path):
    """Rewrite a part file with no provenance, as a pre-stamping run left it."""
    table = pq.read_table(path)
    pq.write_table(table.replace_schema_metadata({}), path)


# the record
# ----------
def test_the_fingerprint_describes_what_can_change_a_number():
    record = provenance.fingerprint()

    assert set(record) == {"cytools", "worktree", "engines", "packages", "digest"}
    assert record["engines"]["stretched_tip"][0] == "highs"
    assert record["packages"]["numpy"]
    assert len(record["digest"]) == 16


def test_the_digest_is_stable_within_a_process_and_the_record_is_not_shared():
    first, second = provenance.fingerprint(), provenance.fingerprint()

    assert first == second
    assert first is not second, "callers must be free to annotate their own copy"

    first["engines"] = {}
    assert provenance.fingerprint()["engines"], "mutation must not leak into the cache"


def test_differences_name_fields_rather_than_digests():
    left = provenance.fingerprint()
    right = provenance.fingerprint()
    right["engines"] = {**right["engines"], "stretched_tip": ["osqp"]}
    right["digest"] = "different"

    # The digest is a summary of the other fields, so reporting it as its own
    # difference would be noise on top of the real answer.
    assert provenance.differences(left, right) == ["engines.stretched_tip"]
    assert provenance.differences(left, left) == []
    assert provenance.differences(left, None) == ["<unstamped>"]
    assert provenance.differences(None, None) == []


# the store
# ---------
def test_a_written_part_carries_its_provenance(store):
    store.write("hodge", [1, 2], [{"h11": 3}, {"h11": 4}])

    (entry,) = store.provenance("hodge")
    assert entry["rows"] == 2
    assert entry["provenance"]["digest"] == provenance.fingerprint()["digest"]


def test_reading_one_generation_is_silent(store):
    store.write("hodge", [1], [{"h11": 3}])
    store.write("hodge", [2], [{"h11": 4}])

    with warnings.catch_warnings():
        warnings.simplefilter("error", provenance.ProvenanceWarning)
        assert store.read("hodge").num_rows == 2


def test_reading_across_generations_names_what_changed(store, other_generation):
    store.write("hodge", [1], [{"h11": 3}])
    other_generation(engines={"stretched_tip": ["osqp"]})
    store.write("hodge", [2], [{"h11": 4}])

    with pytest.warns(provenance.ProvenanceWarning, match="engines.stretched_tip"):
        table = store.read("hodge")

    # The warning must not cost the caller their data.
    assert table.num_rows == 2


def test_an_upgraded_dependency_is_noticed_too(store, other_generation):
    """Not only engine swaps: a solver version change moves numbers as well."""
    store.write("hodge", [1], [{"h11": 3}])
    other_generation(
        packages={**provenance.fingerprint()["packages"], "highspy": "0.0"}
    )
    store.write("hodge", [2], [{"h11": 4}])

    with pytest.warns(provenance.ProvenanceWarning, match="packages.highspy"):
        store.read("hodge")


def test_a_study_can_refuse_to_mix_generations(store, other_generation):
    """The warning has its own class so it can be promoted to an error."""
    store.write("hodge", [1], [{"h11": 3}])
    other_generation(cytools="0.0.0")
    store.write("hodge", [2], [{"h11": 4}])

    with warnings.catch_warnings():
        warnings.simplefilter("error", provenance.ProvenanceWarning)
        with pytest.raises(provenance.ProvenanceWarning):
            store.read("hodge")


# stores written before stamping existed
# --------------------------------------
def test_an_unstamped_store_is_still_readable_and_still_silent(store):
    store.write("hodge", [1], [{"h11": 3}])
    store.write("hodge", [2], [{"h11": 4}])
    for entry in store.provenance("hodge"):
        strip_stamp(store.quantity_dir("hodge") / entry["part"])

    with warnings.catch_warnings():
        warnings.simplefilter("error", provenance.ProvenanceWarning)
        assert store.read("hodge").num_rows == 2
    assert [e["provenance"] for e in store.provenance("hodge")] == [None, None]


def test_appending_to_an_unstamped_store_is_reported(store):
    """The realistic case: an existing store gets new rows after an upgrade."""
    store.write("hodge", [1], [{"h11": 3}])
    strip_stamp(store.quantity_dir("hodge") / store.provenance("hodge")[0]["part"])
    store.write("hodge", [2], [{"h11": 4}])

    with pytest.warns(provenance.ProvenanceWarning, match="unstamped"):
        assert store.read("hodge").num_rows == 2


# compaction
# ----------
def test_compaction_carries_provenance_forward(store):
    store.write("hodge", [1], [{"h11": 3}])
    store.write("hodge", [2], [{"h11": 4}])
    store.compact("hodge")

    entries = store.provenance("hodge")
    assert len(entries) == 1
    assert entries[0]["provenance"]["digest"] == provenance.fingerprint()["digest"]


def test_compaction_across_generations_keeps_both_sources(store, other_generation):
    """A compactor computes nothing, so it must not sign the result as its own."""
    store.write("hodge", [1], [{"h11": 3}])
    other_generation(engines={"stretched_tip": ["osqp"]})
    store.write("hodge", [2], [{"h11": 4}])

    with pytest.warns(provenance.ProvenanceWarning):
        store.compact("hodge")

    (entry,) = store.provenance("hodge")
    record = entry["provenance"]
    assert len(record["sources"]) == 2
    assert record["digest"] != provenance.fingerprint()["digest"]
    engines = [source["engines"]["stretched_tip"] for source in record["sources"]]
    assert ["osqp"] in engines


def test_combined_provenance_of_agreeing_sources_is_just_that_provenance():
    record = provenance.fingerprint()

    assert provenance.combined_fingerprint([record, dict(record)]) == record


def test_combined_provenance_counts_every_unstamped_source():
    two = provenance.combined_fingerprint([None, None])
    three = provenance.combined_fingerprint([None, None, None])

    assert two["unstamped_sources"] == 2
    assert three["unstamped_sources"] == 3
    assert two["digest"] != three["digest"]
