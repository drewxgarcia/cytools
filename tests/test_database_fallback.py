"""
Tests for how `cytools.dataset` obtains 4D database shards.

The policy under test: a shard present locally is used and never fetched, and a
shard that is *missing* is reported rather than downloaded. Reads never start a
transfer on their own -- the database is ~16 GB and a single vertex-count shard
~2 GB -- so they raise `DatabaseUnavailable` and name `download_shards`, which
is the one explicit way to fetch.

None of these tests touch the network. `huggingface_hub`'s transfer functions
are blocked suite-wide (see conftest), and the tests below additionally replace
`_hf_download` so they can observe the *decision* without a transfer.
"""

import pathlib

import pytest

from cytools import dataset as ds


@pytest.fixture
def local_db():
    """The configured committed database slice."""
    directory = ds._optional_dir(None, ds.DB_DIR, "CYTOOLS_DB_DIR")
    assert directory is not None, "conftest must configure the committed database slice"
    assert ds._all_vertex_counts(directory), "the committed database slice is empty"
    return directory


# ------------------------------------------------------------ local is preferred
def test_a_local_shard_is_never_downloaded(local_db, monkeypatch):
    """The whole point of the fallback: it must not fire when it is not needed."""

    def explode(*args, **kwargs):
        raise AssertionError("a locally present shard was fetched from HuggingFace")

    monkeypatch.setattr(ds, "_hf_download", explode)
    records = ds.load_polytopes(n_vertices=5, n=3, db_dir=local_db)
    assert records


def test_available_counts_prefer_a_populated_local_directory(local_db, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("consulted the remote repo despite a local database")

    monkeypatch.setattr(ds, "_hf_4d_vertex_counts", explode)
    counts = ds._available_4d_vertex_counts(local_db)
    assert counts == ds._all_vertex_counts(local_db)


def test_resolve_path_returns_the_local_file_when_present(local_db, monkeypatch):
    monkeypatch.setattr(
        ds, "_hf_download", lambda *a, **k: pytest.fail("should not download")
    )
    path = ds._resolve_4d_path(5, local_db)
    assert path == ds._db_path(5, local_db)
    assert path.exists()


# ------------------------------------------------- a missing shard is reported
def test_a_missing_shard_is_reported_not_downloaded(tmp_path, monkeypatch):
    """The core of the policy: reading does not fetch ~2 GB on its own."""

    def explode(*args, **kwargs):
        raise AssertionError("a read started a download without being asked")

    monkeypatch.setattr(ds, "_hf_download", explode)
    with pytest.raises(ds.DatabaseUnavailable, match="download_shards"):
        ds._resolve_4d_path(7, tmp_path)  # empty dir: nothing present locally


def test_a_missing_shard_is_reported_when_no_directory_is_configured(monkeypatch):
    monkeypatch.setattr(
        ds, "_hf_download", lambda *a, **k: pytest.fail("unrequested download")
    )
    with pytest.raises(ds.DatabaseUnavailable, match="download_shards"):
        ds._resolve_4d_path(12, None)


def test_the_error_names_every_way_out(tmp_path, monkeypatch):
    """A dead end is only useful if it says what to do next."""
    monkeypatch.setattr(ds, "_hf_download", lambda *a, **k: pytest.fail("no fetch"))
    with pytest.raises(ds.DatabaseUnavailable) as excinfo:
        ds._resolve_4d_path(7, tmp_path)
    message = str(excinfo.value)
    assert "download_shards(7)" in message  # how to fetch just what is needed
    assert "CYTOOLS_DB_DIR" in message  # how to point at an existing copy
    assert "CYTOOLS_ALLOW_DOWNLOADS" in message  # how to opt into implicit fetches
    assert "GB" in message  # how much data is at stake


# ------------------------------------------------ downloads happen when asked
def test_download_shards_fetches_a_missing_shard(tmp_path, monkeypatch):
    calls = []

    def fake_download(repo_id, filename):
        calls.append((repo_id, filename))
        return pathlib.Path("/nonexistent/fetched.parquet")

    monkeypatch.setattr(ds, "_hf_download", fake_download)
    ds.download_shards(7, db_dir=tmp_path, quiet=True)
    assert calls == [(ds._HF_4D_REPO, "polytopes-4d-07-vertices.parquet")]


def test_download_shards_skips_what_is_already_local(local_db, monkeypatch):
    monkeypatch.setattr(
        ds, "_hf_download", lambda *a, **k: pytest.fail("re-fetched a local shard")
    )
    have = ds._all_vertex_counts(local_db)[:2]
    paths = ds.download_shards(have, db_dir=local_db, quiet=True)
    assert paths == [ds._db_path(n, local_db) for n in have]


def test_download_shards_rejects_a_nonexistent_vertex_count(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "_hf_download", lambda *a, **k: pytest.fail("no fetch"))
    with pytest.raises(ValueError, match="5 through 36"):
        ds.download_shards(99, db_dir=tmp_path, quiet=True)


def test_the_env_opt_in_restores_implicit_fetching(tmp_path, monkeypatch):
    """`CYTOOLS_ALLOW_DOWNLOADS` is the documented escape hatch, so it must work."""
    calls = []
    monkeypatch.setattr(ds, "_hf_download", lambda r, f: calls.append((r, f)))
    monkeypatch.setenv("CYTOOLS_ALLOW_DOWNLOADS", "1")
    ds._resolve_4d_path(9, tmp_path)
    assert calls == [(ds._HF_4D_REPO, "polytopes-4d-09-vertices.parquet")]


def test_the_permission_flag_does_not_leak_out_of_download_shards(
    tmp_path, monkeypatch
):
    """A single explicit fetch must not leave later reads free to download."""
    monkeypatch.setattr(ds, "_hf_download", lambda r, f: pathlib.Path("/nonexistent"))
    ds.download_shards(7, db_dir=tmp_path, quiet=True)
    assert not ds._EXPLICIT_DOWNLOAD
    with pytest.raises(ds.DatabaseUnavailable):
        ds._resolve_4d_path(8, tmp_path)


def test_shard_filenames_match_the_local_naming_scheme(local_db):
    """Remote and local names must agree, or the cache would never hit."""
    for n in ds._all_vertex_counts(local_db)[:5]:
        assert ds._hf_4d_filename(n) == ds._db_path(n, local_db).name


# ------------------------------------------------------------------- diagnostics
def test_missing_huggingface_hub_gives_an_actionable_error(monkeypatch):
    """The dependency is optional, so its absence must explain the two ways out."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("huggingface_hub"):
            raise ImportError("simulated: huggingface_hub not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"cytools-workbench\[streaming\]"):
        ds._hf_download(ds._HF_4D_REPO, "polytopes-4d-05-vertices.parquet")


def test_optional_dir_returns_none_instead_of_raising(monkeypatch):
    monkeypatch.delenv("CYTOOLS_DB_DIR", raising=False)
    assert ds._optional_dir(None, None, "CYTOOLS_DB_DIR") is None


def test_optional_dir_prefers_the_explicit_argument(tmp_path):
    assert ds._optional_dir(tmp_path, pathlib.Path("/other"), "CYTOOLS_DB_DIR") == (
        tmp_path
    )
