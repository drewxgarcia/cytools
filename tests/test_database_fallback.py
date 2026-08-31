"""
Tests for the HuggingFace fallback in `cytools.dataset`.

The policy under test: a shard present locally is used and never fetched;
only a missing one falls back to the remote repository. None of these tests
touch the network -- `_hf_download` is replaced, and any call to the real one
would be a failure.
"""

import pathlib

import pytest

from cytools import dataset as ds


@pytest.fixture
def local_db():
    """The configured local database, or a skip when there isn't one."""
    directory = ds._optional_dir(None, ds.DB_DIR, "CYTOOLS_DB_DIR")
    if directory is None or not ds._all_vertex_counts(directory):
        pytest.skip("no local KS database configured")
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


# --------------------------------------------------------------- fallback fires
def test_a_missing_shard_falls_back_to_huggingface(tmp_path, monkeypatch):
    calls = []

    def fake_download(repo_id, filename, token=None):
        calls.append((repo_id, filename))
        return pathlib.Path("/nonexistent/fetched.parquet")

    monkeypatch.setattr(ds, "_hf_download", fake_download)
    ds._resolve_4d_path(7, tmp_path)  # empty directory: nothing present locally
    assert calls == [(ds._HF_4D_REPO, "polytopes-4d-07-vertices.parquet")]


def test_fallback_fires_when_no_directory_is_configured(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ds, "_hf_download", lambda r, f, token=None: calls.append((r, f))
    )
    ds._resolve_4d_path(12, None)
    assert calls == [(ds._HF_4D_REPO, "polytopes-4d-12-vertices.parquet")]


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
