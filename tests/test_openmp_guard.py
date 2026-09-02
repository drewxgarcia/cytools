"""Tests for duplicate-OpenMP-runtime detection.

The failure this guards against is a bare SIGABRT with no Python traceback,
which is close to undebuggable from the outside. These tests drive the
detection through fake paths rather than by actually loading two runtimes,
since doing that for real would abort the test process.
"""

import sys

import pytest

from cytools._backends import openmp


def test_no_conflict_when_nothing_is_loaded(monkeypatch):
    monkeypatch.setattr(openmp, "loaded_runtimes", set)
    assert openmp.conflicting_runtime("/anywhere/libomp.dylib") is None


def test_no_conflict_against_the_same_file(monkeypatch):
    """The symlink fix works by making both names resolve to one file."""
    monkeypatch.setattr(
        openmp, "loaded_runtimes", lambda: {"/opt/homebrew/lib/libomp.dylib"}
    )
    assert openmp.conflicting_runtime("/opt/homebrew/lib/libomp.dylib") is None


def test_conflict_against_a_different_file(monkeypatch):
    monkeypatch.setattr(
        openmp, "loaded_runtimes", lambda: {"/opt/homebrew/lib/libomp.dylib"}
    )
    got = openmp.conflicting_runtime("/venv/torch/lib/libomp.dylib")
    assert got == "/opt/homebrew/lib/libomp.dylib"


def test_no_candidate_is_not_a_conflict(monkeypatch):
    """torch absent -> nothing to collide with."""
    monkeypatch.setattr(openmp, "loaded_runtimes", lambda: {"/a/libomp.dylib"})
    assert openmp.conflicting_runtime(None) is None


def test_conflict_error_names_both_runtimes_and_the_fix(monkeypatch):
    monkeypatch.setattr(openmp, "_bundled_torch_runtime", lambda: "/venv/libomp.dylib")
    monkeypatch.setattr(openmp, "conflicting_runtime", lambda _: "/brew/libomp.dylib")

    with pytest.raises(openmp.OpenMPRuntimeConflict) as exc_info:
        openmp.ensure_compatible()

    msg = str(exc_info.value)
    assert "/brew/libomp.dylib" in msg and "/venv/libomp.dylib" in msg
    assert "ln -sf" in msg
    # the unsafe workaround must be named as unsafe, not recommended
    assert "KMP_DUPLICATE_LIB_OK" in msg and "not safe" in msg


def test_compatibility_check_is_silent_when_clean(monkeypatch):
    monkeypatch.setattr(openmp, "_bundled_torch_runtime", lambda: "/venv/libomp.dylib")
    monkeypatch.setattr(openmp, "conflicting_runtime", lambda _: None)
    assert openmp.ensure_compatible() is None


def test_loaded_runtimes_reports_real_paths():
    """Whatever it reports must be absolute and look like an OpenMP runtime."""
    if sys.platform != "darwin":
        assert openmp.loaded_runtimes() == set()
        return

    for path in openmp.loaded_runtimes():
        assert path.startswith("/")
        assert "libomp" in path.rsplit("/", 1)[-1]


def test_loaded_runtimes_is_empty_off_darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert openmp.loaded_runtimes() == set()


# ------------------------------------------------------- upgrade-proof linking
def test_the_suggested_link_target_survives_a_homebrew_upgrade():
    """The repair the guard prints must not be pinned to a version.

    Regression test. `loaded_runtimes` reports realpaths, so the message named
    `.../Cellar/libomp/22.1.7/...`. A link made through that broke when
    homebrew moved to 23.1.0: the Cellar directory vanished and PyTorch failed
    to load at all, which is worse than the duplicate runtime it was fixing.
    """
    from cytools._backends.openmp import _stable_link_target

    versioned = "/opt/homebrew/Cellar/libomp/22.1.7/lib/libomp.dylib"
    assert _stable_link_target(versioned) == "/opt/homebrew/opt/libomp/lib/libomp.dylib"
    # any version maps to the same stable path
    assert _stable_link_target(
        "/opt/homebrew/Cellar/libomp/23.1.0/lib/libomp.dylib"
    ) == _stable_link_target(versioned)


def test_non_homebrew_paths_are_left_alone():
    from cytools._backends.openmp import _stable_link_target

    for path in ("/usr/lib/libomp.dylib", "/opt/homebrew/Cellar/malformed"):
        assert _stable_link_target(path) == path
