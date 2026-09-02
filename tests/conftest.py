"""Shared fixtures and suite-wide safety rails.

Three things live here rather than in individual test modules.

The database resolution below gives the suite the same committed 72 KB slice
on every machine. Without it every test touching Kreuzer--Skarke data skipped
in continuous integration -- 55 of them, measured -- so CI proved that the
wheel imports but never exercised a real polytope.

`experimental_features` was duplicated per module; it is a global flag, so
restoring it afterwards matters and is worth stating once.

`_no_network_downloads` is a hard rail: the 4D database runs to ~16 GB, and no
test may ever start fetching it. `cytools.dataset` already refuses to download
unless asked (see `DatabaseUnavailable`), but that is a policy in library code
which a future edit could relax. This makes it structural for the suite: the
download primitive itself is replaced, so an accidental fetch fails loudly
instead of quietly transferring gigabytes on someone's laptop or in CI.
"""

import functools
import importlib.util
import os
import warnings
from pathlib import Path

import pytest

#: Complete for `h11(lattice="N") <= 5`; see `tests/fixtures/build_ks_slice.py`.
COMMITTED_SLICE = Path(__file__).parent / "fixtures" / "ks-slice"

# Tests are deterministic by default even on a workstation with the full
# mirror. A deliberate full-database run remains possible without changing
# library code: set CYTOOLS_TEST_DB_DIR to that mirror before invoking pytest.
os.environ["CYTOOLS_DB_DIR"] = os.environ.get(
    "CYTOOLS_TEST_DB_DIR", str(COMMITTED_SLICE)
)

import cytools


@functools.cache
def _dependency_available(name: str) -> bool:
    """Whether an optional integration can execute safely in this process."""
    try:
        if importlib.util.find_spec(name) is None:
            return False
    except (ImportError, ValueError):
        return False

    if name == "dualgnn":
        # Merely finding dualgnn is insufficient on macOS: importing its
        # PyTorch runtime beside another LLVM OpenMP can abort the interpreter.
        from cytools._backends import openmp

        try:
            openmp.ensure_compatible()
        except openmp.OpenMPRuntimeConflict:
            return False
    return True


def pytest_collection_modifyitems(config, items):
    """Deselect integrations absent from this environment instead of skipping.

    Optional backends have dedicated CI jobs that install their extras. In a
    base-wheel run they are outside the configured test environment, not tests
    that started and then declined to run. Reporting them as deselected keeps
    the distinction visible and makes every collected test executable.
    """
    selected = []
    deselected = []
    for item in items:
        requirements = [
            marker.args[0]
            for marker in item.iter_markers("requires_dependency")
            if marker.args
        ]
        if all(_dependency_available(name) for name in requirements):
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


@pytest.fixture
def experimental_features():
    """
    Temporarily enable the experimental features, restoring the previous state
    afterwards so that the global flag does not leak into other tests.

    Use this only where the experimental gate is genuinely what blocks the
    operation. Several things that look gated are not -- a non-reflexive
    polytope has no GLSM charge matrix at all, for instance, and enabling the
    flag does not change that.
    """
    prev = cytools.config._exp_features_enabled
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cytools.config.enable_experimental_features()
    yield
    cytools.config._exp_features_enabled = prev


@pytest.fixture(autouse=True, scope="session")
def _no_network_downloads():
    """Make a database transfer impossible for the duration of the suite.

    Autouse and session-scoped, so it cannot be forgotten.

    The block is on `huggingface_hub`'s transfer functions, not on CYTools'
    `_hf_download` wrapper. Replacing the wrapper was the first attempt and it
    was too blunt: it also swallowed the wrapper's own behaviour, so the test
    asserting that a missing `huggingface_hub` produces an actionable error hit
    this rail instead of the code path it was checking. Blocking one layer down
    leaves every line of CYTools' own logic reachable while still guaranteeing
    that no byte moves.
    """
    try:
        import huggingface_hub as hub
    except ImportError:
        # No client means no route to the network and therefore nothing to
        # patch. This is the normal base-wheel environment in CI.
        yield
        return

    def refuse(*args, **kwargs):
        target = kwargs.get("filename") or (args[1] if len(args) > 1 else "?")
        raise AssertionError(
            f"a test tried to fetch {target!r} from HuggingFace. The suite must "
            "never transfer database shards -- they run to ~2 GB each. Set "
            "CYTOOLS_DB_DIR to a local mirror instead."
        )

    saved = {
        name: getattr(hub, name) for name in ("hf_hub_download", "list_repo_files")
    }
    for name in saved:
        setattr(hub, name, refuse)
    yield
    for name, fn in saved.items():
        setattr(hub, name, fn)
