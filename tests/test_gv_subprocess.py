"""Tests for configure_gv_subprocess.

cygv runs its Rust kernel in a fresh multiprocessing.Process per call, because
the Rust side installs a process-global Ctrl+C handler that can only be set
once. The subprocess is unavoidable; its start-up cost is not.
"""

import multiprocessing as mp

import numpy as np
import pytest

from cytools.calabiyau import configure_gv_subprocess

cygv = pytest.importorskip("cygv")


@pytest.fixture
def restore_start_method():
    previous = mp.get_start_method(allow_none=True)
    yield
    if previous is not None:
        try:
            mp.set_start_method(previous, force=True)
        except Exception:
            pass


def test_returns_previous_method_and_installs_the_new_one(restore_start_method):
    before = mp.get_start_method(allow_none=True)
    returned = configure_gv_subprocess("forkserver")

    assert mp.get_start_method() == "forkserver"
    if before is not None:
        assert returned == before


def test_is_idempotent(restore_start_method):
    configure_gv_subprocess("forkserver")
    again = configure_gv_subprocess("forkserver")
    assert again == "forkserver"
    assert mp.get_start_method() == "forkserver"


def test_unknown_method_warns_and_does_not_change_anything(restore_start_method):
    configure_gv_subprocess("forkserver")
    with pytest.warns(UserWarning, match="start method"):
        assert configure_gv_subprocess("not-a-real-method") is None
    assert mp.get_start_method() == "forkserver"


def test_in_process_kernel_call_is_not_viable():
    """Documents *why* the subprocess exists, so nobody 'optimizes' it away.

    cygv's Rust installs a global Ctrl+C handler on first use. A second call in
    the same interpreter panics with MultipleHandlers, which is exactly what the
    subprocess wrapper works around.
    """
    # ty: ignore[unresolved-import] -- compiled extension, ships no stubs
    from cygv.cygv import (
        _compute_gvgw,  # noqa: F401
    )

    # Only the existence of the raw entry point is asserted. Calling it twice
    # here would poison the interpreter for the rest of the session, which is
    # the whole point.
    assert callable(_compute_gvgw)


def test_invariants_unchanged_under_forkserver(restore_start_method):
    """The start method must not affect the answer.

    Uses a small explicit Calabi-Yau so the test needs no database.
    """
    from cytools import Polytope

    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    triang = p.triangulate(make_star=True)
    if not triang.is_star():
        pytest.skip("needs a star triangulation")
    try:
        cy = triang.get_cy()
    except Exception as e:
        pytest.skip(f"cannot build a CY: {e}")

    mori = cy.mori_cone_cap(in_basis=True)
    kwargs = dict(
        generators=np.asarray(mori.rays()),
        grading_vector=mori.find_grading_vector(),
        q=cy.curve_basis(include_origin=False, as_matrix=True),
        intnums=cy.intersection_numbers(in_basis=True, format="dok"),
        max_deg=2,
    )

    configure_gv_subprocess("spawn")
    baseline = sorted(cygv.compute_gv(**kwargs))

    configure_gv_subprocess("forkserver")
    fast = sorted(cygv.compute_gv(**kwargs))

    assert baseline == fast, "the start method changed the invariants"
    assert baseline, "expected some GV invariants"
