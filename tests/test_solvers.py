"""Tests for the sparse linear solver backends in cytools.utils."""

import builtins
import contextlib
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp

from cytools.utils import solve_linear_system


def _least_squares_system(seed=0, m=60, n=15):
    """An overdetermined, consistent system M x + C = 0 with known solution."""
    rng = np.random.default_rng(seed)
    dense = rng.integers(-3, 4, size=(m, n)).astype(float)
    dense[:n, :n] += np.eye(n) * (n + 1)  # keep M^T M well conditioned
    M = sp.csr_matrix(dense)
    x_true = rng.integers(-5, 6, size=n).astype(float)
    C = -(M @ x_true)
    return M, list(C), x_true


def test_scikit_sparse_api_when_installed():
    """The optional performance extra must expose the 0.5+ API.

    Pinned because the API this depends on has moved: scikit-sparse 0.5 removed
    ``cholesky_AAt`` in favour of ``cho_factor``. A missing module or a missing
    name should fail loudly here rather than degrade the solver silently.
    """
    pytest.importorskip("sksparse.cholmod")
    from sksparse.cholmod import (  # noqa: F401  # ty: ignore[unresolved-import]  # compiled extension, no stubs
        CholmodError,
        cho_factor,
    )


@pytest.mark.parametrize("backend", ["sksparse", "scipy", "all"])
def test_backend_solves(backend):
    """Every backend must actually return a solution, not None.

    Regression test: the sksparse branch called ``cholesky_AAt``, which
    scikit-sparse removed in 0.5. The resulting ImportError was swallowed by a
    blanket ``except``, so the backend returned None and every solve silently
    degraded to the scipy SuperLU fallback -- a large, invisible slowdown rather
    than a failure. Assert success, not merely correctness.
    """
    if backend == "sksparse":
        pytest.importorskip("sksparse.cholmod")
    M, C, x_true = _least_squares_system()
    sol = solve_linear_system(M, C, backend=backend)
    assert sol is not None, f"backend {backend!r} returned no solution"
    assert np.allclose(np.asarray(sol).ravel(), x_true, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_backends_agree(seed):
    pytest.importorskip("sksparse.cholmod")
    M, C, _ = _least_squares_system(seed=seed)
    a = solve_linear_system(M, C, backend="sksparse")
    b = solve_linear_system(M, C, backend="scipy")
    assert a is not None and b is not None
    assert np.allclose(np.asarray(a).ravel(), np.asarray(b).ravel(), atol=1e-6)


def test_inconsistent_system_is_rejected_by_the_residual_check():
    """A system with no solution must be reported as such, not returned."""
    M = sp.csr_matrix(np.array([[1.0], [1.0]]))
    C = [0.0, -100.0]  # x = 0 and x = 100 simultaneously
    assert solve_linear_system(M, C, backend="all", check=True) is None


@contextlib.contextmanager
def _sksparse_unavailable():
    """Simulate a base install in which importing `sksparse` fails.

    `mock.patch.object` rather than assigning `builtins.__import__` directly:
    it restores the real import hook even if the body raises, so a failing
    assertion cannot leave the interpreter unable to import anything.
    """
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("sksparse"):
            raise ImportError("simulated: scikit-sparse not installed")
        return real_import(name, *args, **kwargs)

    with mock.patch.object(builtins, "__import__", blocked):
        yield


def test_explicit_optional_backend_fails_with_install_guidance():
    """Explicit sksparse stays fail-loud and tells the user how to enable it."""
    M, C, _ = _least_squares_system()
    with _sksparse_unavailable():
        with pytest.raises(ImportError, match=r"cytools\[performance\]"):
            solve_linear_system(M, C, backend="sksparse")


def test_automatic_backend_falls_back_when_scikit_sparse_is_absent():
    """A base install remains functional on platforms without SuiteSparse."""
    M, C, x_true = _least_squares_system()
    with _sksparse_unavailable():
        sol = solve_linear_system(M, C, backend="all")

    assert sol is not None
    assert np.allclose(np.asarray(sol).ravel(), x_true, atol=1e-6)
