"""Tests for the sparse linear solver backends in cytools.utils."""

import builtins
import contextlib
import sys
import warnings
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp

from cytools.utils import PerformanceWarning, solve_linear_system


def _least_squares_system(seed=0, m=60, n=15):
    """An overdetermined, consistent system M x + C = 0 with known solution."""
    rng = np.random.default_rng(seed)
    dense = rng.integers(-3, 4, size=(m, n)).astype(float)
    dense[:n, :n] += np.eye(n) * (n + 1)  # keep M^T M well conditioned
    M = sp.csr_matrix(dense)
    x_true = rng.integers(-5, 6, size=n).astype(float)
    C = -(M @ x_true)
    return M, list(C), x_true


@pytest.mark.requires_dependency("sksparse.cholmod")
def test_scikit_sparse_api_when_installed():
    """The optional performance extra must expose the 0.5+ API.

    Pinned because the API this depends on has moved: scikit-sparse 0.5 removed
    ``cholesky_AAt`` in favour of ``cho_factor``. A missing module or a missing
    name should fail loudly here rather than degrade the solver silently.
    """
    from sksparse.cholmod import CholmodError, cho_factor

    assert callable(cho_factor)
    assert issubclass(CholmodError, Exception)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "sksparse", marks=pytest.mark.requires_dependency("sksparse.cholmod")
        ),
        "scipy",
        "all",
    ],
)
def test_backend_solves(backend):
    """Every backend must actually return a solution, not None.

    Regression test: the sksparse branch called ``cholesky_AAt``, which
    scikit-sparse removed in 0.5. The resulting ImportError was swallowed by a
    blanket ``except``, so the backend returned None and every solve silently
    degraded to the scipy SuperLU fallback -- a large, invisible slowdown rather
    than a failure. Assert success, not merely correctness.
    """
    M, C, x_true = _least_squares_system()
    sol = solve_linear_system(M, C, backend=backend)
    assert sol is not None, f"backend {backend!r} returned no solution"
    assert np.allclose(np.asarray(sol).ravel(), x_true, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.requires_dependency("sksparse.cholmod")
def test_backends_agree(seed):
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
        with pytest.raises(ImportError, match=r"cytools-workbench\[performance\]"):
            solve_linear_system(M, C, backend="sksparse")


def test_automatic_backend_falls_back_when_scikit_sparse_is_absent():
    """A base install remains functional on platforms without SuiteSparse."""
    M, C, x_true = _least_squares_system()
    with _sksparse_unavailable():
        sol = solve_linear_system(M, C, backend="all")

    assert sol is not None
    assert np.allclose(np.asarray(sol).ravel(), x_true, atol=1e-6)


# ---------------------------------------------------------------------------
# Falling back to SciPy must be loud, and the SciPy ordering must not regress
# ---------------------------------------------------------------------------


def _hide_sksparse(monkeypatch):
    """Make `import sksparse.cholmod` raise, even when it is installed."""
    monkeypatch.setitem(sys.modules, "sksparse", None)
    monkeypatch.setitem(sys.modules, "sksparse.cholmod", None)


def test_missing_cholmod_warns_instead_of_silently_slowing_down(monkeypatch):
    """The fallback is correct but ~20x slower on real systems, and used to be
    announced only at verbosity>=1 -- which nothing sets during a sweep."""
    _hide_sksparse(monkeypatch)
    M, C, x_true = _least_squares_system()

    with pytest.warns(PerformanceWarning) as rec:
        sol = solve_linear_system(M, C, backend="all")

    assert sol is not None
    assert np.allclose(np.asarray(sol).ravel(), x_true, atol=1e-6)

    msg = str(rec[0].message)
    assert "cytools-workbench[performance]" in msg  # says how to fix it
    assert "3x" in msg or "20x" in msg  # says what it costs


@pytest.mark.requires_dependency("sksparse.cholmod")
def test_no_warning_when_cholmod_is_available():
    M, C, _ = _least_squares_system()
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        assert solve_linear_system(M, C, backend="all") is not None


def test_explicit_scipy_backend_does_not_warn(monkeypatch):
    """Asking for SciPy is a choice, not an accident."""
    _hide_sksparse(monkeypatch)
    M, C, _ = _least_squares_system()
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        assert solve_linear_system(M, C, backend="scipy") is not None


def test_scipy_backend_uses_the_measured_ordering():
    """`MMD_ATA` beat the COLAMD default by ~1.2x on real intersection-number
    systems, while `MMD_AT_PLUS_A` and `NATURAL` were 45x and 100x slower.
    Pin the choice so it is not "tidied" back without a measurement.
    """
    captured = {}
    real = sp.linalg.spsolve

    def spy(A, b, **kwargs):
        captured.update(kwargs)
        return real(A, b, **kwargs)

    M, C, x_true = _least_squares_system()
    with mock.patch.object(sp.linalg, "spsolve", spy):
        sol = solve_linear_system(M, C, backend="scipy")

    assert captured.get("permc_spec") == "MMD_ATA"
    assert np.allclose(np.asarray(sol).ravel(), x_true, atol=1e-6)
