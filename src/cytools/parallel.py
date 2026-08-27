# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""
Distributed / parallel execution for large-scale KS database scans.

Two backends are supported and selected automatically:

**Ray** (preferred)
    When ``ray`` is importable, all tasks are dispatched to a Ray cluster
    (or local Ray instance).  This gives true distributed compute, shared
    object store (zero-copy numpy arrays between workers), and fault
    tolerance.  Install with ``pip install ray[default]``.

**ProcessPoolExecutor** (fallback)
    When ``ray`` is not available, tasks are dispatched to a local
    ``concurrent.futures.ProcessPoolExecutor``.  Semantics are identical;
    only distribution and the object store are absent.

Usage::

    from cytools.parallel import remote, get, init, shutdown

    # Define a task (must be module-level for pickling)
    @remote
    def pipeline(record):
        t = record.polytope.triangulate()
        cy = t.get_toric_variety().get_cy()
        return cy.hodge_numbers()

    # Run a scan
    init()                                          # start pool / Ray
    refs = [pipeline.remote(rec) for rec in records]
    results = get(refs)                             # blocks until all done
    shutdown()                                      # clean up

Or use the context manager::

    from cytools.parallel import pool
    from cytools.dataset import load_polytopes

    records = load_polytopes(n_vertices=list(range(5, 15)), n=500)

    with pool(n_workers=8) as p:
        refs  = [p.submit(pipeline, rec) for rec in records]
        results = p.get(refs)

The :func:`parallel_scan` convenience wrapper in :mod:`cytools.dataset` uses
this module internally.
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Callable, TypeVar

try:
    import ray  # ty:ignore[unresolved-import]
except ImportError:
    ray = None

T = TypeVar("T")

def _ray_available() -> bool:
    return ray is not None


_BACKEND: str | None = None   # "ray" | "ppe" | None (not initialised)
_PPE: ProcessPoolExecutor | None = None
_N_WORKERS: int | None = None


def _active_backend() -> str:
    if _BACKEND is None:
        raise RuntimeError(
            "CYTools parallel pool not initialised. "
            "Call cytools.parallel.init() first, or use the pool() context manager."
        )
    return _BACKEND


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def init(n_workers: int | None = None, *, force_backend: str | None = None) -> str:
    """
    Start the parallel pool.

    Automatically selects Ray when available, otherwise falls back to
    ``ProcessPoolExecutor``.

    **Arguments:**
    - `n_workers`: Number of parallel workers.  Defaults to ``os.cpu_count()``.
      Ignored by Ray when connecting to a pre-existing cluster.
    - `force_backend`: ``"ray"`` or ``"ppe"`` to override auto-detection.

    **Returns:**
    The name of the active backend (``"ray"`` or ``"ppe"``).
    """
    global _BACKEND, _PPE, _N_WORKERS

    workers = n_workers or os.cpu_count() or 1
    _N_WORKERS = workers

    use_ray = (
        force_backend == "ray"
        or (force_backend is None and _ray_available())
    )

    if use_ray:
        assert ray is not None
        if not ray.is_initialized():
            ray.init(num_cpus=workers, ignore_reinit_error=True)
        _BACKEND = "ray"
    else:
        if _PPE is not None:
            _PPE.shutdown(wait=False)
        _PPE = ProcessPoolExecutor(max_workers=workers)
        _BACKEND = "ppe"

    return _BACKEND


def shutdown(wait: bool = True) -> None:
    """Shut down the parallel pool and release resources."""
    global _BACKEND, _PPE

    if _BACKEND == "ray":
        assert ray is not None
        if ray.is_initialized():
            ray.shutdown()
    elif _BACKEND == "ppe" and _PPE is not None:
        _PPE.shutdown(wait=wait)
        _PPE = None

    _BACKEND = None


# ---------------------------------------------------------------------------
# Task submission and retrieval
# ---------------------------------------------------------------------------

def get(refs: list) -> list:
    """
    Block until all *refs* are complete and return results in order.

    *refs* can be Ray ObjectRefs or ``concurrent.futures.Future`` objects.
    """
    if not refs:
        return []

    backend = _active_backend()

    if backend == "ray":
        assert ray is not None
        return ray.get(refs)

    # ProcessPoolExecutor path
    return [f.result() for f in refs]


def submit(fn: Callable, *args, **kwargs) -> Any:
    """
    Submit a single task and return a reference (Future or Ray ObjectRef).

    Prefer the :class:`RemoteFunction` API (``@remote`` + ``.remote()``)
    for large-scale dispatch; this is a lower-level escape hatch.
    """
    backend = _active_backend()

    if backend == "ray":
        assert ray is not None
        ray_fn = ray.remote(fn)
        return ray_fn.remote(*args, **kwargs)

    assert _PPE is not None
    return _PPE.submit(fn, *args, **kwargs)


# ---------------------------------------------------------------------------
# @remote decorator
# ---------------------------------------------------------------------------

class RemoteFunction:
    """
    Wraps a callable so that ``.remote(*args)`` dispatches it to the pool.

    The object is itself callable for normal (non-parallel) use.

    Example::

        @remote
        def compute(record):
            return record.polytope.triangulate().get_toric_variety().get_cy().hodge_numbers()

        init()
        refs    = [compute.remote(rec) for rec in records]
        results = get(refs)
        shutdown()
    """

    def __init__(self, fn: Callable) -> None:
        self._fn = fn
        self.__name__ = getattr(fn, "__name__", repr(fn))
        self.__doc__  = getattr(fn, "__doc__",  None)

    def __call__(self, *args, **kwargs):
        """Call the underlying function synchronously (no pool involved)."""
        return self._fn(*args, **kwargs)

    def remote(self, *args, **kwargs):
        """Dispatch to the pool and return a reference."""
        return submit(self._fn, *args, **kwargs)


def remote(fn: Callable) -> RemoteFunction:
    """
    Decorator that marks a function for remote/parallel execution.

    Mirrors the ``@ray.remote`` decorator API so that code written against
    this module works unmodified once Ray gains Python 3.14 wheels::

        @remote
        def pipeline(record): ...

        refs = [pipeline.remote(rec) for rec in records]
        results = get(refs)
    """
    return RemoteFunction(fn)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class pool:
    """
    Context manager that initialises and tears down a parallel pool.

    Example::

        with pool(n_workers=8) as p:
            refs    = [p.submit(fn, rec) for rec in records]
            results = p.get(refs)
    """

    def __init__(
        self,
        n_workers: int | None = None,
        *,
        force_backend: str | None = None,
    ) -> None:
        self._n_workers = n_workers
        self._force_backend = force_backend
        self.backend: str | None = None

    def __enter__(self) -> "pool":
        self.backend = init(self._n_workers, force_backend=self._force_backend)
        return self

    def __exit__(self, *_) -> None:
        shutdown()

    def submit(self, fn: Callable, *args, **kwargs) -> Any:
        """Submit one task; returns a Future or Ray ObjectRef."""
        return submit(fn, *args, **kwargs)

    def map(self, fn: Callable, iterable, *, chunksize: int = 1) -> list:
        """
        Apply *fn* to every element of *iterable* in parallel; return results
        in order.  Mirrors ``ProcessPoolExecutor.map`` semantics.
        """
        refs = [submit(fn, item) for item in iterable]
        return get(refs)

    def get(self, refs: list) -> list:
        """Block until all refs complete; return results in order."""
        return get(refs)
