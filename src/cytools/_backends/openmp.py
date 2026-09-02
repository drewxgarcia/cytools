# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""Detection of duplicate OpenMP runtimes.

LLVM's OpenMP runtime calls `abort()` from `__kmp_register_library_startup`
when it finds another runtime already registered in the process. The
interpreter dies with SIGABRT and no Python traceback, which is close to
undebuggable from the outside.

On macOS this is reachable from a normal install: homebrew's SuiteSparse
(reached through the `performance` extra, via `scikit-sparse` -> `libcholmod`)
links `/opt/homebrew/opt/libomp/lib/libomp.dylib`, while PyTorch (reached
through the `gnn` extra) bundles its own copy under `torch/lib/`. dyld keys
loaded images on their resolved path, so the two distinct files register
twice and the second one aborts.

This module refuses to load a second runtime. A Python exception is a much
safer failure mode than a process-wide SIGABRT, especially in a notebook where
the kernel and unsaved state would otherwise be lost.

It deliberately performs no repair: pointing one library at the other's
runtime mutates an installed package, which is the user's call.
"""

import ctypes
import importlib.util
import os
import sys
from pathlib import Path

__all__ = [
    "OpenMPRuntimeConflict",
    "conflicting_runtime",
    "ensure_compatible",
    "loaded_runtimes",
]


class OpenMPRuntimeConflict(RuntimeError):
    """Loading an optional backend would register a second OpenMP runtime."""


# Basenames of the OpenMP runtimes that register with LLVM's startup guard and
# therefore abort on a duplicate. `libiomp5` is Intel's, reached through MKL and
# through some numpy/scipy builds; matching only `libomp` missed it, which is
# the most common source of a second runtime after PyTorch's bundled copy.
# `libgomp` is deliberately absent: GCC's runtime tolerates duplicates and does
# not participate in the registration that aborts.
_OPENMP_BASENAMES = (b"libomp", b"libiomp5")


def _is_openmp_runtime(basename: bytes) -> bool:
    """Whether a dyld image basename names an abort-on-duplicate OpenMP runtime."""
    return any(name in basename for name in _OPENMP_BASENAMES)


def loaded_runtimes() -> set[str]:
    """
    **Description:**
    The resolved paths of every OpenMP runtime already loaded into this
    process. Returns an empty set on platforms where this check does not
    apply, or if the dyld introspection API is unavailable.

    **Returns:**
    A set of absolute, symlink-resolved paths.
    """
    if sys.platform != "darwin":
        # The same class of conflict exists elsewhere, but the detection below
        # is dyld-specific and the abort has not been reproduced there.
        return set()
    try:
        libc = ctypes.CDLL(None)
        libc._dyld_image_count.restype = ctypes.c_uint32
        libc._dyld_get_image_name.restype = ctypes.c_char_p
        libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        found = set()
        for i in range(libc._dyld_image_count()):
            raw = libc._dyld_get_image_name(i)
            if raw and _is_openmp_runtime(raw.rsplit(b"/", 1)[-1]):
                found.add(os.path.realpath(os.fsdecode(raw)))
        return found
    except (AttributeError, OSError):
        return set()


def _bundled_torch_runtime() -> Path | None:
    """Locate PyTorch's bundled runtime *without importing torch*.

    `find_spec` resolves the package location without executing its
    `__init__`, which matters here: executing it is precisely the thing that
    would abort.
    """
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    lib = Path(next(iter(spec.submodule_search_locations))) / "lib" / "libomp.dylib"
    return lib if lib.exists() else None


def conflicting_runtime(candidate: "str | os.PathLike | None") -> str | None:
    """
    **Description:**
    Report the OpenMP runtime that `candidate` would collide with, if loading
    it would register a second runtime in this process.

    **Arguments:**
    - `candidate`: Path of the runtime that is about to be loaded. When None,
        no conflict is reported.

    **Returns:**
    The resolved path of the already-loaded conflicting runtime, or None when
    loading `candidate` is safe (nothing loaded yet, or the same file).
    """
    if candidate is None:
        return None
    loaded = loaded_runtimes()
    if not loaded:
        return None
    resolved = os.path.realpath(os.fspath(candidate))
    if resolved in loaded:
        # Same file -- dyld will reuse the existing image.
        return None
    return sorted(loaded)[0]


def ensure_compatible() -> None:
    """
    **Description:**
    Refuse to import PyTorch if a second OpenMP runtime is already loaded.

    **Returns:**
    Nothing. Raises `OpenMPRuntimeConflict` with an actionable repair when the
    import would be unsafe.
    """
    bundled = _bundled_torch_runtime()
    other = conflicting_runtime(bundled)
    if other is None:
        return
    raise OpenMPRuntimeConflict(
        "A second OpenMP runtime is already loaded in this process. LLVM's "
        "runtime calls abort() rather than tolerating a duplicate, so "
        "importing PyTorch here can kill the interpreter with SIGABRT and no "
        "Python traceback.\n"
        f"  already loaded: {other}\n"
        f"  about to load:  {bundled}\n"
        "When both are LLVM builds, the usual fix is to make PyTorch share the "
        "one already loaded:\n"
        f'  ln -sf "{other}" "{bundled}"\n'
        "Re-apply that after any reinstall of PyTorch, which restores its "
        "bundled copy. KMP_DUPLICATE_LIB_OK=TRUE silences the abort instead, "
        "but it suppresses the guard rather than removing the duplicate and "
        "is not safe for numerical work."
    )
