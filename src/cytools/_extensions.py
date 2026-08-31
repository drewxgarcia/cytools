# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""Lazy bindings for methods implemented in optional feature modules.

Domain classes declare their complete public surface themselves.  A binding
imports its implementation only when the method is first accessed, avoiding
both import-time class mutation and eager loading of feature modules.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import MethodType
from typing import Any


class LazyMethod:
    """Descriptor that resolves a module-level method implementation lazily.

    Resolution happens once. It used to happen on *every* attribute access:
    `__get__` called `_resolve()` unconditionally, and because this is a
    non-data descriptor with no caching, nothing ever shadowed it. That cost an
    `import_module`, a `getattr` and a `callable()` check per call -- measured
    at 562 ns against 61 ns for an ordinary method on the same class, a 9.2x
    penalty that never went away after warm-up.
    """

    __slots__ = ("_attribute", "_implementation", "_module", "_target")

    def __init__(self, module: str, target: str | None = None) -> None:
        self._module = module
        self._target = target
        self._attribute: str | None = None
        self._implementation: Callable[..., Any] | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        self._attribute = name

    def _resolve(self) -> Callable[..., Any]:
        implementation = self._implementation
        if implementation is not None:
            return implementation

        target = self._target or self._attribute
        if target is None:  # pragma: no cover - Python calls __set_name__
            raise RuntimeError("lazy method has not been bound to a class")

        implementation = getattr(import_module(self._module), target)
        if not callable(implementation):
            raise TypeError(f"{self._module}.{target} is not callable")
        # Cached on the descriptor rather than on the instance or the owner
        # class. The target is a module-level function shared by every
        # instance, and writing to the owner would shadow the descriptor for
        # one subclass only.
        self._implementation = implementation
        return implementation

    def __get__(self, instance: object | None, owner: type | None = None):
        implementation = self._resolve()
        if instance is None:
            return implementation
        return MethodType(implementation, instance)


def lazy_method(module: str, target: str | None = None) -> LazyMethod:
    """Declare a class method whose implementation lives in another module."""

    return LazyMethod(module, target)
