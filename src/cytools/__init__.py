# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""Public CYTools API with side-effect-free, feature-lazy imports."""

from importlib import import_module

from cytools._updates import check_for_updates
from cytools._version import __version__, version, versions_with_serious_bugs
from cytools.polytope import Polytope
from cytools.h_polytope import HPolytope
from cytools.cone import Cone
from cytools.dataset import load_polytopes
from cytools.landscape import (
    Geometry,
    Unsupported,
    quantities,
    quantity,
    scan,
    status,
    sweep,
)
from cytools.utils import fetch_polytopes, read_polytopes
import cytools.config as config


_LAZY_SUBMODULES = frozenset({"ntfe", "vector_config"})


def __getattr__(name: str):
    """Load feature namespaces only when users explicitly request them."""

    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES)


__all__ = [
    "Cone",
    "Geometry",
    "HPolytope",
    "Polytope",
    "Unsupported",
    "__version__",
    "check_for_updates",
    "config",
    "fetch_polytopes",
    "load_polytopes",
    "ntfe",
    "quantities",
    "quantity",
    "read_polytopes",
    "scan",
    "status",
    "sweep",
    "vector_config",
    "version",
    "versions_with_serious_bugs",
]
