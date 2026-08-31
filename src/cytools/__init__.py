# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""The lightweight public CYTools facade.

Every non-metadata export is resolved on first access.  ``import cytools`` is
therefore a constant-time operation that does not initialize numerical
engines, dataframe libraries, optional features, or network clients.
"""

from importlib import import_module

from cytools._version import (
    __upstream_version__,
    __version__,
    upstream_version,
    version,
    versions_with_serious_bugs,
)

_LAZY_EXPORTS = {
    "Cone": ("cytools.cone", "Cone"),
    "Geometry": ("cytools.landscape", "Geometry"),
    "HPolytope": ("cytools.h_polytope", "HPolytope"),
    "Polytope": ("cytools.polytope", "Polytope"),
    "PerformanceWarning": ("cytools.utils", "PerformanceWarning"),
    "Unsupported": ("cytools.store", "Unsupported"),
    "check_for_updates": ("cytools._updates", "check_for_updates"),
    "config": ("cytools.config", None),
    "fetch_polytopes": ("cytools.utils", "fetch_polytopes"),
    "load_polytopes": ("cytools.dataset", "load_polytopes"),
    "ntfe": ("cytools.ntfe", None),
    "quantities": ("cytools.landscape", "quantities"),
    "quantity": ("cytools.landscape", "quantity"),
    "read_polytopes": ("cytools.utils", "read_polytopes"),
    "scan": ("cytools.landscape", "scan"),
    "status": ("cytools.landscape", "status"),
    "sweep": ("cytools.landscape", "sweep"),
    "vector_config": ("cytools.vector_config", None),
}


def __getattr__(name: str):
    """Resolve one public object without initializing unrelated subsystems."""

    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error

    module = import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_EXPORTS.keys())


__all__ = [
    "Cone",
    "Geometry",
    "HPolytope",
    "PerformanceWarning",
    "Polytope",
    "Unsupported",
    "__upstream_version__",
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
    "upstream_version",
    "vector_config",
    "version",
    "versions_with_serious_bugs",
]
