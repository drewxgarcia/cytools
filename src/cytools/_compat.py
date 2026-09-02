# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""Small compatibility primitives for deliberate public API migrations."""

from __future__ import annotations

import warnings

__all__ = ["resolve_deprecated_bool"]


def resolve_deprecated_bool(
    value: bool,
    legacy_value: bool | None,
    *,
    name: str,
    legacy_name: str,
) -> bool:
    """Resolve a renamed boolean keyword without hiding contradictory input.

    ``None`` is the sentinel for an omitted legacy spelling. A caller may use
    both spellings when they agree, but conflicting values are rejected rather
    than letting argument order decide behavior.
    """
    if legacy_value is None:
        return value

    warnings.warn(
        f"{legacy_name}= is deprecated; use {name}= instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    if value and value != legacy_value:
        raise ValueError(
            f"Conflicting values for {name}={value!r} and its deprecated "
            f"alias {legacy_name}={legacy_value!r}."
        )
    return legacy_value
