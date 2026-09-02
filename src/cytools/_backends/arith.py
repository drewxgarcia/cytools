# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""Small exact-arithmetic helpers shared by the engine adapters.

Duplicated rather than imported from `cytools.utils`: backend modules must not
import domain code, which `tests/test_architecture.py` pins mechanically.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

__all__ = ["gcd_int"]


def gcd_int(values: Iterable[float]) -> int:
    """The exact integer gcd of `values`, never zero."""
    g = 0
    for v in values:
        g = math.gcd(g, abs(int(v)))
    return g or 1
