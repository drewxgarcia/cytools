# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""Problem-size thresholds at which one engine overtakes another.

These used to be bare integers inline in the domain code -- most notably
`ambient_dim >= 25` in five places in `cone.py` -- with no derivation anywhere
in the tree and a docstring that cited itself ("up to around 25"). Collecting
them here lets an unmeasured value be *labelled* as unmeasured instead of
passing for evidence.

Regenerate with `benchmarks/calibrate_engines.py`; pinned by
`tests/test_crossovers.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CROSSOVERS", "Crossover"]


@dataclass(frozen=True)
class Crossover:
    """One measured (or explicitly unmeasured) engine threshold.

    **Attributes:**
    - `value`: The problem size at which `above` overtakes `below`.
    - `below`: Engine preferred strictly below `value`.
    - `above`: Engine preferred at or above `value`.
    - `metric`: The problem descriptor key `value` is compared against.
    - `provenance`: Where the number comes from. `None` means *not measured* --
        inherited from upstream and carried forward unverified. Callers must
        be able to tell the difference.
    """

    value: float
    below: str
    above: str
    metric: str
    provenance: str | None = None

    @property
    def measured(self) -> bool:
        """Whether this threshold rests on a recorded measurement."""
        return self.provenance is not None


CROSSOVERS: dict[str, Crossover] = {
    # NOT MEASURED. Inherited from upstream, where it gated the Mosek default
    # and printed a "this may not work" hint. Mosek is licence-gated and is
    # not installed in this environment, so the threshold cannot be checked
    # here. It is retained only to order Mosek ahead of the open-source QP
    # engines when a licence *is* present -- never to exclude anything, so a
    # wrong value costs speed and not correctness.
    "stretched_tip.osqp_to_mosek": Crossover(
        value=25,
        below="osqp",
        above="mosek",
        metric="dim",
        provenance=None,
    ),
}
