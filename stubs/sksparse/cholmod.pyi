"""Types for `sksparse.cholmod`, a compiled extension that ships no stubs.

Kept deliberately narrow: it declares exactly the surface `cytools` uses, so a
name disappearing upstream -- as `cholesky_AAt` did in scikit-sparse 0.5 -- is
a type error here rather than a runtime `ImportError` in the solver.
"""

from typing import Any

class CholmodError(Exception): ...

class Factor:
    def solve(self, b: Any) -> Any: ...
    def solve_A(self, b: Any) -> Any: ...

def cho_factor(
    A: Any,
    beta: float = ...,
    *,
    lower: bool = ...,
    order: str = ...,
    sym_kind: Any | None = ...,
    supernodal_mode: Any | None = ...,
) -> Factor: ...
def cholesky(
    A: Any,
    beta: float = ...,
    *,
    lower: bool = ...,
    order: str = ...,
    sym_kind: Any | None = ...,
    supernodal_mode: Any | None = ...,
) -> Factor: ...
