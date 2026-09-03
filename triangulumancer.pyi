from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

type Scalar = int | float | np.number
type VectorLike = NDArray[Any] | Sequence[Scalar]
# rows may themselves be arrays, not just sequences
type PointMatrixLike = NDArray[Any] | Sequence[VectorLike]
type IndexMatrixLike = NDArray[Any] | Sequence[NDArray[Any] | Sequence[int]]
type HeightLike = VectorLike

class PointConfiguration:
    def __init__(self, points: PointMatrixLike) -> None: ...
    def triangulate_with_heights(self, heights: HeightLike) -> Triangulation: ...
    def fine_triangulation(self) -> Triangulation: ...
    def all_triangulations(self, only_fine: bool = False) -> list[Triangulation]: ...

class Triangulation:
    def __init__(
        self, point_config: PointConfiguration, simplices: IndexMatrixLike
    ) -> None: ...
    # attributes, not methods
    dim: int
    n_simplices: int
    simplices: NDArray[Any]

    def neighbors(self) -> list[Triangulation]: ...
    def flips(self) -> list[Triangulation]: ...
    def bistellar_flips(self) -> list[Triangulation]: ...

class VectorConfiguration:
    def __init__(self, vectors: PointMatrixLike) -> None: ...
