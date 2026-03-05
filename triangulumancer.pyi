from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

from numpy.typing import NDArray

Scalar: TypeAlias = int | float
PointMatrixLike: TypeAlias = NDArray[Any] | Sequence[Sequence[Scalar]]
IndexMatrixLike: TypeAlias = NDArray[Any] | Sequence[Sequence[int]]
HeightLike: TypeAlias = NDArray[Any] | Sequence[Scalar]


class PointConfiguration:
    def __init__(self, points: PointMatrixLike) -> None: ...
    def triangulate_with_heights(self, heights: HeightLike) -> Triangulation: ...
    def fine_triangulation(self) -> Triangulation: ...
    def all_triangulations(self, only_fine: bool = False) -> list[Triangulation]: ...


class Triangulation:
    def __init__(
        self, point_config: PointConfiguration, simplices: IndexMatrixLike
    ) -> None: ...
    def simplices(self) -> NDArray[Any]: ...
    def neighbors(self) -> list[Triangulation]: ...


class VectorConfiguration:
    def __init__(self, vectors: PointMatrixLike) -> None: ...
