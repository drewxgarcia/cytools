from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, overload

from numpy.typing import NDArray

Scalar: TypeAlias = int | float
MatrixLike: TypeAlias = NDArray[Any] | Sequence[Sequence[Scalar]]
PolytopeInput: TypeAlias = MatrixLike | str
NefPartitionWithHodge: TypeAlias = tuple[list[list[int]], list[list[int]], int]
NefPartitionNoHodge: TypeAlias = tuple[list[list[int]], None, None]


class Polytope:
    def __init__(self, data: PolytopeInput) -> None: ...
    def vertices(self) -> NDArray[Any]: ...
    def points(self) -> NDArray[Any]: ...
    def equations(self) -> NDArray[Any]: ...
    def normal_form(self, affine: bool = False) -> NDArray[Any]: ...
    @overload
    def nef_partitions(
        self,
        codim: int = ...,
        keep_symmetric: bool = ...,
        keep_products: bool = ...,
        keep_projections: bool = ...,
        with_hodge_numbers: Literal[True] = ...,
    ) -> list[NefPartitionWithHodge]: ...
    @overload
    def nef_partitions(
        self,
        codim: int = ...,
        keep_symmetric: bool = ...,
        keep_products: bool = ...,
        keep_projections: bool = ...,
        with_hodge_numbers: Literal[False] = ...,
    ) -> list[NefPartitionNoHodge]: ...
    @overload
    def nef_partitions(
        self,
        codim: int = ...,
        keep_symmetric: bool = ...,
        keep_products: bool = ...,
        keep_projections: bool = ...,
        with_hodge_numbers: bool = ...,
    ) -> list[NefPartitionWithHodge] | list[NefPartitionNoHodge]: ...
