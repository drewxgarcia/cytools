"""Public choice types stay aligned with the runtime contracts they describe."""

from typing import get_args

from cytools._typing import (
    IntnumFormat,
    Lattice,
    LinearSolverBackend,
    PolytopeFormat,
    PolytopeInputType,
    PolytopeSource,
)
from cytools.utils import fetch_polytopes, read_polytopes, solve_linear_system


def test_choice_aliases_are_exact():
    assert get_args(Lattice) == ("N", "M")
    assert get_args(LinearSolverBackend) == ("all", "sksparse", "scipy")
    assert get_args(IntnumFormat) == ("dok", "coo", "dense")
    assert get_args(PolytopeSource) == ("auto", "database", "web")
    assert get_args(PolytopeFormat) == ("ks", "ws")
    assert get_args(PolytopeInputType) == ("file", "str")


def test_notebook_facing_annotations_use_the_shared_aliases():
    assert fetch_polytopes.__annotations__["lattice"] is Lattice
    assert fetch_polytopes.__annotations__["source"] is PolytopeSource
    assert read_polytopes.__annotations__["input_type"] is PolytopeInputType
    assert read_polytopes.__annotations__["format"] is PolytopeFormat
    assert solve_linear_system.__annotations__["backend"] is LinearSolverBackend
