# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# CYTools is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# CYTools. If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Type aliases for the array-like inputs used across CYTools.
#
#               `numpy.typing.ArrayLike` is deliberately not used for these:
#               it also admits scalars and bare strings, so it cannot express
#               "something you can call len() on and index into", which is what
#               essentially every CYTools entry point actually requires.
# -----------------------------------------------------------------------------

from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np

#: A single number, Python or numpy.
Scalar = int | float | np.number

#: A 1-dimensional array of numbers: a row/point/height vector.
#:
#: Spelled as a union of sequence types rather than the more obvious
#: `Sequence[Scalar]`. numpy's `ArrayLike` is a union whose sequence members are
#: `_NestedSequence[_SupportsArray[...]]` and `_NestedSequence[bool | int |
#: float | ...]`; `Sequence[int | float]` satisfies the second and
#: `Sequence[np.number]` the first, but `Sequence[int | float | np.number]`
#: satisfies neither, because no single member covers a mixed element type.
#: Writing it this way keeps every `Vector` assignable to `ArrayLike`, which is
#: what the numpy-facing call sites and the `regfans` base classes require.
Vector = np.ndarray | Sequence[int | float] | Sequence[np.number]


class SupportsArray(Protocol):
    """Anything numpy can consume through the array protocol.

    `np.asarray` accepts any object exposing `__array__`, and several CYTools
    types do (`helpers.matrix.CSR_stack`, for one). Kept out of `Matrix`
    deliberately: `Matrix` must stay assignable to numpy's `ArrayLike`, and a
    local protocol is not. Use it alongside `Matrix` at the call sites that
    genuinely accept the array protocol, such as `Cone(hyperplanes=...)`.
    """

    def __array__(self, dtype: Any = None) -> np.ndarray: ...

    def __len__(self) -> int: ...


#: A 2-dimensional array of numbers: a list of points, rays, inequalities, ...
#: Spelled out arm-by-arm for the same reason as `Vector` above: `Sequence[T]`
#: where `T` is itself a union satisfies no single member of `ArrayLike`.
Matrix = (
    np.ndarray
    | Sequence[np.ndarray]
    | Sequence[Sequence[int | float]]
    | Sequence[Sequence[np.number]]
)

#: Either of the above -- for arguments accepting a vector or a matrix.
VectorOrMatrix = Vector | Matrix

#: Lattice convention used throughout the public geometry API.
Lattice: TypeAlias = Literal["N", "M"]

#: Sparse linear solver selected by :func:`cytools.utils.solve_linear_system`.
LinearSolverBackend: TypeAlias = Literal["all", "sksparse", "scipy"]

#: Serialized representation of an intersection-number tensor.
IntnumFormat: TypeAlias = Literal["dok", "coo", "dense"]

#: Where :func:`cytools.fetch_polytopes` obtains 4D records.
PolytopeSource: TypeAlias = Literal["auto", "database", "web"]

#: Input encoding accepted by :func:`cytools.read_polytopes`.
PolytopeFormat: TypeAlias = Literal["ks", "ws"]

#: Container accepted by :func:`cytools.read_polytopes`.
PolytopeInputType: TypeAlias = Literal["file", "str"]

#: Convex-hull engine used to construct a :class:`cytools.Polytope`.
PolytopeBackend: TypeAlias = Literal["ppl", "qhull", "palp"]

#: Side on which polytope automorphism matrices act.
AutomorphismAction: TypeAlias = Literal["left", "right"]

#: Engine used to compute a Kreuzer--Skarke normal form.
NormalFormBackend: TypeAlias = Literal["native", "palp"]

#: Engine used to construct a triangulation.
TriangulationBackend: TypeAlias = Literal["cgal", "qhull", "topcom"]

#: Triangulation engines that accept sampled heights.
RandomTriangulationBackend: TypeAlias = Literal["cgal", "qhull"]

#: Strategy used to sample fine triangulations of large two-faces.
FaceTriangulationMethod: TypeAlias = Literal["fast", "fair", "grow2d", "dualgnn"]

#: Engine used to compute a triangulation's secondary cone.
SecondaryConeBackend: TypeAlias = Literal["native", "topcom"]

#: Cone ray-pruning algorithm.
ExtremalRaysMethod: TypeAlias = Literal["extremalrays", "legacy", "nnls"]

#: Algorithm used to test whether one ray is extremal.
ExtremalityMethod: TypeAlias = Literal["lp", "nnls"]

#: Algorithm used to test whether a cone is pointed.

#: Optimizer accepted by :meth:`cytools.Cone.tip_of_stretched_cone`.
StretchedConeBackend: TypeAlias = Literal["mosek", "osqp", "cvxopt", "highs", "glop"]

#: Optimizer accepted by :meth:`cytools.Cone.find_interior_point`.
InteriorPointBackend: TypeAlias = Literal[
    "highs", "glop", "scip", "cpsat", "mosek", "osqp", "cvxopt"
]

#: Linear/integer optimizer used by the cone feasibility kernel.
FeasibilityBackend: TypeAlias = Literal["highs", "glop", "scip", "cpsat"]

#: Serialized representation of Gopakumar--Vafa or Gromov--Witten invariants.
InvariantFormat: TypeAlias = Literal["dok", "coo"]

#: Invariant family selected by the shared GV/GW implementation.
InvariantKind: TypeAlias = Literal["gv", "gw"]

#: Start method supported by Python's multiprocessing module.
ProcessStartMethod: TypeAlias = Literal["fork", "forkserver", "spawn"]

__all__ = [
    "SupportsArray",
    "AutomorphismAction",
    "ExtremalityMethod",
    "ExtremalRaysMethod",
    "FaceTriangulationMethod",
    "FeasibilityBackend",
    "IntnumFormat",
    "InteriorPointBackend",
    "InvariantFormat",
    "InvariantKind",
    "Lattice",
    "LinearSolverBackend",
    "Matrix",
    "NormalFormBackend",
    "PolytopeBackend",
    "PolytopeFormat",
    "PolytopeInputType",
    "PolytopeSource",
    "ProcessStartMethod",
    "RandomTriangulationBackend",
    "Scalar",
    "SecondaryConeBackend",
    "StretchedConeBackend",
    "TriangulationBackend",
    "Vector",
    "VectorOrMatrix",
]
