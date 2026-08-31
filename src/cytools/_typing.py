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

import numpy as np

#: A single number, Python or numpy.
Scalar = int | float | np.number

#: A 1-dimensional array of numbers: a row/point/height vector.
Vector = np.ndarray | Sequence[Scalar]

#: A 2-dimensional array of numbers: a list of points, rays, inequalities, ...
Matrix = np.ndarray | Sequence[Vector]

#: Either of the above -- for arguments accepting a vector or a matrix.
VectorOrMatrix = Vector | Matrix

__all__ = ["Matrix", "Scalar", "Vector", "VectorOrMatrix"]
