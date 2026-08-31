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
# Description:  Exact integer gcd reduction, and primitivization of lattice
#               vectors -- dividing out the gcd so a ray, hyperplane equation
#               or height vector is expressed by its shortest lattice
#               representative.
#
#               Primitivization was open-coded at six sites in three mutually
#               inconsistent spellings before this module existed: exact
#               floor division in `utils` and `ntfe`, but float division via
#               `np.rint(x / np.gcd.reduce(x)).astype(int)` in `f_theory`.
#               The float spelling routes an exact integer operation through
#               float64 and, on the zero vector, through `nan -> int`, which
#               numpy leaves undefined. One implementation, exact everywhere.
# -----------------------------------------------------------------------------

# 'standard' imports
# 3rd party imports
import numpy as np

# CYTools imports
from cytools._typing import Matrix, Vector

__all__ = ["gcd_reduce", "primitive"]


def gcd_reduce(a: Matrix | Vector, axis: int | None = None) -> np.ndarray:
    """
    **Description:**
    The gcd of the integers in *a*, over the whole array or along one axis.

    The gcd is insensitive to sign, so *a* need not be non-negative.

    **Arguments:**
    - `a`: An integer array.
    - `axis`: The axis to reduce along. `None` reduces the whole array to a
        scalar.

    **Returns:**
    The gcd(s). A 0-d array when `axis` is `None`, otherwise an array with
    `axis` removed. Zero where the corresponding input is entirely zero,
    matching `np.gcd.reduce`.

    **Example:**
    ```python {3}
    import numpy as np
    from cytools.helpers.arithmetic import gcd_reduce
    gcd_reduce(np.array([[2, 4], [9, 6]]), axis=1)
    # array([2, 3])
    ```
    """
    # `np.ufunc.reduce(np.gcd, ...)` rather than `np.gcd.reduce(...)`: the same
    # C routine reached through the class rather than the instance. numpy's
    # stubs type the unbound form but not the bound one, so this is the
    # spelling that needs no type-checker suppression. Measured within 1% of
    # the bound form on a (1500, 200) matrix.
    #
    # `functools.reduce(np.gcd, a.T)` measured ~20% faster than either -- but
    # only when the reduced axis is the short one, and proportionally slower
    # when it is the long one. A general helper should not have a cost that
    # inverts with the caller's array shape, so it is deliberately not used
    # here. Optimize an individual hot call site if a profile asks for it.
    #
    # No `np.abs` first: `np.gcd` is already sign-insensitive, and skipping the
    # copy is measurably cheaper on the ray matrices this runs on.
    return np.asarray(np.ufunc.reduce(np.gcd, np.asarray(a), axis=axis))


def primitive(a: Matrix | Vector, axis: int | None = None) -> np.ndarray:
    """
    **Description:**
    Divides integer vectors by their gcd, yielding primitive lattice vectors.

    Division is exact -- the gcd divides every entry by construction, so this
    is floor division, never a float round-trip. An all-zero vector has no
    primitive representative and is returned unchanged rather than producing
    a division by zero.

    **Arguments:**
    - `a`: An integer array.
    - `axis`: The axis along which each vector lies. `None` treats the whole
        array as a single vector.

    **Returns:**
    An array of the same shape as *a*, with each vector divided by its gcd.

    **Example:**
    ```python {3}
    import numpy as np
    from cytools.helpers.arithmetic import primitive
    primitive(np.array([[2, 4], [9, 6]]), axis=1)
    # array([[1, 2], [3, 2]])
    ```
    """
    a = np.asarray(a)
    gcds = gcd_reduce(a, axis=axis)

    # a zero vector has gcd 0; dividing it by 1 leaves it alone
    gcds = np.where(gcds == 0, 1, gcds)

    if axis is None:
        return a // gcds
    return a // np.expand_dims(gcds, axis)
