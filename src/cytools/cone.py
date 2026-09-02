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
# Description:  This module contains tools designed to perform cone
#               computations.
# -----------------------------------------------------------------------------
from __future__ import annotations

import itertools
import warnings

# 'standard' imports
from collections.abc import Callable, Iterable
from copy import deepcopy
from fractions import Fraction
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, cast, overload

import joblib
import latticepts
import numpy as np

# 3rd party imports
from flint import fmpq, fmpz, fmpz_mat
from ortools.sat.python import cp_model
from scipy import sparse
from scipy.optimize import linprog, nnls

import cytools.config as config
import cytools.utils as utils
from cytools._backends.arith import exact_rank, is_unimodular
from cytools._backends.engines import INTERIOR_POINT, STRETCHED_TIP
from cytools._backends.extremalrays import (
    exhaustive_indices as _extremal_indices,
)
from cytools._backends.extremalrays import (
    is_available as _extremalrays_available,
)
from cytools._backends.lp import SolverFailure
from cytools._backends.normaliz import hilbert_basis as _normaliz_hilbert_basis
from cytools._backends.ppl import ppl
from cytools._backends.registry import (
    CERTIFIES_INFEASIBLE,
    RECOVERABLE,
    EngineUnavailable,
)

# CYTools imports
from cytools._extensions import lazy_method
from cytools._typing import (
    ExtremalityMethod,
    ExtremalRaysMethod,
    InteriorPointBackend,
    Matrix,
    PointednessBackend,
    StretchedConeBackend,
    Vector,
)
from cytools.helpers.arithmetic import gcd_reduce


class Cone:
    """
    This class handles all computations relating to rational polyhedral cones,
    such cone duality and extremal ray computations. It is mainly used for the
    study of Kähler and Mori cones.

    :::important warning
    This class is primarily tailored to pointed (i.e. strongly convex) cones.
    There are a few computations, such as finding extremal rays, that may
    produce some unexpected results when working with non-pointed cones.
    :::

    ## Constructor

    ### `cytools.cone.Cone`

    **Description:**
    Constructs a `Cone` object. This is handled by the hidden
    [`__init__`](#__init__) function.

    **Arguments:**
    - `rays`: A list of rays that generates the cone. If it is not specified then the hyperplane normals must be specified.
    - `hyperplanes` *(array_like, optional)*: A list of inward-pointing
        hyperplane normals that define the cone. If it is not specified then the
        generating rays must be specified.
    - `check` *(bool, optional, default=True)*: Whether to check the input.
        Recommended if constructing a cone directly.

    :::note
    Exactly one of `rays` or `hyperplanes` must be specified. Otherwise an
    exception is raised.
    :::

    **Example:**
    We construct a cone in two different ways. First from a list of rays then
    from a list of hyperplane normals. We verify that the two inputs result in
    the same cone.
    ```python {2,3}
    from cytools import Cone
    c1 = Cone([[0,1],[1,1]]) # Create a cone using rays. It can also be done with Cone(rays=[[0,1],[1,1]])
    c2 = Cone(hyperplanes=[[1,0],[-1,1]]) # Create a cone using hyperplane normals.
    c1 == c2 # We verify that the two cones are the same.
    # True
    ```
    """

    if TYPE_CHECKING:
        # Signature for the lazily loaded vector-configuration method.
        from cytools.vector_config.vectorconfiguration import cone_vc as vc
    else:
        vc = lazy_method("cytools.vector_config.vectorconfiguration", "cone_vc")

    def __init__(
        self,
        rays: Matrix | None = None,
        hyperplanes: Matrix | None = None,
        parse_inputs: bool = True,
        check: bool = True,
        copy: bool = True,
        ambient_dim: int | None = None,
    ):
        """
        **Description:**
        Initializes a `Cone` object.

        **Arguments:**
        - `rays`: A list of rays that generates the cone. If it is not
            specified then the hyperplane normals must be specified.
        - `hyperplanes`: A list of inward-pointing hyperplane normals that
            define the cone. If it is not specified then the generating rays
            must be specified.
        - `check`: Whether to check the input. Recommended if constructing a
            cone directly.
        - `copy`: Whether to ensure we copy the input rays/hyperplanes.
            Recommended.
        - `ambient_dim`: The ambient dimension of the cone, if not inferrable.

        :::note
        Exactly one of `rays` or `hyperplanes` must be specified. Otherwise, an
        exception is raised.
        :::

        **Returns:**
        Nothing.

        **Example:**
        This is the function that is called when creating a new `Cone` object.
        We construct a cone in two different ways. First from a list of rays
        then from a list of hyperplane normals. We verify that the two inputs
        result in the same cone.
        ```python {2,3}
        from cytools import Cone
        c1 = Cone([[0,1],[1,1]]) # Create a cone using rays. It can also be done with Cone(rays=[[0,1],[1,1]])
        c2 = Cone(hyperplanes=[[1,0],[-1,1]]) # Create a cone using hyperplane normals.
        c1 == c2 # We verify that the two cones are the same.
        # True
        ```
        """
        # check whether rays or hyperplanes were input
        if not ((rays is None) ^ (hyperplanes is None)):
            raise ValueError(
                'Exactly one of "rays" and "hyperplanes" must be specified.'
            )

        # parse empty hyperplanes
        if (hyperplanes is not None) and (len(hyperplanes) == 0):
            hyperplanes = np.asarray(hyperplanes)

            # check if ambient dim is inferrable from hyperplanes
            if (len(hyperplanes.shape) > 1) and (hyperplanes.shape[1] != 0):
                # yes inferrable - ensure no conflicts in specification
                if (ambient_dim is not None) and (ambient_dim != hyperplanes.shape[1]):
                    raise ValueError(
                        f"Specified ambient dim = {ambient_dim} doesn't match inferrable shape from hyperplanes = {hyperplanes.shape[1]}..."
                    )

                ambient_dim = hyperplanes.shape[1]
            else:
                if ambient_dim is None:
                    raise ValueError(
                        "Must specify ambient dimension if len(hyperplanes)=0."
                    )

            # move to a ray representation
            hyperplanes = None
            rays = []
            for i in range(ambient_dim):
                # add e_i and -e_i
                rays.append([int(i == j) for j in range(ambient_dim)])
                rays.append([-int(i == j) for j in range(ambient_dim)])

        # minimal work if we don't parse the data
        if not parse_inputs:
            if rays is None:
                data_name = "hyperplane(s)"
                self._rays_were_input = False
                self._rays = None
                data = np.asarray(hyperplanes)
            else:
                raise NotImplementedError(
                    "Currently, parse_inputs is required if rays are input..."
                )

            # initialize other variables
            self.clear_cache()
            self._ambient_dim = data.shape[1]
            self._dim = None

            if self._rays_were_input:
                self._rays = data
            else:
                self._hyperplanes = data
            return

        # standard case
        if rays is None:
            data_name = "hyperplane(s)"
            self._rays_were_input = False
            self._rays = None
            if copy:
                data = np.array(hyperplanes)
            else:
                data = np.asarray(hyperplanes)
        else:
            data_name = "ray(s)"
            self._rays_were_input = True
            self._hyperplanes = None
            if copy:
                data = np.array(rays)
            else:
                data = np.asarray(rays)

        # initialize other variables
        self.clear_cache()

        # basic data-checking
        if len(data.shape) != 2:
            raise ValueError(f"Input {data_name} must be a 2D matrix.")
        elif data.shape[1] < 1:
            raise ValueError("Zero-dimensional cones are not supported.")
        # elif data.shape[0]<1:
        #    raise ValueError(f"At least one {data_name} is required.")

        self._ambient_dim = data.shape[1]

        if len(data):
            # check size of coordinates
            if np.min(data) <= -100000000000000:
                warnings.warn(
                    f"Extremely small coordinate, {np.min(data)}, "
                    f"found in {data_name}. Computations may be incorrect."
                )
            if np.max(data) >= +100000000000000:
                warnings.warn(
                    f"Extremely large coordinate, {np.max(data)}, "
                    f"found in {data_name}. Computations may be incorrect."
                )

            # parse input according to data type
            t = type(data[0, 0])
            if t in (fmpz, fmpq):
                if not config._exp_features_enabled:
                    raise Exception(
                        "Arbitrary precision data types only have "
                        "experimental support, so experimental "
                        "features must be enabled in configuration."
                    )
                if t == fmpz:
                    data = utils.array_fmpz_to_int(data)
                else:
                    data = utils.array_fmpq_to_float(data)
            elif t == np.int8:
                # rest of calculations assume ints are 64-bit? convert...
                data = data.astype(np.int64)
                t = np.int64
            elif t not in (np.int64, np.float64):
                raise NotImplementedError("Unsupported data type.")

            # reduce by GCD
            if check or t in (fmpz, np.float64):
                # get GCDs
                if t == np.int64:
                    gcds = gcd_reduce(data, axis=1)
                else:
                    gcds = np.asarray([utils.gcd_list(v) for v in data])

                # reduce by them
                if t == np.int64:
                    mask = gcds > 0
                    if False in mask:
                        warnings.warn("0 gcd found (row of zeros)... Skipping it!")
                    data = data[mask] // gcds[mask].reshape(-1, 1).astype(int)
                else:
                    mask = gcds >= 1e-5
                    if False in mask:
                        warnings.warn(
                            "Extremely small gcd found... "
                            "Computations may be incorrect!"
                        )
                    data = np.rint(data[mask] / gcds[mask].reshape(-1, 1)).astype(int)
            else:
                data = data.astype(int)

        # put data in correct variable
        if self._rays_were_input:
            self._rays = np.asarray(data)
            # Left lazy on purpose. The rank is an SVD of the ray matrix --
            # ~(1500 x h11) for a Mori cone at h11 ~ 200 -- and most callers,
            # including the Kahler-cone/tip path that dominates ensemble scans,
            # never ask for the dimension. `dimension()` computes and caches it
            # on first use.
            self._dim = None
        else:
            self._hyperplanes = np.asarray(data)
            self._dim = None

    def clear_cache(self):
        """
        **Description:**
        Clears the cached results of any previous computation.

        **Arguments:**
        None.

        **Returns:**
        Nothing.

        **Example:**
        We construct a cone, compute its extremal rays, clear the cache and
        then compute them again.
        ```python {5}
        c = Cone([[1,0],[1,1],[0,1]])
        c.extremal_rays()
        # array([[0, 1],
        #        [1, 0]])
        c.clear_cache() # Clears the cached result
        c.extremal_rays() # The extremal rays recomputed
        # array([[0, 1],
        #        [1, 0]])
        ```
        """
        self._hash = None
        self._dual = None
        self._ext_rays: list[np.ndarray | None] = [None, None]
        self._is_solid = None
        self._is_pointed = None
        self._is_simplicial = None
        self._is_smooth = None
        self._hilbert_basis = None
        self._face_lattice = None
        if self._rays_were_input:
            self._hyperplanes = None
        else:
            self._rays = None

    def __repr__(self):
        """
        **Description:**
        Returns a string describing the cone.

        **Arguments:**
        None.

        **Returns:**
        *(str)* A string describing the cone.

        **Example:**
        This function can be used to convert the Cone to a string or to print
        information about the cone.
        ```python {2,3}
        c = Cone([[1,0],[1,1],[0,1]])
        cone_info = str(c) # Converts to string
        print(c) # Prints cone info
        # A 2-dimensional rational polyhedral cone in RR^2 generated by 3 rays
        ```
        """
        if self._rays is not None:
            return (
                f"A {self.dim()}-dimensional rational polyhedral cone in "
                f"RR^{self._ambient_dim} generated by {len(self._rays)} "
                f"rays"
            )
        return (
            f"A rational polyhedral cone in RR^{self._ambient_dim} "
            f"defined by {len(self.hyperplanes())} hyperplanes"
        )

    def __eq__(self, other):
        """
        **Description:**
        Implements comparison of cones with ==.

        :::note
        The comparison of cones that are not pointed, and whose duals are also
        not pointed, is not supported.
        :::

        **Arguments:**
        - `other` *(Cone)*: The other cone that is being compared.

        **Returns:**
        *(bool)* The truth value of the cones being equal.

        **Example:**
        We construct two cones and compare them.
        ```python {3}
        c1 = Cone([[0,1],[1,1]])
        c2 = Cone(hyperplanes=[[1,0],[-1,1]])
        c1 == c2
        # True
        ```
        """
        if not isinstance(other, Cone):
            return NotImplemented

        if (
            self._rays is not None
            and other._rays is not None
            and sorted(self._rays.tolist()) == sorted(other._rays.tolist())
        ):
            # rays trivially match
            # N.B.: doesn't check for non-trivial equivalence. E.g.,
            # self._rays  = {e_1, -e_1, e2, -e_2}
            # other._rays = {e_1+e_2, -(e_1+e_2), e_1-e_2, -(e_1-e_2)}
            return True
        if (
            self._hyperplanes is not None
            and other._hyperplanes is not None
            and sorted(self._hyperplanes.tolist())
            == sorted(other._hyperplanes.tolist())
        ):
            # hyperplanes trivially match
            # N.B.: doesn't check for non-trivial equivalence. Same as above
            return True
        if self.is_pointed() ^ other.is_pointed():
            return False
        if self.is_pointed() and other.is_pointed():
            return sorted(self.extremal_rays().tolist()) == sorted(
                other.extremal_rays().tolist()
            )
        if self.dual().is_pointed() ^ other.dual().is_pointed():
            return False
        if self.dual().is_pointed() and other.dual().is_pointed():
            return sorted(self.dual().extremal_rays().tolist()) == sorted(
                other.dual().extremal_rays().tolist()
            )

        # ugly method... check if each ray self is contained in other
        # (and vice-versa)
        self_contained_in_other = np.all(
            other.hyperplanes() @ self.rays().transpose() >= 0
        )
        other_contained_in_self = np.all(
            self.hyperplanes() @ other.rays().transpose() >= 0
        )
        return self_contained_in_other and other_contained_in_self

    def __ne__(self, other):
        """
        **Description:**
        Implements comparison of cones with !=.

        :::note
        The comparison of cones that are not pointed, and whose duals are also
        not pointed, is not supported.
        :::

        **Arguments:**
        - `other` *(Cone)*: The other cone that is being compared.

        **Returns:**
        *(bool)* The truth value of the cones being different.

        **Example:**
        We construct two cones and compare them.
        ```python {3}
        c1 = Cone([[0,1],[1,1]])
        c2 = Cone(hyperplanes=[[1,0],[-1,1]])
        c1 != c2
        # False
        ```
        """
        if not isinstance(other, Cone):
            return NotImplemented
        return not self == other

    def __hash__(self):
        """
        **Description:**
        Implements the ability to obtain hash values from cones.

        :::note
        Cones that are not pointed, and whose duals are also not pointed, are
        not supported.
        :::

        **Arguments:**
        None.

        **Returns:**
        *(int)* The hash value of the cone.

        **Example:**
        We compute the hash value of a cone. Also, we construct a set and a
        dictionary with a cone, which make use of the hash function.
        ```python {2,3,4}
        c = Cone([[0,1],[1,1]])
        h = hash(c) # Obtain hash value
        d = {c: 1} # Create dictionary with cone keys
        s = {c} # Create a set of cones
        ```
        """
        if self._hash is not None:
            return self._hash
        if self.is_pointed():
            self._hash = hash(tuple(sorted(tuple(v) for v in self.extremal_rays())))
            return self._hash
        if self.dual().is_pointed():
            # Note: The minus sign is important because otherwise the dual cone
            # would have the same hash.
            self._hash = -hash(
                tuple(sorted(tuple(v) for v in self.dual().extremal_rays()))
            )
            return self._hash

        warnings.warn(
            "Cones that are not pointed and whose duals are also "
            "not pointed are assigned a hash value of 0."
        )
        return 0

    def ambient_dimension(self):
        """
        **Description:**
        Returns the dimension of the ambient lattice.

        **Arguments:**
        None.

        **Returns:**
        *(int)* The dimension of the ambient lattice.

        **Aliases:**
        `ambient_dim`.

        **Example:**
        We construct a cone and find the dimension of the ambient lattice.
        ```python {2}
        c = Cone([[0,1,0],[1,1,0]])
        c.ambient_dimension()
        # 3
        ```
        """
        return self._ambient_dim

    # aliases
    ambient_dim = ambient_dimension

    def dimension(self):
        """
        **Description:**
        Returns the dimension of the cone.

        **Arguments:**
        None.

        **Returns:**
        *(int)* The dimension of the cone.

        **Aliases:**
        `dim`.

        **Example:**
        We construct a cone and find its dimension.
        ```python {2}
        c = Cone([[0,1,0],[1,1,0]])
        c.dimension()
        # 2
        ```
        """
        if self._dim is not None:
            return self._dim

        if self._rays is not None:
            # know the rays... semi simple computation
            self._dim = exact_rank(self._rays)
        else:
            # don't know the rays... still simple if the cone is solid...
            if self.is_solid():
                self._dim = self.ambient_dim()
            else:
                # yikes need to compute the rays
                self._dim = exact_rank(self.rays())
        return self._dim

    # aliases
    dim = dimension

    def rays(self, use_extremal_hyperplanes: bool = False, verbosity: int = 0):
        """
        **Description:**
        Returns the (not necessarily extremal) rays that generate the cone.

        **Arguments:**
        - `use_extremal_hyperplanes`: Whether to use extremal hyperplanes in
            this computation, or just any hyperplanes.
        - `verbosity`: The verbosity level.

        **Returns:**
        *(numpy.ndarray)* The list of rays that generate the cone.

        **Example:**
        We construct two cones and find their generating rays.
        ```python {3,6}
        c1 = Cone([[0,1],[1,1]])
        c2 = Cone(hyperplanes=[[0,1],[1,1]])
        c1.rays()
        # array([[0, 1],
        #        [1, 1]])
        c2.rays()
        # array([[ 1,  0],
        #        [-1,  1]])
        ```
        """
        if self._rays is not None:
            return np.array(self._rays)
        if self._ambient_dim >= 12 and len(self.hyperplanes()) != self._ambient_dim:
            warnings.warn(
                "This operation might take a while for d > ~12 "
                "and is likely impossible for d > ~18."
            )

        # select the hyperplanes
        if use_extremal_hyperplanes:
            H = self.extremal_hyperplanes()
        else:
            H = self.hyperplanes()

        # compute the rays
        rays = dualize(H, verbosity=verbosity)

        # save/return
        if verbosity >= 1:
            print("Saving the rays...", flush=True)
        self._rays = np.asarray(rays, dtype=int)
        self._dim = None  # lazy; see the note in __init__
        return np.array(self._rays)

    def hyperplanes(self, use_extremal_rays: bool = False, verbosity: int = 0):
        """
        **Description:**
        Returns the inward-pointing normals to the hyperplanes that define the
        cone.

        **Arguments:**
        - `use_extremal_rays` :Whether to use extremal rays in this
            computation, or just any rays.
        - `verbosity`: The verbosity level.

        **Returns:**
        *(numpy.ndarray)* The list of inward-pointing normals to the
        hyperplanes that define the cone.

        **Example:**
        We construct two cones and find their hyperplane normals.
        ```python {3,6}
        c1 = Cone([[0,1],[1,1]])
        c2 = Cone(hyperplanes=[[0,1],[1,1]])
        c1.hyperplanes()
        # array([[ 1,  0],
        #        [-1,  1]])
        c2.hyperplanes()
        # array([[0, 1],
        #        [1, 1]])
        ```
        """
        if self._hyperplanes is not None:
            return np.array(self._hyperplanes)
        if self._ambient_dim >= 12 and len(self.rays()) != self._ambient_dim:
            warnings.warn(
                "This operation might take a while for d > ~12 "
                "and is likely impossible for d > ~18."
            )

        # select the rays
        if use_extremal_rays:
            R = self.extremal_rays()
        else:
            R = self.rays()

        # compute the hyperplanes
        H = dualize(R, verbosity=verbosity)

        # save/return
        if verbosity >= 1:
            print("Saving the hyperplanes...", flush=True)
        self._hyperplanes = np.asarray(H, dtype=int)
        if len(self._hyperplanes) == 0:
            self._hyperplanes = np.zeros((0, self._ambient_dim), dtype=int)
        return np.array(self._hyperplanes)

    def contains(self, other, eps: float = 0) -> bool | tuple[bool, ...]:
        """
        **Description:**
        Checks if a point is in the (strict) interior.

        **Arguments:**
        - `other`: The object to check containment of. Can be a 1D array, which
            is treated as a point. Can be a 2D array, which is treated as a
            list of points. Can be a Cone.
        - `eps`: Check H@pt >= eps.

        **Returns:**
        Whether pt is in the (strict) interior.
        """
        if isinstance(other, Cone):
            # just check if we contain all of other's rays...
            contained = self.contains(other.rays(), eps=eps)
            return all(contained) if isinstance(contained, tuple) else bool(contained)

        # other was a point(s)
        H = self.hyperplanes()
        pt = np.array(other)

        # cast to 2D array, transpose
        if len(pt.shape) == 1:
            pt = pt.reshape(-1, 1)
            return_list = False
        else:
            # transpose so columns are points
            pt = pt.transpose()
            return_list = True

        # compute which points are in the cone
        if len(H):
            contained = np.all(H @ pt >= eps, axis=0)
        else:
            contained = [True for _ in range(pt.shape[1])]

        # return
        if return_list:
            return tuple(contained)
        return contained[0]

    def dual_cone(self):
        """
        **Description:**
        Returns the dual cone.

        **Arguments:**
        None.

        **Returns:**
        *(Cone)* The dual cone.

        **Aliases:**
        `dual`.

        **Example:**
        We construct a cone and find its dual cone.
        ```python {2,4}
        c = Cone([[0,1],[1,1]])
        c.dual_cone()
        # A rational polyhedral cone in RR^2 defined by 2 hyperplanes normals
        c.dual_cone().rays()
        # array([[ 1,  0],
        #        [-1,  1]])
        ```
        """
        if self._dual is None:
            if self._rays is not None:
                self._dual = Cone(hyperplanes=self.rays(), check=False)
            else:
                self._dual = Cone(rays=self.hyperplanes(), check=False)
            self._dual._dual = self
        return self._dual

    # aliases
    dual = dual_cone

    def extremal_rays(
        self,
        tol: float = 1e-4,
        minimal: bool = True,
        method: ExtremalRaysMethod = "extremalrays",
        verbose: bool = False,
    ) -> np.ndarray:
        """
        **Description:**
        Returns the extremal rays of the cone.

        :::note
        By default, this function will use as many CPU threads as there are
        available. To fix the number of threads, you can set the `n_threads`
        variable in the `config` submodule.
        :::

        **Arguments:**
        - `tol`: Specifies the tolerance for deciding whether a ray is extremal
            or not. Only used if method=="nnls".
        - `minimal`: Whether to return a minimal generating set of rays. For
            pointed cones, there is a unique minimal generating set -- the
            extremal rays. For non-pointed cones, one can have a collection of
            extremal rays generating the cone that is not minimal with respect
            to ray count.
        - `method`: The backend used to prune the rays. One of
            "extremalrays" (default; needs a pointed cone and the extremalrays
            package, else falls back to "legacy"), "legacy", or "nnls". "lp"
            is a synonym for "legacy".
        - verbose: When set to True it show the progress while finding the
            extremal rays.

        **Returns:**
        The list of extremal rays of the cone.

        **Example:**
        We construct a cone and find its extremal_rays.
        ```python {2}
        c = Cone([[0,1],[1,1],[1,0]])
        c.extremal_rays()
        # array([[0, 1],
        #        [1, 0]])
        ```
        """
        if self._ext_rays[minimal] is not None:
            return np.array(self._ext_rays[minimal])

        # non-pointed cones are tricky
        # A ray r of the ray set (i.e., generating matrix) R is extremal if it
        # cannot be written as a non-negative combination of the other rays
        #
        # For pointed cones, there is a unique collection of extremal rays
        # defining a cone. For non-pointed cones, this is not true.
        #
        # Furthermore, for non-pointed cones, every ray r of R may be extremal
        # with respect to R, but there might be a smaller set of rays R'
        # defining the same region.
        #
        # For simplicity, we return minimal (in terms of ray count) generating
        # matrices by analyzing the lineality space and the pointed bit of the
        # cone separately
        if minimal and (not self.is_pointed()):
            ext_rays = np.vstack(
                [
                    self.lineality_space().extremal_rays(),
                    self.pointed_space().extremal_rays(),
                ]
            )
            self._ext_rays[minimal] = ext_rays

            return ext_rays

        # It is important to delete duplicates
        rays = np.array(list({tuple(r) for r in self.rays()}))

        # a cone with no rays (e.g. the pointed part of a whole linear space)
        # has none that are extremal; np.array([]) is 1d, so restore the shape
        if rays.shape[0] == 0:
            ext_rays = rays.reshape(0, self.ambient_dim())
            self._ext_rays[minimal] = ext_rays
            return ext_rays

        # if only 1 ray, this is trivial
        if rays.shape[0] == 1:
            self._ext_rays[minimal] = rays
            if self._rays is None:
                self._rays = rays

            return rays

        # "lp" named the per-ray backend before extremalrays became default
        if method == "lp":
            method = "legacy"
        if method not in ("extremalrays", "legacy", "nnls"):
            raise ValueError(
                f"Unknown method '{method}'; expected "
                "'extremalrays', 'legacy' or 'nnls'."
            )

        if method == "extremalrays":
            if not _extremalrays_available():
                warnings.warn(
                    "The extremalrays package is not installed, so the slower "
                    "legacy backend is being used. Install it with "
                    "'pip install extremalrays', or pass method='legacy' to "
                    "silence this warning."
                )
                method = "legacy"
            elif not self.is_pointed():
                # the sweep needs a pointed cone; one only gets here with
                # minimal=False, since minimal=True is decomposed above
                method = "legacy"

        if method == "extremalrays":
            if verbose:
                print(
                    f"Computing extremal rays for a cone with {len(rays)} "
                    "rays using extremalrays..."
                )
            # Its own tolerance is a separation tolerance, unrelated to the
            # nnls tolerance above.
            keep = _extremal_indices(rays, verbose=verbose)
            ext_rays = rays[keep]
            self._ext_rays[minimal] = ext_rays
            if self._rays is None:
                self._rays = ext_rays

            return ext_rays

        # the per-ray backend. is_extremal takes "lp" or "nnls"
        check_method = "nnls" if method == "nnls" else "lp"

        # configure threads
        n_threads = config.n_threads
        if n_threads is None:
            if rays.shape[0] < 32 or not self.is_pointed():
                n_threads = 1
            else:
                n_threads = cpu_count()
        elif n_threads > 1 and not self.is_pointed():
            warnings.warn(
                "When finding the extremal rays of a non-pointed "
                "cone in parallel, there can be conflicts that end up "
                "producing erroneous results. It is highly recommended to "
                "use a single thread."
            )

        # compute the extremal rays
        ext_rays = [True for _ in range(len(rays))]
        to_check = list(range(len(rays)))

        if verbose:
            print(
                f"Computing extremal rays for a cone with {len(rays)} using {n_threads} threads..."
            )

        # bound the retries; unbounded ones spin forever on a ray that
        # deterministically fails
        max_attempts = 3
        attempts = {}

        # one ray per worker per round: ext_rays is only pruned between
        # rounds, so bigger batches re-check known-redundant rays
        # (4*n_threads took >10 min on a 908x125 cone, vs 7 s here)
        batch_size = n_threads

        def learn(results, checking):
            """Record verdicts; queue failures for a bounded retry."""
            for i, extremalQ, err in results:
                if err is None:
                    ext_rays[i] = extremalQ
                    continue
                attempts[i] = attempts.get(i, 0) + 1
                if attempts[i] >= max_attempts:
                    raise RuntimeError(
                        f"Failed to check whether ray #{i} was extremal after "
                        f"{max_attempts} attempts. (Last error was: {err})"
                    )
                to_check.append(i)
                if verbose:
                    print(f"Failed to check whether ray #{i} was extremal")
                    print(f"(Error was: {err})")
                    print("(Putting it at the end and retrying later...)")

        if n_threads == 1:
            # skip the pool; with one worker the serialization is pure
            # overhead
            while len(to_check):
                checking = to_check[:batch_size]
                to_check = to_check[batch_size:]
                learn(
                    [
                        is_extremal(rays, i, ext_rays, method=check_method, tol=tol)
                        for i in checking
                    ],
                    checking,
                )
        else:
            # one context for the whole loop: joblib auto-memmaps args
            # over 1 MB, so a new context per round re-dumped the rays each
            # round; held open it dumps once and keeps the workers warm
            with joblib.Parallel(n_jobs=n_threads) as parallel:
                while len(to_check):
                    checking = to_check[:batch_size]
                    to_check = to_check[batch_size:]
                    learn(
                        parallel(
                            joblib.delayed(is_extremal)(
                                rays, i, ext_rays, method=check_method, tol=tol
                            )
                            for i in checking
                        ),
                        checking,
                    )

        # save the answer
        ext_rays = rays[list(ext_rays)]
        self._ext_rays[minimal] = ext_rays
        if self._rays is None:
            self._rays = ext_rays

        return ext_rays

    def extremal_hyperplanes(
        self,
        tol: float = 1e-4,
        minimal=True,
        method="extremalrays",
        verbose: bool = False,
    ) -> np.ndarray:
        """
        **Description:**
        Returns the extremal hyperplanes of the cone.

        **Arguments:**
        - `tol`: Specifies the tolerance for deciding whether a hyperplane is
            extremal or not. Only used if method=="nnls".
        - `minimal`: Whether to return a minimal generating set of hyperplane.
            For duals of pointed cones, there is a unique minimal generating
            set -- the extremal hyperplanes. For non-pointed cones, one can
            have a collection of extremal hyperplanes defining the cone that is
            not minimal with respect to hyperplane count.
        - `method`: The backend used to prune the hyperplanes; see
            `extremal_rays`, to which this delegates on the dual cone.
        - verbose: When set to True it show the progress while finding the
            extremal hyperplanes.

        **Returns:**
        The list of extremal hyperplanes of the cone.
        """
        return self.dual().extremal_rays(
            tol=tol, minimal=minimal, method=method, verbose=verbose
        )

    @overload
    def face_lattice(
        self, codim: int, include_self: bool = False, verbosity: int = 0
    ) -> tuple[Cone, ...]: ...

    @overload
    def face_lattice(
        self, codim: None = None, include_self: bool = False, verbosity: int = 0
    ) -> tuple[tuple[Cone, ...], ...]: ...

    def face_lattice(
        self, codim: int | None = None, include_self: bool = False, verbosity: int = 0
    ) -> tuple[Cone, ...] | tuple[tuple[Cone, ...], ...]:
        """
        **Description:**
        Computes the positive-dimensional face lattice of a pointed cone.

        The faces are organized in a tuple of increasing codim. This method is
        distinct from `facets` since this will be a lot slower for high-dim
        H-cones.

        **Arguments:**
        - `codim`: Optional codim of the desired faces. When set to `0`, returns
            the cone itself.
        - `include_self`: Whether to include the codim-0 face when returning all
            faces.
        - `verbosity`: The verbosity level.

        **Returns:**
        A tuple of `Cone` objects of codimension `codim`, if specified.
        Otherwise, a tuple of tuples of cone faces.
        """
        dim = self.dim()

        # input guard
        if (codim is not None) and ((codim < 0) or (codim > dim)):
            raise ValueError(f"Cone does not have faces of codimension {codim}")

        if dim > 20:
            warnings.warn("Getting the face lattice for high-dim cones is expensive")

        # easy answers
        if codim == 0:
            return (self,)
        if codim == dim:
            if self.is_pointed():
                # return cone spanned by 0 rays
                I = np.eye(self.ambient_dim(), dtype=int)
                return (Cone(hyperplanes=np.vstack([I, -I])),)
            # return empty list... no 0D faces here
            return tuple()

        # fast track if cached
        if self._face_lattice is not None:
            return (
                self._face_lattice[codim]
                if codim is not None
                else (self._face_lattice if include_self else self._face_lattice[1:])
            )

        if not self.is_pointed():
            raise NotImplementedError(
                "Cone.face_lattice() currently supports only pointed cones."
            )

        if verbosity >= 1:
            print(
                "Computing cone face lattice via extremal ray/hyperplane incidence..."
            )

        # expensive work vvv
        R = self.extremal_rays()
        H = self.extremal_hyperplanes()
        # expensive work ^^^

        # compute the incidences
        if self.is_solid():
            can_saturate = H
        else:
            can_saturate = np.array(
                [h for h in H if not self.dual().contains(-h)],
                dtype=int,
            )

        facet_ray_sets = set()
        for h in can_saturate:
            ray_inds = frozenset(
                i
                for i, r in enumerate(R)
                if sum(int(a) * int(b) for a, b in zip(h, r)) == 0
            )
            if ray_inds:
                facet_ray_sets.add(ray_inds)

        seen = set(facet_ray_sets)
        frontier = list(facet_ray_sets)
        while frontier:
            current = frontier.pop()
            for facet in facet_ray_sets:
                inter = current & facet
                if inter and inter not in seen:
                    seen.add(inter)
                    frontier.append(inter)

        face_sets = [[] for _ in range(dim)]
        face_objects = {}
        for ray_inds in sorted(seen, key=lambda inds: tuple(sorted(inds))):
            face_rays = R[list(ray_inds)]
            face_dim = exact_rank(face_rays)
            if face_dim <= 0:
                continue

            face_codim = dim - face_dim
            if face_codim in (0, dim):
                continue

            face_sets[face_codim].append(ray_inds)
            face_objects[ray_inds] = Cone(rays=face_rays, check=False)

        face_lattice = [(self,)]
        for face_codim in range(1, dim):
            codim_faces = tuple(
                face_objects[ray_inds]
                for ray_inds in sorted(
                    face_sets[face_codim], key=lambda inds: tuple(sorted(inds))
                )
            )
            face_lattice.append(codim_faces)

        # add the 0D cone if this is pointed
        if self.is_pointed():
            I = np.eye(self.ambient_dim(), dtype=int)
            face_lattice.append((Cone(hyperplanes=np.vstack([I, -I])),))

        # cache and return
        self._face_lattice = tuple(face_lattice)
        return (
            self._face_lattice[codim]
            if codim is not None
            else (self._face_lattice if include_self else self._face_lattice[1:])
        )

    def facets(self, verbosity: int = 0):
        """
        **Description:**
        Get the facets of the cone.

        This is easy if:
            -) the cone is simplicial OR
            -) the cone is solid and the extremal hyperplanes can be computed.
        Otherwise, the computation uses both rays and hyperplanes... this is
        semi-expensive to compute...

        **Arguments:**
        - `verbosity`: The verbosity level.

        **Returns:**
        The facets of the cone.
        """
        # ray-based computation
        if self.is_simplicial():
            if verbosity >= 1:
                print("Cone is simplicial! Easy computation...")
            R = self.extremal_rays()

            dim = len(R)

            if dim == 1:
                I = np.eye(self.ambient_dim(), dtype=int)
                return [
                    Cone(hyperplanes=np.vstack([I, -I])),
                ]
            if dim == 0:
                return []

            ray_inds = list(range(dim))

            # facets are defined by collections of #(dim-1) rays
            return [
                Cone(rays=R[list(inds)])
                for inds in itertools.combinations(ray_inds, dim - 1)
            ]

        # hyperplane based-computation
        if verbosity >= 1:
            print("Computing facets via extremal hyperplanes...")
        H = self.extremal_hyperplanes()

        if self.is_solid():
            # still pretty easy
            can_saturate = H
        else:
            # this means that the cone contains both h and -h as hyperplanes...
            # i.e., h is already saturated by definition...
            # need to skip these when looking to saturate hyperplanes
            can_saturate = [h for h in H if not self.dual().contains(-h)]

        return [Cone(hyperplanes=np.vstack((H, -h)), check=False) for h in can_saturate]

    def tip_of_stretched_cone(
        self,
        c: float = 1,
        backend: StretchedConeBackend | None = None,
        check: bool = True,
        constraint_error_tol: float = 5e-2,
        max_iter: int = 10**6,
        show_hints: bool = True,
        verbose: bool = False,
    ) -> np.ndarray | None:
        r"""
        **Description:**
        Finds the tip of the stretched cone. The stretched cone is defined as
        the region where pairing with every primitive integral facet normal is
        at least `c`. Its tip is the point in this region with smallest norm.
        This lattice-normal convention is not Euclidean distance to a facet,
        which would additionally scale each constraint by the normal's norm.

        :::note
        This is a quadratic program: the norm of a vector is minimized subject
        to linear constraints. HiGHS is the certified automatic engine. OSQP,
        Mosek, and CVXOPT remain explicit compatibility and differential-test
        choices.

        Both solve the stated problem. An LP over the same feasible region
        would be cheaper but returns *some* point of the stretched cone rather
        than the minimum-norm one, which is a different mathematical object;
        that is why no LP engine is registered for this task.
        :::

        **Arguments:**
        - `c` *(float)*: A real positive number specifying the stretching of
            the cone: the minimum pairing with each primitive facet normal.
        - `backend`: Optional compatibility selector. The quadratic engines
            are `"highs"`, `"mosek"`, `"osqp"`, and `"cvxopt"`; when omitted,
            the registry selects a recoverable engine whose negative result is
            exactly verified.
        - `check` *(bool, optional, default=True)*: Flag that specifies whether
            to check if the output of the optimizer is consistent and satisfies
            `constraint_error_tol`.
        - `constraint_error_tol` *(float, optional, default=5e-2)*: Error
            tolerance for the linear constraints.
        - `max_iter` *(int, optional, default=10**6)*: Maximum solver
            iterations (maximum permissible value: 2**31-1).
        - `show_hints`: Whether to print diagnostic hints when no tip is found.
        - `verbose` *(boolean, optional)*: Whether to print extra diagnostic
            information (True) or not (False).

        **Returns:**
        *(numpy.ndarray)* The vector specifying the location of the tip.
            Automatic selection returns `None` only when the stretched region
            is exactly proved empty. An explicitly selected legacy solver may
            also return `None` when it does not converge.

        **Example:**
        We construct two cones and find the locations of the tips of the
        stretched cones.
        ```python {3,5}
        c1 = Cone([[1,0],[0,1]])
        c2 = Cone([[3,2],[5,3]])
        c1.tip_of_stretched_cone(1)
        # array([1., 1.])
        c2.tip_of_stretched_cone(1)
        # array([8., 5.])
        ```
        """
        # find the tip of the stretched cone
        hp = self.hyperplanes()
        if len(hp) == 0:
            # trivial
            return np.ones(self._ambient_dim)

        problem = {"dim": self.ambient_dim(), "rows": len(hp)}
        # "glop" names an LP engine, which minimises a linear functional and
        # so returns *a* point of the stretched cone rather than its
        # minimum-norm point. "highs" used to be refused alongside it, but
        # that was a statement about how `lp.py` calls HiGHS: HiGHS solves
        # convex QPs too, and `qp.highs_tip` is now the default engine here.
        if backend == "glop":
            raise ValueError(
                f"backend={backend!r} solves LP feasibility, not the "
                "minimum-norm quadratic problem. Use find_interior_point() "
                "for an arbitrary point in the stretched cone."
            )
        if backend is None:
            engine = STRETCHED_TIP.resolve(
                need=(CERTIFIES_INFEASIBLE, RECOVERABLE), problem=problem
            )
        else:
            try:
                engine = STRETCHED_TIP.select(backend, problem)
            except EngineUnavailable:
                if backend == "cvxopt":
                    raise ImportError(
                        "The CVXOPT backend is optional. Install it with "
                        '`python -m pip install "cytools-workbench[cvxopt]"`.'
                    ) from None
                raise

        solution = engine.run(hp, c, max_iter, verbose)
        G = -1 * sparse.csc_matrix(hp, dtype=float)

        # parse solution
        if solution is None:
            if show_hints and CERTIFIES_INFEASIBLE not in engine.provides:
                print("Calculated 'solution' was None...", end=" ")
                print("some potential reasons why:")

                print(f"-) maybe max_iter={max_iter} was too low?")

                if (self.ambient_dim() >= 25) and (engine.name != "mosek"):
                    print(
                        f"-) given the high dimension, {self.ambient_dim()},",
                        end=" ",
                    )
                    print(
                        f"and engine={engine.name}, this could be a numerical",
                        end=" ",
                    )
                    print("issue. Mosek handles these better; see")
                    print("   cytools.config.available_engines().")

                # scaling
                print(
                    f"-) if the cone is narrow, try decreasing c from {c}",
                    end=" ",
                )
                print(
                    "(you can then scale up the tip to hit the desired stretching...)"
                )

                print("For more info, re-run with verbose=True")
            return None
        if check:
            res = max(G.dot(solution)) + c
            if res > constraint_error_tol:
                warnings.warn(
                    f"The solution that was found is invalid: {res} > {constraint_error_tol}"
                )
                return None
        return solution

    def find_grading_vector(
        self, backend: InteriorPointBackend | None = None
    ) -> np.ndarray | None:
        r"""
        **Description:**
        Finds a grading vector for the cone, i.e. a vector $\mathbf{v}$ such
        that any non-zero point in the cone $\mathbf{p}$ has a positive dot
        product $\mathbf{v}\cdot\mathbf{p}>0$. Thus, the grading vector must be
        strictly interior to the dual cone, so it is only defined for pointed
        cones. This function returns an integer grading vector.

        **Arguments:**
        - `backend`: Optional compatibility selector for the interior-point
            computation. Automatic engine selection is preferred.

        **Returns:**
        *(numpy.ndarray)* A grading vector. If it could not be found then None
            is returned.

        **Example:**
        We construct a cone and find a grading vector.
        ```python {2}
        c = Cone([[3,2],[5,3]])
        c.find_grading_vector()
        # array([-1,  2])
        ```
        """
        if not self.is_pointed():
            raise Exception("Grading vectors are only defined for pointed cones.")
        return self.dual().find_interior_point(integral=True, backend=backend)

    def find_interior_point(
        self,
        c: float = 1,
        lower: float | None = None,
        integral: bool = False,
        backend: InteriorPointBackend | None = None,
        check: bool = True,
        show_hints: bool = False,
        verbose: bool = False,
    ) -> np.ndarray | None:
        r"""
        **Description:**
        Finds a point in the strict interior of the cone. If no point is found
        then None is returned.

        **Arguments:**
        - `c`: A real positive number specifying the stretching of the cone
            (the minimum pairing with each primitive facet normal). Only used
            if rays are not known.
        - `lower`: A lower bound on the components of the interior point.
        - `integral`: A flag that specifies whether the point should have
            integral coordinates.
        - `backend`: Optional compatibility selector. Automatic selection
            chooses an engine that can distinguish solver failure from proved
            infeasibility.
        - `check`: Whether to verify that the point is inside the cone.
        - `show_hints`: Unused; retained for call compatibility.
        - `verbose`: Whether to print diagnostic information.

        **Returns:**
        A point in the strict interior of the cone. `None` means the cone was
        *proved* to have empty interior, never that a solver gave up: the
        engine is resolved with `CERTIFIES_INFEASIBLE`, and one that cannot
        reach a conclusion raises `SolverFailure` instead. `Cone.is_solid`
        depends on that distinction.

        **Example:**
        We construct a cone and find some interior points.
        ```python {2,4}
        c = Cone([[3,2],[5,3]])
        c.find_interior_point()
        # array([4. , 2.5])
        c.find_interior_point(integral=True)
        # array([8, 5])
        ```
        """
        # If the rays are already computed then this is a simple task
        if (self._rays is not None) and (backend is None) and (lower is None):
            if exact_rank(self._rays) != self._ambient_dim:
                return None

            point = self._rays.sum(axis=0)

            if max(abs(point)) > 1e-3:
                point //= utils.gcd_list(point)
            else:
                # looks like the point is all zeros
                if np.prod(self.hyperplanes().shape) == 0:
                    # trivial cone... all space
                    point = [0 for _ in range(self._ambient_dim)]
                    point[0] = 1
                    return np.asarray(point)
                raise Exception(
                    f"Unexpected error in finding point in cone with rays = {self._rays}"
                )

            if not integral:
                point = point / len(self._rays)

            return point

        # Otherwise we need to do a harder computation...
        H = self.hyperplanes()

        # CERTIFIES_INFEASIBLE is required, not preferred. `is_solid` reads a
        # None return from here as "the cone has empty interior", so an engine
        # that cannot tell infeasibility from its own failure would turn a
        # numerical problem into a false geometric claim.
        if backend in ("mosek", "osqp", "cvxopt"):
            if lower is not None:
                raise ValueError(
                    f"Cannot set a custom lower bound for backend={backend!r}."
                )
            solution = self.tip_of_stretched_cone(
                c,
                backend=backend,
                show_hints=show_hints,
                verbose=verbose,
            )
        else:
            problem = {"dim": self._ambient_dim, "rows": len(H)}
            if backend is None:
                engine = INTERIOR_POINT.resolve(
                    need=(CERTIFIES_INFEASIBLE, RECOVERABLE), problem=problem
                )
            else:
                # Resolve explicit compatibility choices through the same
                # guarantee gate as automatic selection. In particular, SCIP
                # and CP-SAT cannot turn "no integer point in this box" into a
                # proof that a rational cone has empty interior.
                with config.engines(interior_point=backend):
                    engine = INTERIOR_POINT.resolve(
                        need=(CERTIFIES_INFEASIBLE, RECOVERABLE), problem=problem
                    )
            solution = engine.run(H, c, self._ambient_dim, lower, verbose)
        if solution is None:
            if backend in ("mosek", "osqp", "cvxopt"):
                raise SolverFailure(
                    f"The {backend!r} quadratic solver returned no point; "
                    "this is not a proof that the cone has empty interior."
                )
            return None

        # Containment test for every hyperplane at once. Iterating the rows in
        # Python cost one tiny numpy call each -- ~1500 of them per cone at
        # h11 ~ 200, and up to 1000x that inside the rounding retry below.
        if isinstance(H, (list, np.ndarray)):
            H_arr = np.asarray(H)

            def all_positive(x):
                return bool(np.all(H_arr.dot(x) > 0))

        else:
            # Sparse rows (dicts of index -> value); no dense array to build.
            def all_positive(x):
                return all(sum(val * x[ind] for ind, val in hp.items()) > 0 for hp in H)

        # Make sure that the solution is valid
        if check and not all_positive(solution):
            raise SolverFailure(
                "The interior-point engine returned a point outside the "
                "strict interior."
            )

        # Finally, round to an integer if necessary
        if integral:
            n_tries = 1000
            for i in range(1, n_tries):
                int_sol = np.rint(i * np.asarray(solution)).astype(int)
                if all_positive(int_sol):
                    break
                if i == n_tries - 1:
                    raise SolverFailure(
                        "Could not convert the feasible point to a strict "
                        "integral interior point after 999 rescalings."
                    )
            solution = int_sol

        return solution

    def find_lattice_points(
        self,
        min_points: int | None = None,
        max_deg: int | None = None,
        grading_vector: Vector | None = None,
        c: int = 0,
        max_coord: int | None = None,
        deg_window: int = 0,
        filter_function: Callable | None = None,
        process_function: Callable | None = None,
        fast_mode: bool = True,
        max_B: int = 10000,
        verbose: bool = False,
    ):
        """
        **Description:**
        Finds lattice points in the cone. The points are found in the region
        bounded by the cone, and by a cutoff surface given by the grading
        vector. Note that this requires the cone to be pointed. The minimum
        number of points to find can be specified, or if working with a
        preferred grading vector it is possible to specify the maximum degree.

        **Arguments:**
        - `min_points` *(int, optional)*: Specifies the minimum number of points
            to find. The degree will be increased until this minimum number is
            achieved.
        - `max_deg` *(int, optional)*: The maximum degree of the points to
            find. This is useful when working with a preferred grading.
        - `grading_vector` *(array_like, optional)*: The grading vector that
            will be used. If it is not specified then it is computed.
        - `c` *(numeric or array_like, optional)*: The minimum allowed
            stretching. Can be a single number or a stretching per each
            hyperplane (applied in the order of self.hyperplanes()).
        - `max_coord` *(int, optional)*: The maximum magnitude of the
            coordinates of the points. When not specified, the CP-SAT search
            uses its largest supported integer bound.
        - `deg_window` *(int, optional)*: If using min_points, search for
            lattice points with degrees in range [n*(deg_window+1),
            n*(deg_window+1)+deg_window] for 0<=n
        - `filter_function` *(function, optional)*: A function to use as a
            filter of the points that will be kept. It should return a boolean
            indicating whether to keep the point. Note that `min_points` does
            not take the filtering into account.
        - `process_function` *(function, optional)*: A function to process the
            points as they are found. This is useful to avoid first constructing
            a large list of points and then processing it.
        - `fast_mode` *(bool, optional)*: Allow quicker lattice point
            computations for small cones. Doesn't use degree-based methods.
            Instead uses Linf norm.
        - `max_B`: *(int, optional)*: Max Linf norm allowed in fast_mode.
        - `verbose` *(boolean, optional)*: Whether to print extra diagnostic
            information (True) or not (False).

        **Returns:**
        *(numpy.ndarray)* The list of points.

        **Example:**
        We construct a cone and find at least 20 lattice points in it.
        ```python {2}
        c = Cone([[3,2],[5,3]])
        pts = c.find_lattice_points(min_points=20)
        print(len(pts)) # We see that it found 21 points
        # 21
        ```
        Let's also give an example where we use a function to apply some
        filtering. This can be something very complicated, but here we just
        pick the points where all coordinates are odd.
        ```python {5}
        def filter_function(pt):
            return all(c%2 for c in pt)

        c = Cone([[3,2],[5,3]])
        pts = c.find_lattice_points(min_points=20, filter_function=filter_function)
        print(len(pts)) # Now we get only 6 points instead of 21
        # 6
        ```
        Finally, let's give an example where we process the data as it comes
        instead of first constructing a list. In this simple example we just
        print each point with odd coordinates, but in general it can be a
        complex algorithm.
        ```python {6}
        def process_function(pt):
            if all(c%2 for c in pt):
                print(f"Processing point {pt}")

        c = Cone([[3,2],[5,3]])
        c.find_lattice_points(min_points=20, process_function=process_function)
        # Processing point (5, 3)
        # Processing point (11, 7)
        # Processing point (15, 9)
        # Processing point (17, 11)
        # Processing point (21, 13)
        # Processing point (25, 15)
        ```
        """
        # initial checks
        if max_deg is None and min_points is None:
            raise Exception(
                "Either the maximum degree or the minimum number of points must be specified."
            )

        # shortcut if min_points is set and dim is low
        if fast_mode and (min_points is not None) and (self.ambient_dim() <= 10):
            return np.array(
                latticepts.enum_lattice_points(
                    H=self.hyperplanes(),
                    rhs=c,
                    min_N_pts=min_points,
                    max_B=max_B,
                )
            )

        if not self.is_pointed():
            raise Exception("Only pointed cones are currently supported.")

        if process_function is not None and filter_function is not None:
            raise Exception(
                "Only one of filter_function or process_function can be specified."
            )
        if grading_vector is None:
            grading_vector = self.find_grading_vector()
        if grading_vector is None:
            raise RuntimeError("Could not find a grading vector for this cone.")
        finite_max_coord = max_coord is not None
        coord_bound = max_coord if max_coord is not None else cp_model.INT32_MAX - 1

        hp = self.hyperplanes()

        # We start by defining a class that will store the points we find.
        # Which work happens per solution is fixed up front, in `on_solution`,
        # rather than re-tested inside the callback: this runs once per
        # solution found, so the conditions -- which never change during a
        # search -- must not be re-checked each time.
        class SolutionStorage(cp_model.CpSolverSolutionCallback):
            def __init__(self, variables, on_solution):
                super().__init__()
                self._variables = variables
                self._solutions = set()
                self._on_solution = on_solution
                self._n_sol = 0

            def on_solution_callback(self) -> None:
                self._on_solution(self)

        # This first variant is for when we want to check that it is a pointed
        # cone with a good grading vector
        class MoreThanOneSolution(Exception):
            pass

        def on_soln_single_pt(store) -> None:
            store._n_sol += 1
            if store._n_sol > 1:
                raise MoreThanOneSolution

        # This one is the standard one that will be used
        def on_soln_default(store) -> None:
            store._n_sol += 1
            store._solutions.add(tuple(store.value(v) for v in store._variables))

        # This one will be used when a custom filtering is specified
        def make_on_soln_filter(filter_fn):
            def on_soln_filter(store) -> None:
                store._n_sol += 1
                point = tuple(store.value(v) for v in store._variables)
                if filter_fn(point):
                    store._solutions.add(point)

            return on_soln_filter

        def make_on_soln_process(process_fn):
            def on_soln_process(store) -> None:
                store._n_sol += 1
                process_fn(tuple(store.value(v) for v in store._variables))

            return on_soln_process

        # pick the per-solution behaviour once, here
        if filter_function is not None:
            on_solution = make_on_soln_filter(filter_function)
        elif process_function is not None:
            on_solution = make_on_soln_process(process_function)
        else:
            on_solution = on_soln_default

        # If pointed cone, first check that we have a good grading vector
        if self.is_pointed():
            solver = cp_model.CpSolver()
            model = cp_model.CpModel()

            # define variables
            var = [
                model.new_int_var(-coord_bound, coord_bound, f"x_{i}")
                for i in range(hp.shape[1])
            ]

            # define constraints
            for v in hp:
                model.add(sum(int(ii) * var[i] for i, ii in enumerate(v)) >= 0)
            model.add(sum(int(ii) * var[i] for i, ii in enumerate(grading_vector)) <= 0)

            solution_storage = SolutionStorage(var, on_soln_single_pt)

            try:
                status = solver.SearchForAllSolutions(model, solution_storage)
            except MoreThanOneSolution:
                raise ValueError(
                    "More than one solution was found. The grading"
                    " vector must be wrong."
                ) from None

        if not isinstance(c, Iterable):
            c_vals = [c] * len(hp)
        else:
            c_vals = list(c)

        def make_lattice_model():
            # define the model
            solver = cp_model.CpSolver()
            model = cp_model.CpModel()

            # define variables
            var = [
                model.new_int_var(-coord_bound, coord_bound, f"x_{i}")
                for i in range(hp.shape[1])
            ]

            # define constraints
            for h, cc in zip(hp, c_vals):
                # clear the denominator
                cc_rat = Fraction(cc).limit_denominator()
                denom = cc_rat.denominator
                numer = cc_rat.numerator

                # add the constraint
                model.add(
                    sum(int(ii) * var[i] * denom for i, ii in enumerate(h)) >= numer
                )

            soln_deg = sum(int(ii) * var[i] for i, ii in enumerate(grading_vector))
            return solver, model, var, soln_deg

        solver, model, var, soln_deg = make_lattice_model()
        # the storage that will hold the points we find
        solution_storage = SolutionStorage(var, on_solution)

        finite_coord_degree_cap = None
        if finite_max_coord and max_deg is None:
            cap_solver, cap_model, _, cap_soln_deg = make_lattice_model()
            cap_model.maximize(cap_soln_deg)
            cap_status = cap_solver.solve(cap_model)
            if cap_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                finite_coord_degree_cap = int(cap_solver.objective_value)
            else:
                raise ValueError(
                    f"Could not find lattice points within the finite "
                    f"max_coord={max_coord} search box. Increase max_coord "
                    "or leave it unset to use the automatic CP-SAT bound."
                )

        # solve according to whether max_deg or min_points was specified
        if max_deg is not None:
            # If the maximum degree is specified, we use it as a constraint
            model.add(soln_deg <= max_deg)

            # solve and check status
            status = solver.SearchForAllSolutions(model, solution_storage)
            if status != cp_model.OPTIMAL:
                print(
                    "There was a problem finding the points. Status code: "
                    f"{solver.status_name(status)}"
                )
                return None
        elif min_points is not None:
            # Else, add points until the minimum number is reached
            deg = 0
            while solution_storage._n_sol < min_points:
                if (finite_coord_degree_cap is not None) and (
                    deg > finite_coord_degree_cap
                ):
                    raise ValueError(
                        f"Could not find {min_points} lattice points within "
                        f"the finite max_coord={max_coord} search box; found "
                        f"{solution_storage._n_sol}. Increase max_coord or "
                        "leave it unset to use the automatic CP-SAT bound."
                    )

                # define model with windowed degree constraints
                window_model = deepcopy(model)
                window_model.add(deg <= soln_deg)
                window_model.add(soln_deg <= deg + deg_window)

                # solve and check status
                status = solver.SearchForAllSolutions(window_model, solution_storage)
                if verbose and status != cp_model.OPTIMAL:
                    print(
                        "There was a problem finding the points b/t degrees "
                        f"{deg} and {deg + deg_window}. "
                        f"Status code: {solver.status_name(status)}"
                    )

                deg += deg_window + 1

        # parse solutions
        if process_function is not None:
            return None
        pts = np.array(list(solution_storage._solutions), dtype=int)

        # provide uniform sorting of points
        degs = pts @ grading_vector

        out = []
        for deg in sorted(set(degs)):
            out.append(sorted(pts[degs == deg].tolist()))

        return np.vstack(out)

    def is_solid(self, backend: InteriorPointBackend | str | None = None) -> bool:
        """
        **Description:**
        Returns True if the cone is solid, i.e. if it is full-dimensional.

        :::note
        If the generating rays are known this is the rank of the span. When
        only the hyperplanes are known, solidity is decided by searching for a
        point in the strict interior: none exists precisely when the cone is
        not full-dimensional.

        That inference is only sound because
        [`find_interior_point`](#find_interior_point) resolves an engine that
        certifies infeasibility. A solver that merely failed would otherwise
        report a narrow cone as degenerate -- the false negative this note
        used to warn about. Such an engine now raises rather than returning
        None, so the failure is visible instead of silently becoming a claim
        about the geometry.
        :::

        **Arguments:**
        - `backend`: Optional compatibility selector. `"ppl"` performs the
            exact polyhedral dimension check; optimizer names are forwarded to
            `find_interior_point`. Automatic certified selection is preferred.

        **Returns:**
        *(bool)* The truth value of the cone being solid.

        **Aliases:**
        `is_full_dimensional`.

        **Example:**
        We construct two cones and check if they are solid.
        ```python {3,5}
        c1 = Cone([[1,0],[0,1]])
        c2 = Cone([[1,0,0],[0,1,0]])
        c1.is_solid()
        # True
        c2.is_solid()
        # False
        ```
        """
        # check for cached answer or if we have rays... makes calc is easy
        if self._is_solid is not None:
            return self._is_solid
        if self._rays is not None:
            return exact_rank(self._rays) == self._ambient_dim

        # we just have hyperplanes... a bit harder
        if backend == "ppl":
            # One `Linear_Expression` per hyperplane, not a Python `sum` of
            # ppl products: the generator-expression form pays an object
            # multiply and add per coefficient, which is the ambient dimension
            # times the number of hyperplanes.
            cs = ppl.Constraint_System()
            for hyperplane in self.hyperplanes().tolist():
                cs.insert(ppl.Linear_Expression(hyperplane, 0) >= 0)
            polyhedron = ppl.C_Polyhedron(cs)
            self._is_solid = polyhedron.affine_dimension() == self._ambient_dim
        else:
            self._is_solid = (
                self.find_interior_point(
                    show_hints=False,
                    backend=cast("InteriorPointBackend | None", backend),
                )
                is not None
            )
        return self._is_solid

    # aliases
    is_full_dimensional = is_solid

    def is_pointed(
        self, backend: PointednessBackend = "dual", tol: float = 1e-7
    ) -> bool:
        """
        **Description:**
        Returns True if the cone is pointed (i.e. strongly convex). A cone is
        pointed if no x exists such that both x and -x are in the cone.

        Decided by duality: a cone is pointed exactly when its dual is
        full-dimensional. That reduces to a rank computation when the rays are
        known and to a certified LP otherwise, both handled by
        [`is_solid`](#is_solid).

        :::note
        Three further algorithms used to be selectable here -- a hyperplane
        rank test, an LP, and an NNLS residual against a tolerance. Each was
        valid for only one of the two representations and raised for the
        other, none was reachable without naming it explicitly, and the NNLS
        variant decided an exact question (is a rank deficient?) by comparing
        a floating-point residual to 1e-7. They have been removed; the duality
        route is exact and representation-agnostic.
        :::

        **Arguments:**
        - `backend`: Compatibility selector. All values now use the exact,
            representation-independent duality test; the historical numerical
            implementations are no longer separate code paths.
        - `tol`: Retained for call compatibility; unused by the exact test.

        **Returns:**
        The truth value of the cone being pointed.

        **Aliases:**
        `is_strongly_convex`.

        **Example:**
        We construct two cones and check if they are pointed.
        ```python {3,5}
        c1 = Cone([[1,0],[0,1]])
        c2 = Cone([[1,0],[0,1],[-1,0]])
        c1.is_pointed()
        # True
        c2.is_pointed()
        # False
        ```
        """
        valid: tuple[PointednessBackend, ...] = ("dual", "null", "lp", "nnls")
        if backend not in valid:
            raise ValueError(f"Invalid backend. The options are {valid}.")
        if backend != "dual":
            warnings.warn(
                f"backend={backend!r} is deprecated; pointedness is now "
                "computed by the exact duality test for every representation.",
                DeprecationWarning,
                stacklevel=2,
            )
        del tol
        if self._is_pointed is None:
            self._is_pointed = self.dual().is_solid()
        return self._is_pointed

    # aliases
    is_strongly_convex = is_pointed

    def is_simplicial(self):
        """
        **Description:**
        Returns True if the cone is simplicial.

        N.B.: if c is solid, then c is simplicial <=> c.dual() is simplicial.

        A sometimes-simpler check if c is solid, then, is to check if
        #(extremal hyperplanes) = dim.

        **Arguments:**
        None.

        **Returns:**
        *(bool)* The truth value of the cone being simplicial.

        **Example:**
        We construct two cones and check if they are simplicial.
        ```python {3,5}
        c1 = Cone([[1,0,0],[0,1,0],[0,0,1]])
        c2 = Cone([[1,0,0],[0,1,0],[0,0,1],[1,1,-1]])
        c1.is_simplicial()
        # True
        c2.is_simplicial()
        # False
        ```
        """
        if self._is_simplicial is not None:
            return self._is_simplicial

        # split analysis by whether we know rays or not
        if (self._rays is None) and (self.is_solid()):
            self._is_simplicial = len(self.extremal_hyperplanes()) == self.dim()
        else:
            self._is_simplicial = len(self.extremal_rays()) == self.dim()

        return self._is_simplicial

    def is_degenerate(
        self,
        use_extremal_hyperplanes: bool = True,
        M: int | None = None,
        certificate: bool = False,
        verbosity: int = 0,
    ):
        """
        **Description:**
        Checks if a cone {x : H@x>=0} is degenerate. I.e., does any x in this
        cone saturate >=d+1 hyperplanes simultaneously, for d the ambient dim?
        If so, the cone is degenerate.

        This is representation-sensitive. Just because the cone is degenerate
        for a certain representation matrix, H, doesn't mean that it's
        degenerate for all representation matrices. Probably best to use H as
        the *extremal hyperplanes*.


        Application: It is more difficult to compute the (extremal or not) rays
        of a degenerate cone.

        **Arguments:**
        - `use_extremal_hyperplanes`: Whether the check the extremal hyperplanes
            for degeneracy. If False, the naive self.hyperplanes() will be used.
        - `M`: The (absolute value of the) bounds on variables considered.
        - `certificate`: Whether to return a certificate x as well as the
            hyperplanes the solver claims it saturates
        - `verbosity`: The verbosity level.

        **Returns:**
        The maximum number of hyperplanes that a single x can saturate
        simultaneously.

        If certificate==True, also return (x,z)
        """
        if use_extremal_hyperplanes:
            H = self.extremal_hyperplanes()
        else:
            H = self.hyperplanes()

        # try a common representative of degeneracy
        xtest = np.ones(self.ambient_dim(), dtype=int)
        dists = H @ xtest
        z = dists == 0
        if sum(z) >= self.ambient_dim() + 1:
            degen, x, z = True, xtest, z
        else:
            # _is_degenerate is an minimal, non-Cone method doing the check
            degen, (x, z) = _is_degenerate(H=H, M=M, verbosity=verbosity)

        # return
        if certificate:
            return degen, (x, z)
        return degen

    def is_smooth(self):
        """
        **Description:**
        Returns True if the cone is smooth, i.e. its extremal rays either form a
        basis of the ambient lattice, or they can be extended into one.

        **Arguments:**
        None.

        **Returns:**
        *(bool)* The truth value of the cone being smooth.

        **Example:**
        We construct two cones and check if they are smooth.
        ```python {3,5}
        c1 = Cone([[1,0,0],[0,1,0],[0,0,1]])
        c2 = Cone([[2,0,1],[0,1,0],[1,0,2]])
        c1.is_smooth()
        # True
        c2.is_smooth()
        # False
        ```
        """
        if self._is_smooth is not None:
            return self._is_smooth
        if not self.is_simplicial():
            self._is_smooth = False
            return self._is_smooth
        if self.is_solid():
            # Smoothness of a simplicial solid cone is unimodularity of its ray
            # matrix, which is an exact question about integers. It used to be
            # decided by `abs(abs(np.linalg.det(rays)) - 1) < 1e-4`, and that
            # was wrong from ambient dimension 16 upward: on ray matrices whose
            # exact determinant is 1, float64 returns 1.19 at n=16 and -4.5e29
            # at n=32, so genuinely smooth cones were reported singular.
            self._is_smooth = is_unimodular(self.extremal_rays())
            return self._is_smooth
        snf = np.array(
            fmpz_mat(self.extremal_rays().tolist()).snf().tolist(), dtype=int
        )
        self._is_smooth = abs(np.prod([snf[i, i] for i in range(len(snf))])) == 1
        return self._is_smooth

    def lineality_space(self):
        """
        **Description:**
        Returns the lineality space as a formal cone object.

        This Cone object a bit odd since, by definition, the lineality space is
        the largest *linear subspace* in the cone, so it allows coefficients of
        any sign. Regardless, it's convenient to package this as a Cone

        **Arguments:**
        None.

        **Returns:**
        *(Cone)* A cone defining the lineality space.
        """
        H = self.hyperplanes()

        # the lineality space is defined by the x such that H@x==0
        # (the following definition is extremely redundant, so it's only listed
        #  for pedagogical purposes. It's better to define the cone via rays
        #  and then compute the hyperplanes via DDM since there will only be 6
        #  rays, since lineality space should typically be 5D)
        # lin = Cone(hyperplanes = np.vstack([H,-H]))

        # linearly spanning vectors are given by null(H)
        R = utils.integral_nullspace(H).T

        # to map to positively spanning rays, add in the ray r=np.sum(axis=0)
        r = -np.sum(R, axis=0)
        r = r // utils.gcd_list(r)
        R = np.vstack([R, [r]])

        lin = Cone(rays=R)

        # save the extremal rays manually
        # (this is split into two saves since _ext_rays stores both the naive
        #  extremal rays [i.e., a subset of _rays] at index 0 and the minimal
        #  extremal rays  at index 1)
        lin._ext_rays[0] = R.copy()
        lin._ext_rays[1] = R.copy()

        return lin

    def pointed_space(self):
        """
        **Description:**
        A cone can be decomposed into its lineality space and its pointed
        component.

        The pointed component is obtained by intersection of the cone with the
        orthogonal complement of the lineality space. I.e., want to impose
        H@x=0 for any x in the lineality space.

        **Arguments:**
        None.

        **Returns:**
        *(Cone)* The pointed part of the cone.
        """
        H = self.hyperplanes()

        # linearly spanning vectors of the lineality space
        # (don't need to add -\\sum_i r_i since we're dealing with linear spans)
        R = utils.integral_nullspace(H).T

        # The hyperplanes defining the orthogonal complement are just [R, -R].
        # This is because
        # R@x==0 <=> y@R@x==0 (for all y)
        #        <=> r.x==0   (for all r in the rowspan of R... lineality space)

        # the pointed part is just the intersection with these hyperplanes
        pointed = Cone(hyperplanes=np.vstack([H, R, -R]))
        return pointed

    def hilbert_basis(self):
        """
        **Description:**
        Returns the Hilbert basis of the cone using PyNormaliz.

        :::note
        This method requires the optional `normaliz` extra.
        :::

        **Arguments:**
        None.

        **Returns:**
        *(numpy.ndarray)* The list of vectors forming the Hilbert basis.

        **Example:**
        We compute the Hilbert basis of a two-dimensional cone.
        ```python {2}
        c = Cone([[1,3],[2,1]])
        c.hilbert_basis()
        # array([[1, 1],
        #        [1, 2],
        #        [1, 3],
        #        [2, 1]])
        ```
        """
        if self._hilbert_basis is not None:
            return np.array(self._hilbert_basis)
        self._hilbert_basis = _normaliz_hilbert_basis(self.rays())
        return np.array(self._hilbert_basis)

    def intersection(self, other):
        """
        **Description:**
        Computes the intersection with another cone, or with a list of cones.

        **Arguments:**
        - `other` *(Cone or array_like)*: The other cone that is being
            intersected, or a list of cones to intersect with.

        **Returns:**
        *(Cone)* The cone that results from the intersection.

        **Example:**
        We construct two cones and find their intersection.
        ```python {3}
        c1 = Cone([[1,0],[1,2]])
        c2 = Cone([[0,1],[2,1]])
        c3 = c1.intersection(c2)
        c3.rays()
        # array([[2, 1],
        #        [1, 2]])
        ```
        """
        # One code path for both spellings, so the dimension check cannot apply
        # to `intersection([c])` and not to `intersection(c)`. It used to be
        # skipped in the single-Cone branch, and a mismatch surfaced as
        # numpy's "setting an array element with a sequence ... inhomogeneous
        # shape" from inside the Cone constructor.
        others = [other] if isinstance(other, Cone) else list(other)

        for c in others:
            if not isinstance(c, Cone):
                raise ValueError("Elements of the list must be Cone objects.")
            if c.ambient_dim() != self.ambient_dim():
                raise ValueError("Ambient lattices must have the same dimension.")

        # `np.vstack`, not `.tolist()` concatenation: the constructor accepts an
        # array, and the round trip through Python lists measured 98 ms against
        # 0.56 ms (175x) for two (4000, 300) hyperplane matrices.
        return Cone(
            hyperplanes=np.vstack(
                [self.hyperplanes()] + [c.hyperplanes() for c in others]
            )
        )


def dualize(M, verbosity=0):
    """
    **Description:**
    Converts between hyperplanes and rays of a cone. Output isn't guaranteed to
    be extremal.

    Internal to this function, we treat M as the hyperplanes since that seems
    to be faster.

    **Arguments:**
    - `M`: The matrix defining the cone.
        Can be thought of as the hyperplanes cone = {x: M@x>=0} in which case we
        return the rays cone = {dualize(M).T@lmbda: lmbda>=0}.
        Can also be thought of as the rays cone = {M.T@lmbda: lmbda>=0} in
        which case we return the hypeplanes cone = {x: dualize(M)@x>=0}.
    - `verbosity`: The verbosity level.

    **Returns:**
    The dual description
    """
    M = np.asarray(M)

    # define the cone in PPL
    if verbosity >= 1:
        print("Defining the cone in PPL...", flush=True)

    cone = ppl.C_Polyhedron(M.shape[1])

    for row in M:
        ineq = ppl.Linear_Expression(row.tolist(), 0)
        cone.add_constraint(ppl.Constraint(ineq >= 0))

    # grab the dual description (in this perspective, the rays)
    if verbosity >= 1:
        print("Computing the rays...", flush=True)
    rays = []
    for gen_i, gen in enumerate(cone.minimized_generators()):
        if verbosity >= 2:
            print(f"ray #{gen_i}...", end="\r")

        if gen.is_ray():
            rays.append(tuple(int(c) for c in gen.coefficients()))
        elif gen.is_line():
            # lineality space... add both signs
            rays.append(tuple(int(c) for c in gen.coefficients()))
            rays.append(tuple(-int(c) for c in gen.coefficients()))

    # return
    return np.array(rays, dtype=int)


def is_extremal(
    R: Matrix,
    i: int,
    extFlags: list[bool] | None = None,
    method: ExtremalityMethod = "lp",
    tol: float = 1e-4,
) -> tuple[int, bool | None, Exception | None]:
    """
    **Description:**
    Auxiliary function that is used to find the extremal rays of cones. Returns
    True if the ray is extremal and False otherwise. It has additional
    parameters that are used when parallelizing.

    **Arguments:**
    - `R`: A matrix whose rows are the rays of the cone.
    - `i`: The index of the ray to check for extremality.
    - `extFlags`: A list of flags indicating if the rays r in R are possibly
        extremal. If a ray is known non-extremal, delete it.
    - `method`: The method to check extremality. Can be "lp" or "nnls".
        Reccomendation is "lp".
    - `tol`: The tolerance for determining whether a ray is extremal.

    **Returns:**
    *(bool or None)* The truth value of the ray being extremal.

    **Example:**
    This function is not meant to be directly used by the end user. Instead it
    is used by the [`extremal_rays`](#extremal_rays) function. We construct a
    cone and find its extremal_rays.
    ```python {2}
    c = Cone([[0,1],[1,1],[1,0]])
    c.extremal_rays()
    # array([[0, 1],
    #        [1, 0]])
    ```
    """
    try:
        # the ray to check if it's extremal
        r = R[i]

        # get the other rays (trim by those which are known non-extremal)
        if extFlags is None:
            R = np.delete(R, i, axis=0)
        else:
            R = np.delete(R, i, axis=0)[np.delete(extFlags, i)]

        # check if it's extremal
        if method.lower() == "lp":
            res = linprog(
                c=np.zeros(R.shape[0], dtype=int),  # no objective
                A_eq=R.T,
                b_eq=r,  # (R\r) lmbda = r
                bounds=[(0, None)],  # lmbda >= 0
                method="highs",
            )
            return (i, not res.success, None)

        if method.lower() == "nnls":
            v = nnls(R.T, r)
            return (i, abs(v[1]) > tol, None)

        raise ValueError(f"Unknown method '{method}'; expected 'lp' or 'nnls'.")
    except Exception as e:
        return (i, None, e)


# cone degeneracy
# ---------------
class EarlyStopCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, threshold, solver):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._threshold = threshold
        self._solver = solver

    def on_solution_callback(self):
        current_value = int(self.objective_value)
        if current_value >= self._threshold:
            self.StopSearch()


def _is_degenerate(
    H: Matrix,
    M: int | None = None,
    verbosity: int = 0,
) -> tuple[bool | None, tuple[np.ndarray | None, np.ndarray | None]]:
    """
    **Description:**
    Checks if a cone {x : H@x>=0} is degenerate. I.e., does any x in this cone
    saturate >=d+1 hyperplanes simultaneously, for d the ambient dim? If so, the
    cone is degenerate.

    This is representation-sensitive. Just because the cone is degenerate for a
    certain representation matrix, H, doesn't mean that it's degenerate for all
    representation matrices. Probably best to use H as the *extremal
    hyperplanes*.

    Uses CP-SAT from OR-Tools.


    Application: It is more difficult to compute the (extremal or not) rays of
    a degenerate cone.

    **Arguments:**
    - `H`: The inwards-facing hyperplanes defining the cone.
    - `M`: The (absolute value of the) bounds on variables considered.
    - `verbosity`: The verbosity level.

    **Returns:**
    Whether the cone {x : H@x>=0} is degenerate, along with the certificate
    (x, z): a witness point and the hyperplanes the solver claims it saturates.
    """
    H = np.asarray(H)

    # accommodate trivial hyperplanes
    if 0 in H.shape:
        return False, (None, None)

    # create the solver/model
    solver = cp_model.CpSolver()
    model = cp_model.CpModel()

    if verbosity >= 2:
        solver.parameters.log_search_progress = True
        solver.parameters.num_search_workers = 1

    # define variables
    # ----------------
    # variable bounds
    if M is None:
        lower = cp_model.INT32_MIN
        upper = cp_model.INT32_MAX
    else:
        lower, upper = -int(M), int(M)

    # actual variables
    x = [model.new_int_var(lower, upper, f"x_{j}") for j in range(H.shape[1])]
    xnz = [model.new_bool_var(f"nz_{j}") for j in range(H.shape[1])]

    satd = [model.new_bool_var(f"z_{i}") for i in range(H.shape[0])]

    # define constraints
    # ------------------
    # count the nonzeros
    for j in range(H.shape[1]):
        model.add(x[j] != 0).only_enforce_if(xnz[j])
        model.add(x[j] == 0).only_enforce_if(xnz[j].negated())

    # enforce nonzeros
    model.add(sum(xnz) >= 1)

    # enforce cone constraints
    for i, v in enumerate(H):
        dist = sum(_x * _v for _x, _v in zip(x, v))

        # enforce that dists are non-negative (cone hyperplane constraint)
        model.add(dist >= 0)

        # saturate the hyperplane if the indicator variable is True.
        ct = model.add(dist == 0)
        ct.only_enforce_if(satd[i])

    # define objective
    # ----------------
    model.maximize(sum(satd))

    # implement early-stop callback
    # -----------------------------
    cb = EarlyStopCallback(H.shape[1] + 1, solver)

    # solve and parse solution
    status = solver.solve(model, cb)
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        x = np.array([solver.value(_x) for _x in x])
        z = np.array([solver.value(z) for z in satd])
        degen = sum(z) >= H.shape[1] + (1 - 1e-4)

        if verbosity >= 1:
            print(f"Found x={x} saturating the indicated hyperplanes={z}...")
    elif status == cp_model.INFEASIBLE:
        if verbosity >= 1:
            warnings.warn("Solver returned status INFEASIBLE.")
        degen, z, x = False, None, None
    else:
        status_list = [
            "OPTIMAL",
            "FEASIBLE",
            "INFEASIBLE",
            "UNBOUNDED",
            "ABNORMAL",
            "MODEL_INVALID",
            "NOT_SOLVED",
        ]
        warnings.warn(f"Solver returned status {status_list[status]}.")
        degen, z, x = None, None, None

    return degen, (x, z)
