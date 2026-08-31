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
# Description:  This module contains tools designed to perform vector
#               configuration computations.
# -----------------------------------------------------------------------------

# external imports
from collections.abc import Iterable
from typing import overload

import numpy as np
import regfans
from numpy.typing import ArrayLike

from cytools._typing import Matrix, Vector

# core CYTools imports
from cytools.cone import Cone
from cytools.polytope import Polytope

# vector configuration imports
from .fan import Fan

__all__ = ["Fan", "VectorConfiguration"]


class VectorConfiguration(regfans.VectorConfiguration):
    """
    This class handles definition/operations on vector configurations. It is
    analogous to the Polytope class. This object can be triangulated,
    making a simplicial fan.

    **Description:**
    Constructs a `VectorConfiguration` object describing a lattice vector
    configuration. This is handled by the hidden [`__init__`](#__init__)
    function.

    **Arguments:**
    - `vectors`:    The vectors defining the VC.
    - `labels`:     A list of labels for the vectors. Only integral labels are
                    allowed.
    - `eps`:        Threshold for checking for non-integral vectors.
    - `gale_basis`: An optional basis for the gale transform. If provided, then
                    the gale transform will be put a basis such that the
                    submatrix given by these labels equals the identity.

    **Returns:**
    Nothing.
    """

    def __init__(self, *args, **kwargs):
        """
        **Description:**
        Initializes a `VectorConfiguration` object.

        **Arguments:**
        - `vectors`:    The vectors defining the VC.
        - `labels`:     A list of integer labels for the vectors. Only integral
                        labels are allowed.
        - `eps`:        Threshold for checking for non-integral vectors.
        - `gale_basis`: An optional basis for the gale transform. If provided,
                        then the gale transform will be put a basis such that
                        the submatrix given by these labels equals the identity.

        **Returns:**
        Nothing.
        """
        # call regfans' initializer
        super().__init__(*args, **kwargs)

        # some Polytope info
        p = Polytope(np.asarray(self.vectors()), labels=self.labels)
        self._is_reflexive = p.is_reflexive(allow_translations=False)
        self._poly: dict[tuple[int, ...], Polytope] = {tuple(self.labels): p}

        # some toric info
        if self._is_reflexive and (self._gale_basis is None):
            self._gale_basis = p.glsm_basis(include_points_interior_to_facets=False)

    # hulls
    # -----
    def conical_hull(self, which: int | Iterable[int] | None = None) -> Cone:
        """
        **Description:**
        Compute the positive/conical hull of (some) vectors of the VC.

        If which = None, then the support over the entire VC is calculated.

        This is the most natural hull (i.e., not the convex hull) to take.

        **Arguments:**
        - `which`: Either a single label, for which the single corresponding
            vector will be returned, or a list of labels.

        **Returns:**
        The associated conical hull.
        """
        return Cone(rays=np.asarray(self.vectors(which=which)))

    # aliases
    positive_hull = conical_hull
    pos = conical_hull
    coni = conical_hull
    cone = conical_hull

    def convex_hull(self, which: int | Iterable[int] | None = None) -> Polytope:
        """
        **Description:**
        Compute the convex hull of (some) vectors of the VC.

        If which = None, then the support over the entire VC is calculated.

        This hull is not very natural from a VC perspective... mainly used to
        connect to polytopes/point configurations.

        **Arguments:**
        - `which`: Either a single label, for which the single corresponding
            vector will be returned, or a list of labels.

        **Returns:**
        The associated convex hull.
        """
        if which is None:
            which = self.labels
        elif isinstance(which, int):
            which = (which,)

        # cache computed polytopes
        which = tuple(which)
        if which not in self._poly:
            self._poly[which] = Polytope(np.asarray(self.vectors(which)), labels=which)

        return self._poly[which]

    # aliases
    conv = convex_hull
    polytope = convex_hull

    # properties of the PC
    # --------------------
    @property
    def is_reflexive(self) -> bool:
        """
        **Description:**
        Return whether or not the convex hull is reflexive.

        **Arguments:**
        None.

        **Returns:**
        True if the convex hull is reflexive. False otherwise.
        """
        return self._is_reflexive

    @property
    def divisor_basis(self) -> np.ndarray:
        """
        **Description:**
        Return the divisor basis corresponding to the Polytope class.

        As labels.

        **Arguments:**
        None.

        **Returns:**
        The divisor basis, as labels.
        """
        if not self._is_reflexive:
            raise ValueError(
                "A divisor basis is only defined for a reflexive configuration."
            )
        return np.asarray(self._gale_basis)

    @property
    def divisor_basis_inds(self) -> np.ndarray:
        """
        **Description:**
        Return the divisor basis corresponding to the Polytope class.

        As labels.

        **Arguments:**
        None.

        **Returns:**
        The divisor basis, as indices.
        """
        # map labels to inds
        return self.divisor_basis - 1

    # misc regularity methods
    # -----------------------
    def central_fan(self) -> "Fan":
        """
        **Description:**
        Generate the central fan of the vector configuration. Can be defined
        as lifting each vector by a height of 1.

        **Arguments:**
        None.

        **Returns:**
        The central fan.
        """
        return self.subdivide(heights=[1 for _ in self.labels])

    def vectors(self, which: int | Iterable[int] | None = None) -> np.ndarray:
        """
        **Description:**
        Return the vectors of the configuration, as an array.

        regfans types this as `ArrayLike`; it is always an array in practice,
        and callers here index into it and transpose it.

        **Arguments:**
        - `which`: Either a single label, for which the single corresponding
            vector will be returned, or a list of labels.

        **Returns:**
        The vectors.
        """
        return np.asarray(super().vectors(which))

    @overload
    def vectors_to_labels(self, vectors: Vector) -> int: ...

    @overload
    def vectors_to_labels(self, vectors: Matrix) -> list[int]: ...

    @overload
    def vectors_to_labels(self, vectors: ArrayLike) -> int | list[int]: ...

    def vectors_to_labels(self, vectors: ArrayLike) -> int | list[int]:
        """
        **Description:**
        Map vectors to their corresponding labels.

        **Arguments:**
        - `vectors`: Either a single vector, for which the single
            corresponding label is returned, or a list of vectors.

        **Returns:**
        The corresponding label(s).
        """
        return super().vectors_to_labels(vectors)

    def gale(self, set_basis: bool = False, **kwargs) -> np.ndarray:
        """
        **Description:**
        Compute the gale transform of the config.

        I.e., a basis of the null-space of the vectors.

        Will automatically be put in the divisor basis iff the associated
        polytope is reflexive.

        **Arguments:**
        None.

        **Returns:**
        The gale transform.
        """
        # reflexivity decides the basis, regardless of what was passed in
        return np.asarray(super().gale(set_basis=self.is_reflexive))

    def moving_cone(self, pushed_down: bool = False, verbosity: int = 0) -> Cone:
        """
        **Description:**
        Compute the moving cone of the vector configuration.

        Equiv to the support of the subfan of fine, regular triangulations.

        **Arguments:**
        - `pushed_down`: Whether to give the moving cone in h11-dim space or
        (h11+4)-dim space.
        - `verbosity`: The verbosity level.

        **Returns:**
        The moving cone.
        """
        glsm = self.gale().T

        hyps = []
        for i in range(glsm.shape[1]):
            if verbosity >= 1:
                msg = "Computing the cone corresponding to deleting "
                msg += f"i={i}/{glsm.shape[1]}..."
                print(msg)
            hyps.append(Cone(rays=np.delete(glsm, i, axis=1).T).hyperplanes())
        hyps = np.vstack(hyps)

        # pull up
        if not pushed_down:
            hyps = hyps @ glsm

        # map to cone, return
        return Cone(hyperplanes=hyps)

    # override lifting to give CYTools Fan object
    # -------------------------------------------
    def triangulate(self, *args, **kwargs):
        """
        **Description:**
        Subdivide the vector configuration either by specified cells/simplices
        or by heights.

        **Arguments:**
        - `heights`:   The heights to lift the vectors by.
        - `cells`:     The cells to use in the triangulation.
        - `backend`:   The lifting backend. Use 'qhull'.
        - `tol`:       Numerical tolerance used.
        - `verbosity`: The verbosity level. Higher is more verbose

        **Returns:**
        The resultant subdivision.
        """
        fan = super().subdivide(*args, **kwargs)
        fan = Fan.from_regfans(fan)  # cast to CYTools type
        return fan

    subdivide = triangulate


# Domain feature methods
# ----------------------
def vc(self, include_points_interior_to_facets: bool = False) -> "VectorConfiguration":
    """
    **Description:**
    Construct the VectorConfiguration associated to the polytope.

    **Arguments:**
    - `include_points_interior_to_facets`: Whether to include points interior
        to facets

    **Returns:**
    The associated VectorConfiguration.
    """
    # see if we already know the answer
    if include_points_interior_to_facets:
        if hasattr(self, "_vc_yesfacet"):
            return self._vc_yesfacet
    else:
        if hasattr(self, "_vc_nofacet"):
            return self._vc_nofacet

    # determine which points set to use
    if include_points_interior_to_facets:
        poly_labels = self.labels
    else:
        poly_labels = self.labels_not_facet

    # get the associated lattice points
    label_origin = self.label_origin
    vc_labels = tuple(sorted([label for label in poly_labels if label != label_origin]))

    # save the VC (for caching purposes)
    vc = VectorConfiguration(
        self.points(which=vc_labels),
        labels=vc_labels,
        gale_basis=self.glsm_basis(include_points_interior_to_facets=False),
    )

    vc._poly = {vc.labels: self}

    if include_points_interior_to_facets:
        self._vc_yesfacet = vc
    else:
        self._vc_nofacet = vc

    return vc


def cone_vc(self):
    """
    **Description:**
    Construct the VectorConfiguration associated to the cone.

    **Arguments:**
    None.

    **Returns:**
    The associated VectorConfiguration.
    """
    return VectorConfiguration(self.rays())
