# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""Convex-hull engines: V-representation to H-representation.

Every engine here returns ``(inequalities, vertices)`` as plain integer arrays.
Returning the vertices alongside the facets, rather than a live engine object
for the caller to interrogate later, is deliberate: `Polytope` previously
cached a `ppl.C_Polyhedron` (or a `scipy` `ConvexHull`) as `_poly_optimal` and
branched on the backend name again inside `vertices()`, which leaked one
engine's object model into the domain class and into pickling.

Inequalities are returned in the convention used throughout CYTools:

    c_0 x_0 + ... + c_{d-1} x_{d-1} + c_d >= 0
"""

from __future__ import annotations

import numpy as np
from flint import fmpz_mat

from cytools._backends.arith import gcd_int
from cytools._backends.ppl import ppl
from cytools._typing import Matrix

__all__ = ["interval_hull", "palp_hull", "ppl_hull", "qhull_hull"]


def ppl_hull(pts: Matrix) -> tuple[np.ndarray, np.ndarray]:
    """
    **Description:**
    Exact convex hull via the Parma Polyhedra Library. All arithmetic is on
    exact rationals, so the facets of a lattice polytope are exact integers
    with no rounding step anywhere in the pipeline.

    **Arguments:**
    - `pts`: The input points, one per row.

    **Returns:**
    The inequalities and the vertices, both integer arrays.
    """
    pts = np.asarray(pts)
    dim = pts.shape[1]

    gs = ppl.Generator_System()
    vrs = np.array([ppl.Variable(i) for i in range(dim)])
    for linexp in pts @ vrs:
        gs.insert(ppl.point(linexp))
    poly = ppl.C_Polyhedron(gs)

    ineqs = np.array(
        [
            list(ineq.coefficients()) + [ineq.inhomogeneous_term()]
            for ineq in poly.minimized_constraints()
        ],
        dtype=int,
    )
    verts = np.array(
        [pt.coefficients() for pt in poly.minimized_generators()], dtype=int
    )
    return ineqs, verts


def palp_hull(pts: Matrix) -> tuple[np.ndarray, np.ndarray]:
    """
    **Description:**
    Exact convex hull via PALP. Retained for reproduction of historical runs;
    automatic selection uses the recoverable PPL adapter because PALP can
    abort the interpreter above its compiled-in limits.

    **Arguments:**
    - `pts`: The input points, one per row.

    **Returns:**
    The inequalities and the vertices, both integer arrays.
    """
    import pypalp

    pts = np.asarray(pts)
    p = pypalp.Polytope(pts)
    return np.asarray(p.equations(), dtype=int), np.asarray(p.vertices(), dtype=int)


def qhull_hull(pts: Matrix) -> tuple[np.ndarray, np.ndarray]:
    """
    **Description:**
    Convex hull via QHull, in double precision.

    :::caution
    This engine is **not exact**. QHull works in floating point and the
    integer facet coefficients below are recovered by dividing out a gcd and
    rounding. For a lattice polytope whose coordinates are large enough that
    the double-precision hull is perturbed, the recovered facets describe a
    *different polytope* -- a wrong Hodge number rather than a slower one.
    It is registered without the `EXACT` guarantee and is therefore never
    selected for lattice work; it remains available for differential testing
    against the exact engines.
    :::

    **Arguments:**
    - `pts`: The input points, one per row.

    **Returns:**
    The inequalities and the vertices, both integer arrays.
    """
    from scipy.spatial import ConvexHull

    pts = np.asarray(pts)
    dim = pts.shape[1]
    hull = ConvexHull(pts)

    # QHull identifies the facets in floating point, but once a facet's input
    # vertices are known its primitive lattice equation can be reconstructed
    # exactly. Normalizing QHull's unit normals by an *integer* gcd used to
    # truncate every coefficient below one to zero and could describe a
    # completely different polytope. Compute the integer null vector of
    # [facet_points | 1] instead, then orient it toward the hull interior.
    ineqs = set()
    homogeneous = np.column_stack((pts, np.ones(len(pts), dtype=int)))
    for simplex in hull.simplices:
        kernel, nullity = fmpz_mat(homogeneous[simplex].tolist()).nullspace()
        if nullity != 1:
            raise RuntimeError(
                "QHull returned a facet whose exact affine span has "
                f"nullity {nullity}, expected 1."
            )
        equation = np.array([int(kernel[i, 0]) for i in range(dim + 1)])
        equation //= gcd_int(equation)
        values = homogeneous @ equation
        if np.max(values) <= 0:
            equation *= -1
            values *= -1
        if np.min(values) < 0:
            raise RuntimeError(
                "QHull returned a facet that does not support the exact "
                "lattice point configuration."
            )
        ineqs.add(tuple(equation.tolist()))
    verts = np.asarray(hull.points[hull.vertices], dtype=int)
    return np.array(sorted(ineqs), dtype=int), verts


def interval_hull(pts: Matrix) -> tuple[np.ndarray, np.ndarray]:
    """
    **Description:**
    The exact hull of a one-dimensional configuration, which is an interval.

    QHull rejects 1D input outright and PPL/PALP carry avoidable overhead for
    it, so the two-line closed form is registered as its own engine rather
    than living as a special case inside each of the others.

    **Arguments:**
    - `pts`: The input points, one per row; must be one-dimensional.

    **Returns:**
    The inequalities and the vertices, both integer arrays.
    """
    pts = np.asarray(pts)
    lo, hi = int(np.min(pts)), int(np.max(pts))
    ineqs = np.array([[1, -lo], [-1, hi]], dtype=int)
    return ineqs, np.array([[lo], [hi]], dtype=int)
