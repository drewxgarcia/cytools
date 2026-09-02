# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""Triangulation engines for a point configuration.

A note on the names this replaces. The public API previously offered
``backend="cgal"``, ``backend="qhull"`` and ``backend="topcom"``. Neither CGAL
nor TOPCOM was involved: both of those paths called `triangulumancer`, and
differed only in whether the caller supplied heights. The names survived the
library they referred to, and a user choosing "topcom" for its enumeration
guarantees got a placing triangulation from an unrelated engine.

What actually varies:

- `heights_triangulate` -- the regular triangulation induced by given heights.
  Regular by construction, and deterministic.
- `fine_triangulate` -- some fine triangulation, no heights required. Used
  when the caller has no height vector to offer.
- `qhull_triangulate` -- the same lifting computed through SciPy's QHull in
  floating point. Retained for differential testing only; see the caution.
"""

from __future__ import annotations

import numpy as np

from cytools._typing import Matrix, Vector

__all__ = ["fine_triangulate", "heights_triangulate", "qhull_triangulate"]


def heights_triangulate(points: Matrix, heights: Vector) -> np.ndarray:
    """
    **Description:**
    The regular triangulation induced by lifting `points` to `heights`. The
    result is regular by construction: it is the projection of the lower hull
    of the lifted configuration, which is what "regular" means. No
    verification step is required and none can fail.

    **Arguments:**
    - `points`: The point configuration, one point per row.
    - `heights`: One height per point.

    **Returns:**
    The simplices, as sorted index arrays.
    """
    import triangulumancer

    pc = triangulumancer.PointConfiguration(points)
    simp = pc.triangulate_with_heights(heights).simplices
    return np.array(sorted([sorted(s) for s in simp]))


def fine_triangulate(points: Matrix) -> np.ndarray:
    """
    **Description:**
    A fine triangulation of `points`, chosen by the engine. Used when the
    caller has no heights to supply.

    :::note
    The result is fine, but no height vector is returned, so regularity is
    not established by construction. Call sites that require a regular
    triangulation must resolve an engine providing that guarantee instead.
    :::

    **Arguments:**
    - `points`: The point configuration, one point per row.

    **Returns:**
    The simplices, as sorted index arrays.
    """
    import triangulumancer

    pc = triangulumancer.PointConfiguration(points)
    simp = pc.fine_triangulation().simplices
    return np.array(sorted([sorted(s) for s in simp]))


def qhull_triangulate(points: Matrix, heights: Vector) -> np.ndarray:
    """
    **Description:**
    The lifted-hull construction computed in double precision by QHull.

    :::caution
    Not exact. The lower facets are selected by the sign of a floating-point
    facet normal and the degenerate simplices are filtered by a rounded
    determinant, so a configuration near a flip can produce a different -- and
    not necessarily regular -- triangulation. Historically this engine was
    used with heights perturbed by Gaussian noise precisely because the
    unperturbed lift was unreliable, which also made it non-reproducible.
    Registered without `REGULAR` or `DETERMINISTIC`.
    :::

    **Arguments:**
    - `points`: The point configuration, one point per row.
    - `heights`: One height per point.

    **Returns:**
    The simplices, as sorted index arrays.
    """
    from scipy.spatial import ConvexHull

    lifted = [tuple(points[i]) + (heights[i],) for i in range(len(points))]
    hull = ConvexHull(lifted)

    # the lower facets; the -2 component is the lifting dimension
    low_fac = [hull.simplices[n] for n, eq in enumerate(hull.equations) if eq[-2] < 0]

    # keep only faces projecting to full-dimensional simplices
    homogeneous = [pt[:-1] + (1,) for pt in lifted]
    simp = [
        s
        for s in low_fac
        if int(round(np.linalg.det([homogeneous[i] for i in s]))) != 0
    ]
    return np.array(sorted([sorted(s) for s in simp]))
