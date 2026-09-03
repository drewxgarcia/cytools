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

# Re-export the focused face-triangulation namespace for compatibility.
from cytools.ntfe import face_triangulations
from cytools.ntfe.ntfe import (
    cone_of_permissible_heights,
    expanded_secondary_fan,
    iter_ntfe_cones,
    iter_ntfe_hypers,
    ntfe_cones,
    ntfe_frsts,
    ntfe_frts,
    ntfe_hypers,
    triangface_ineqs,
    triangface_ineqs_and_triangs,
    triangfaces_to_frst,
    triangfaces_to_frt,
)

__all__ = [
    "cone_of_permissible_heights",
    "expanded_secondary_fan",
    "face_triangulations",
    "iter_ntfe_cones",
    "ntfe_cones",
    "ntfe_frsts",
    "ntfe_frts",
    "iter_ntfe_hypers",
    "ntfe_hypers",
    "triangface_ineqs",
    "triangface_ineqs_and_triangs",
    "triangfaces_to_frst",
    "triangfaces_to_frt",
]
