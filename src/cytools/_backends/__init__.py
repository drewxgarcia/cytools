# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""Internal adapters for third-party computational engines.

Backend modules accept and return plain Python or NumPy values. They do not
import CYTools domain classes, so optional dependencies and engine-specific
APIs cannot leak into the core object model.
"""

