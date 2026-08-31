# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""CYTools version metadata, isolated for build tools and runtime imports."""

version = "1.4.12"
__version__ = version
versions_with_serious_bugs: tuple[str, ...] = ()

__all__ = ["__version__", "version", "versions_with_serious_bugs"]
