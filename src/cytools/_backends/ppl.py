# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""PPL import boundary with process-state compatibility handling."""

import ppl

from cytools._backends.fpu import reset_rounding_mode

__all__ = ["ppl"]

# Older PPL builds may change the process rounding mode while loading. Keep
# that native side effect contained at the one import boundary shared by all
# domain modules.
reset_rounding_mode()
