# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""Process-level floating-point compatibility helpers."""

import ctypes

__all__ = ["reset_rounding_mode"]

# C99 fenv.h. CYTools supports POSIX platforms, where FE_TONEAREST is zero.
_FE_TONEAREST = 0


def reset_rounding_mode() -> None:
    """Restore round-to-nearest after importing native libraries such as PPL."""

    libc = ctypes.CDLL(None)
    fesetround = libc.fesetround
    fesetround.argtypes = [ctypes.c_int]
    fesetround.restype = ctypes.c_int
    if fesetround(_FE_TONEAREST) != 0:
        raise RuntimeError("Could not restore the process FPU rounding mode.")
