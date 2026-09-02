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
# Description:  Miscellaneous utilities not needed for mainline CYTools.
# -----------------------------------------------------------------------------

# 'standard' imports
import gzip
import os

# 3rd party imports
import pickle

# typing
from collections.abc import Sequence
from typing import Any

from platformdirs import user_cache_dir


# numbers
# -------
def to_base10(c: Sequence[int], B: Sequence[int]) -> int:
    """
    **Description:**
    Converts a number given in components w.r.t. some bases to an integer base
    10.

    **Arguments:**
    - `c`: A list of the components.
    - `B`: A list of the bases.

    **Returns:**
    The integer in base-10.

    :::warning
    `c` and `B` must be the same length. `zip` stops at the shorter one, and
    since both are reversed, a mismatch dropped the *leading* components
    silently -- `to_base10([1, 2, 3], [10, 10])` returned 23, exactly as
    `to_base10([2, 3], [10, 10])` does. That is a wrong answer with no
    indication, so it is now an error.
    :::
    """
    if len(c) != len(B):
        raise ValueError(
            f"got {len(c)} components for {len(B)} bases; they must match "
            "(a shorter component list would silently drop leading digits)"
        )
    result = 0
    multiplier = 1
    for c_i, B_i in zip(reversed(c), reversed(B), strict=True):
        result += int(c_i) * multiplier
        multiplier *= B_i
    return result


def from_base10(n: int, B: list[int]) -> list[int]:
    """
    **Description:**
    Split an integer in base 10 to components components w.r.t. some bases.

    **Arguments:**
    - `n`: The integer in base 10.
    - `B`: A list of the bases.

    **Returns:**
    The bases
    """
    c = []
    for B_i in reversed(B):
        c.append(n % B_i)
        n //= B_i
    return list(reversed(c))


# loading/saving zipped pickle files
# ----------------------------------
# default directory to save to
cache_dir = user_cache_dir("CYTools", "CYTools")


# saving/loading functions
def load_zipped_pickle(fname, path=cache_dir):
    """
    **Description:**
    Loads zipped pickle files.

    Custom/atypical classes may fail to load.

    **Arguments:**
    - `fname`: Filename.
    - `path`: Path to file.

    **Returns:**
    Data from file.
    """
    if "." not in fname:
        fname += ".p"
    file = os.path.join(path, fname)

    if not os.path.isfile(file):
        return None

    try:
        with gzip.open(file, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        # Broadly, on purpose. The narrow `(EOFError, UnpicklingError)` this
        # replaces covered a truncated or empty file but not a corrupt one: a
        # partially overwritten cache raises `gzip.BadGzipFile`, which escaped
        # and left the caller permanently unable to read its own cache. The
        # docstring's own caveat -- "custom/atypical classes may fail to load"
        # -- names still more ways in (AttributeError, ImportError, ...). A
        # cache is by definition reconstructible, so anything unreadable should
        # be discarded rather than raised.
        print(
            f"Warning: cache {file} is broken ({type(e).__name__}: {e}), removing it..."
        )
        try:
            os.remove(file)
        except OSError:
            pass
        return None


def save_zipped_pickle(
    obj: Any,
    fname: str,
    path: str = cache_dir,
    protocol: int = pickle.DEFAULT_PROTOCOL,
):
    """
    **Description:**
    Saves zipped pickle files.

    **Arguments:**
    - `obj`: The object to save.
    - `fname`: Filename.
    - `path`: Path to file.
    - `protocol`: Protocol to use for saving the file. Defaults to
        `pickle.DEFAULT_PROTOCOL`.

    **Returns:**
    Nothing.
    """
    if "." not in fname:
        fname += ".p"
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, fname)

    # Write to a per-process temp file, then atomically rename into place. This
    # keeps concurrent writers (e.g. a 128-way array sweep all sharing one cache)
    # and processes killed mid-write from ever leaving a truncated/zero-padded
    # file: readers always see either the old or the new complete file.
    tmp = f"{file}.tmp.{os.getpid()}"
    try:
        with gzip.open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol)
        os.replace(tmp, file)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
