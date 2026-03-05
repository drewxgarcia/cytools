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

# Make the main classes and function accessible from the root of CYTools.
from cytools.polytope import Polytope
from cytools.cone import Cone
from cytools.dataset import (
    load_polytopes,
    load_sample,
    load_tier,
    load_5d_polytopes,
    PolytopeRecord,
    PolytopeRecord5D,
)

# Latest version
version = "1.4.6"
versions_with_serious_bugs = []
__all__ = [
    "Cone",
    "Polytope",
    "PolytopeRecord",
    "PolytopeRecord5D",
    "load_polytopes",
    "load_sample",
    "load_tier",
    "load_5d_polytopes",
    "version",
]

# Check for more recent versions of CYTools
def check_for_updates():
    """
    **Description:**
    Checks for updates of cytools-dvg on PyPI. Prints a message if a newer
    version is available.

    **Arguments:**
    None.

    **Returns:**
    Nothing.

    **Example:**
    ```python {2}
    import cytools
    cytools.check_for_updates()
    ```
    """
    import json
    import urllib.request

    try:
        url = "https://pypi.org/pypi/cytools-dvg/json"
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = json.loads(resp.read())
        latest = data["info"]["version"]
        if tuple(int(x) for x in latest.split(".")) > tuple(int(x) for x in version.split(".")):
            print(
                f"\nInfo: A newer version of cytools-dvg is available: "
                f"v{version} -> v{latest}\n"
                "Upgrade with: pip install --upgrade cytools-dvg\n"
                "           or: uv add cytools-dvg\n"
            )
    except Exception:
        pass
