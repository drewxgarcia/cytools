# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""Explicit update checks for interactive users.

Importing a scientific library should never start network activity.  This
module therefore exposes an opt-in check used by ``cytools.check_for_updates``.
"""

from __future__ import annotations

from cytools._version import version, versions_with_serious_bugs


def _release_tuple(value: str) -> tuple[int, ...]:
    """Return the numeric release prefix of a PEP 440-style version."""
    import re

    numbers = []
    for piece in value.split("+", 1)[0].split("."):
        match = re.match(r"\d+", piece)
        if match is None:
            break
        numbers.append(int(match.group()))
    return tuple(numbers)


def check_for_updates() -> None:
    """Print a short notice when PyPI contains a newer Workbench release.

    The check is explicit and best-effort: network and response errors are
    silently ignored so it remains safe in notebooks, batch jobs, and offline
    environments.
    """
    import requests

    try:
        response = requests.get(
            "https://pypi.org/pypi/cytools-workbench/json", timeout=2
        )
        response.raise_for_status()
        latest = str(response.json()["info"]["version"])
    except (KeyError, TypeError, ValueError, requests.RequestException):
        return

    if version in versions_with_serious_bugs:
        print(
            f"Warning: CYTools Workbench {version} contains a serious bug. "
            "Upgrade before continuing."
        )
    if _release_tuple(latest) > _release_tuple(version):
        print(
            f"A newer CYTools Workbench release is available: {version} -> {latest}. "
            "Upgrade with `python -m pip install --upgrade cytools-workbench`."
        )


__all__ = ["check_for_updates"]
