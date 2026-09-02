# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""What produced a derived number, recorded alongside the number.

A stored result is evidence, and evidence without an attribution cannot be
checked. This module builds a small record of everything in the environment
that can change a computed value -- the library version, the working tree it
was run from, which engine each task resolved to, and the versions of the
libraries that actually do the arithmetic -- and reduces it to a digest.
`cytools.store` stamps it into every part file it writes.

The concrete failure this prevents is not hypothetical. The stretched-cone tip
engine changed from a first-order method to a certified one, and at h11 = 100
the old answers were 45% below the true optimum. Both generations of results
key on `ks_id` and land in the same `tip/v1/` directory, and `DerivedStore.read`
concatenates them. Without a stamp, nothing in the store distinguishes a row
computed before that change from one computed after, and no amount of care at
read time can recover the difference.

The `version` argument threaded through `DerivedStore` remains a *deliberate*
statement by the caller that an algorithm changed. This is the involuntary
counterpart: it records what actually happened, including the changes nobody
remembered to declare.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import warnings
from importlib import metadata
from pathlib import Path
from typing import Any

__all__ = [
    "PARQUET_KEY",
    "ProvenanceWarning",
    "differences",
    "fingerprint",
    "read_fingerprint",
    "warn_if_mixed",
    "with_fingerprint",
]

#: Key under which the record is stored in a Parquet file's key-value
#: metadata. Parquet metadata is bytes, and the value is UTF-8 JSON.
PARQUET_KEY = b"cytools_provenance"

#: Distributions whose version can move a computed number. Deliberately not
#: "everything installed": a fingerprint that changes when an unrelated
#: notebook dependency is upgraded would cry wolf, and a fingerprint that cries
#: wolf gets ignored.
_NUMERICS = (
    "numpy",
    "scipy",
    "python-flint",
    "pplpy",
    "highspy",
    "osqp",
    "ortools",
    "cygv",
    "triangulumancer",
    "scikit-sparse",
    "Mosek",
)

_CACHED: str | None = None


class ProvenanceWarning(UserWarning):
    """Results computed under different conditions are being read together.

    Its own class so that a study which must not mix generations can promote
    it: `warnings.simplefilter("error", ProvenanceWarning)`.
    """


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _worktree() -> dict[str, Any]:
    """The git commit the package is running from, when there is one.

    An installed wheel has no worktree and reports nulls; that is a fact about
    the run worth recording rather than an error. `dirty` matters as much as
    the commit: a result computed from uncommitted edits is not reproducible
    from the commit alone, and the store should say so rather than imply
    otherwise.
    """
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if commit.returncode != 0:
            return {"commit": None, "dirty": None}
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return {
            "commit": commit.stdout.strip() or None,
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _engines() -> dict[str, list[str]]:
    """Which engines each task can use, in preference order.

    Deferred import: `cytools` keeps its numerical modules lazy at package
    import, and `tests/test_architecture.py` pins that.
    """
    from cytools._backends.engines import all_registries

    return {registry.task: list(registry.available()) for registry in all_registries()}


def fingerprint() -> dict[str, Any]:
    """
    **Description:**
    The provenance record for this process, computed once and reused.

    **Returns:**
    *(dict)* Keys `cytools`, `worktree`, `engines`, `packages`, and a `digest`
        summarising the rest. A fresh dict on every call, so a caller may
        annotate its own copy.

    **Example:**
    ```python {2}
    from cytools.provenance import fingerprint
    fingerprint()["engines"]["stretched_tip"]
    # ['highs', 'osqp']
    ```
    """
    global _CACHED
    if _CACHED is None:
        from cytools._version import version

        record: dict[str, Any] = {
            "cytools": version,
            "worktree": _worktree(),
            "engines": _engines(),
            "packages": {name: _distribution_version(name) for name in _NUMERICS},
        }
        # Sorted keys, so the digest depends on the content and not on the
        # order a dict happened to be built in.
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
        record["digest"] = hashlib.sha256(payload.encode()).hexdigest()[:16]
        _CACHED = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return json.loads(_CACHED)


def with_fingerprint(table, record: dict[str, Any] | None = None):
    """Return *table* carrying provenance in its schema metadata.

    Defaults to this process's fingerprint; pass *record* to stamp a file with
    the provenance of whatever produced its rows instead. Existing schema
    metadata is preserved, so the stamp cannot displace anything Arrow or a
    caller put there.
    """
    return table.replace_schema_metadata(
        stamp(table.schema.metadata, fingerprint() if record is None else record)
    )


def read_fingerprint(source) -> dict[str, Any] | None:
    """
    **Description:**
    The provenance recorded in a Parquet file, table, or schema.

    **Arguments:**
    - `source`: A path, a `pyarrow.Table`, or a `pyarrow.Schema`.

    **Returns:**
    *(dict | None)* The record, or `None` for data written before stamping
        existed. `None` is informative and is preserved rather than filled in:
        guessing what produced an unstamped row is the very thing this module
        exists to stop.
    """
    import pyarrow.parquet as pq

    schema = getattr(source, "schema", None)
    if schema is None:
        try:
            schema = pq.read_schema(source)
        except Exception:
            return None
    # A Table's `.schema` is a Schema; a ParquetFile's is also a Schema-like
    # wrapper whose `.metadata` we can read the same way.
    metadata_map = getattr(schema, "metadata", None) or {}
    raw = metadata_map.get(PARQUET_KEY)
    if raw is None:
        return None
    try:
        return json.loads(bytes(raw).decode())
    except (ValueError, UnicodeDecodeError):
        return None


def differences(left: dict | None, right: dict | None) -> list[str]:
    """
    **Description:**
    The provenance fields on which two records disagree, as dotted paths.

    Reports the *fields*, not the digests. "These rows came from different
    conditions" is not actionable; "these rows came from different
    `engines.stretched_tip`" is.

    **Arguments:**
    - `left`, `right`: Records from `fingerprint` or `read_fingerprint`, either
        of which may be `None` for unstamped data.

    **Returns:**
    *(list[str])* Sorted dotted paths, e.g. `["engines.stretched_tip"]`.
    """
    if left is None and right is None:
        return []
    if left is None or right is None:
        return ["<unstamped>"]

    def flatten(record: dict, prefix: str = "") -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for key, value in record.items():
            if key == "digest":
                continue  # a summary of the rest, not an independent field
            path = f"{prefix}{key}"
            if isinstance(value, dict):
                flat.update(flatten(value, f"{path}."))
            else:
                flat[path] = value
        return flat

    flat_left, flat_right = flatten(left), flatten(right)
    return sorted(
        path
        for path in set(flat_left) | set(flat_right)
        if flat_left.get(path) != flat_right.get(path)
    )


def combined_fingerprint(records: list[dict | None]) -> dict[str, Any]:
    """
    **Description:**
    Provenance for a file assembled out of others, such as a compaction.

    The result describes its *sources*, never the process doing the assembling.
    A compactor computes no numbers, so stamping it with the compactor's own
    environment would assert something false and would erase the only record
    of what actually produced the rows.

    **Arguments:**
    - `records`: One record per source file, `None` for unstamped sources.

    **Returns:**
    *(dict)* The single shared record when the sources agree; otherwise a
        record with a `sources` list, so a merge across generations stays
        visible instead of resolving to one arbitrary answer.
    """
    distinct: list[dict | None] = []
    for record in records:
        key = None if record is None else record.get("digest")
        if key not in [None if r is None else r.get("digest") for r in distinct]:
            distinct.append(record)

    if len(distinct) == 1 and distinct[0] is not None:
        return dict(distinct[0])

    sources = [r for r in distinct if r is not None]
    payload = json.dumps(sources, sort_keys=True, separators=(",", ":"))
    return {
        "sources": sources,
        "unstamped_sources": sum(1 for r in distinct if r is None),
        "digest": hashlib.sha256(payload.encode()).hexdigest()[:16],
    }


def stamp(metadata_map, record: dict[str, Any]) -> dict:
    """Return *metadata_map* with *record* added under `PARQUET_KEY`."""
    stamped = dict(metadata_map or {})
    stamped[PARQUET_KEY] = json.dumps(
        record, sort_keys=True, separators=(",", ":")
    ).encode()
    return stamped


def warn_if_mixed(records: list[dict | None], what: str) -> list[str]:
    """
    **Description:**
    Warn when *records* do not all describe the same conditions.

    **Arguments:**
    - `records`: One record per source being combined.
    - `what`: Human-readable name of the thing being read, for the message.

    **Returns:**
    *(list[str])* The differing field paths, empty when everything agrees.
    """
    if len(records) < 2:
        return []

    fields: set[str] = set()
    for other in records[1:]:
        fields.update(differences(records[0], other))
    if not fields:
        return []

    warnings.warn(
        f"{what} combines results computed under different conditions; "
        f"they disagree on {sorted(fields)}. Read each generation separately, "
        "or recompute under one, before comparing the numbers.",
        ProvenanceWarning,
        stacklevel=3,
    )
    return sorted(fields)
