# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""
Canonical access to the Kreuzer-Skarke 4D and Schöller-Skarke 5D reflexive
polytope databases, stored as Parquet files.

Two access modes are supported for each database:

**Local** (default)
    Reads Parquet files from a directory on the user's machine.  The directory
    must be supplied via the ``db_dir`` parameter or an environment variable:

    - 4D: ``CYTOOLS_DB_DIR``
    - 5D: ``CYTOOLS_5D_DB_DIR``

**Streaming**
    Downloads individual Parquet files on demand from HuggingFace and caches
    them under ``~/.cache/huggingface/hub/``.  Requires the ``huggingface_hub``
    package (``pip install 'cytools[streaming]'``).  Pass ``stream=True`` and,
    if needed, a HuggingFace token via ``hf_token=`` or the ``HF_TOKEN``
    environment variable.

    - 4D repo: ``calabi-yau-data/polytopes-4d``
    - 5D repo: ``calabi-yau-data/ws-5d``

----

**4D file naming convention** (local)::

    polytopes-4d-{NN}-vertices.parquet   (NN = 05 … 36)

**4D schema**::

    vertices            list<list<int32>>   — vertex coordinates, shape (n_verts, 4)
    vertex_count        int32
    facet_count         int32
    point_count         int32
    dual_point_count    int32
    h11                 int32
    h12                 int32
    euler_characteristic int32

----

**5D file naming convention** (local, mirrors HuggingFace layout)::

    {db_dir}/reflexive/0000.parquet … 0399.parquet
    {db_dir}/non-reflexive/0000.parquet … 0405.parquet

**5D reflexive schema**::

    weight0 … weight5   int32   — weight system (q0 … q5)
    vertex_count        int32
    facet_count         int32
    point_count         int32
    dual_point_count    int32
    h11                 int32
    h12                 int32
    h13                 int32

**5D non-reflexive schema** (subset of reflexive; no Hodge numbers)::

    weight0 … weight5   int32
    vertex_count        int32
    facet_count         int32
    point_count         int32
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from cytools.polytope import Polytope

# ---------------------------------------------------------------------------
# Environment-variable-based database locations
# ---------------------------------------------------------------------------

_db_dir_env = os.environ.get("CYTOOLS_DB_DIR")
DB_DIR: Path | None = Path(_db_dir_env) if _db_dir_env else None

_db_5d_dir_env = os.environ.get("CYTOOLS_5D_DB_DIR")
DB_5D_DIR: Path | None = Path(_db_5d_dir_env) if _db_5d_dir_env else None

# ---------------------------------------------------------------------------
# HuggingFace repository identifiers
# ---------------------------------------------------------------------------

_HF_4D_REPO = "calabi-yau-data/polytopes-4d"
_HF_5D_REPO = "calabi-yau-data/ws-5d"

# ---------------------------------------------------------------------------
# Complexity tiers (used by benchmarks and convenience helpers)
# ---------------------------------------------------------------------------

TIERS: dict[str, dict] = {
    "tiny":   {"vertex_files": [5],        "n": int(os.environ.get("CYTOOLS_BENCH_N_TINY",   "20"))},
    "small":  {"vertex_files": [6, 7],     "n": int(os.environ.get("CYTOOLS_BENCH_N_SMALL",  "20"))},
    "medium": {"vertex_files": [9, 10],    "n": int(os.environ.get("CYTOOLS_BENCH_N_MEDIUM", "20"))},
    "large":  {"vertex_files": [12, 13],   "n": int(os.environ.get("CYTOOLS_BENCH_N_LARGE",  "10"))},
    # "full" samples N polytopes from every available vertex-count file so that
    # benchmarks cover the full shape/complexity distribution of the KS database.
    # Default N=100 per file (~2900 total across the 29 available files).
    "full":   {"vertex_files": None,       "n": int(os.environ.get("CYTOOLS_BENCH_N_FULL",   "100"))},
}

# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------

class PolytopeRecord(NamedTuple):
    polytope:            Polytope
    vertex_count:        int
    h11:                 int
    h12:                 int
    euler_characteristic: int


class PolytopeRecord5D(NamedTuple):
    polytope:   Polytope
    weights:    np.ndarray   # shape (6,) int32 — original weight system
    vertex_count: int
    h11:        int | None   # None for non-reflexive polytopes
    h12:        int | None
    h13:        int | None
    reflexive:  bool


# ---------------------------------------------------------------------------
# Well-known single polytopes for micro-benchmarks / quick tests
# ---------------------------------------------------------------------------

POLY_5V = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-6,-9]])
POLY_6V = Polytope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-3,-6],[-1,-1,-1,-1]])

# ---------------------------------------------------------------------------
# Internal column definitions
# ---------------------------------------------------------------------------

_LOAD_COLUMNS = ["vertices", "vertex_count", "h11", "h12", "euler_characteristic"]

_5D_WEIGHT_COLUMNS = [f"weight{i}" for i in range(6)]
_5D_REFLEXIVE_LOAD_COLUMNS = _5D_WEIGHT_COLUMNS + [
    "vertex_count", "facet_count", "point_count", "dual_point_count",
    "h11", "h12", "h13",
]
_5D_NONREFLEXIVE_LOAD_COLUMNS = _5D_WEIGHT_COLUMNS + [
    "vertex_count", "facet_count", "point_count",
]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_dir(
    db_dir: Path | str | None,
    global_default: Path | None,
    env_var: str,
    label: str,
) -> Path:
    """
    Resolve a database directory from (in priority order):
    1. The caller-supplied ``db_dir`` argument.
    2. The module-level global default (set from an environment variable at
       import time).
    3. A fresh read of the environment variable (in case it was set after
       import).

    Raises :exc:`ValueError` with an actionable message if none are set.
    """
    if db_dir is not None:
        return Path(db_dir)
    if global_default is not None:
        return global_default
    env = os.environ.get(env_var)
    if env:
        return Path(env)
    raise ValueError(
        f"No {label} database directory configured. Pass db_dir= or set "
        f"the {env_var} environment variable to the directory containing "
        f"the Parquet files."
    )


def _hf_download(repo_id: str, filename: str, token: str | None) -> Path:
    """
    Download *filename* from a HuggingFace dataset repo to the local HF cache
    and return the local path.  Repeated calls for the same file are instant
    (HF cache hit).

    Requires ``huggingface_hub``.
    """
    return Path(hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
    ))


def _hf_4d_filename(n_verts: int) -> str:
    return f"polytopes-4d-{n_verts:02d}-vertices.parquet"


def _hf_5d_filename(file_idx: int, reflexive: bool) -> str:
    subset = "reflexive" if reflexive else "non-reflexive"
    return f"{subset}/full/{file_idx:04d}.parquet"


def _weights_to_vertices(weights: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion of weight systems to simplex vertex matrices.

    Parameters
    ----------
    weights : ndarray, shape (n, d)
        Each row ``(q_0, …, q_{d-1})`` is one weight system.

    Returns
    -------
    ndarray, shape (n, d+1, d)
        ``result[i]`` is the ``(d+1) × d`` vertex matrix for weight system
        ``i``, constructed as::

            vertex j  =  q_j * e_j          for j = 0 … d-1
            vertex d  =  -(q_0, …, q_{d-1})

        This is the standard simplex whose interior point is the origin,
        suitable for direct use as ``Polytope(result[i])``.
    """
    n, d = weights.shape
    verts = np.zeros((n, d + 1, d), dtype=np.int32)
    j = np.arange(d)
    verts[:, j, j] = weights
    verts[:, d, :] = -weights
    return verts


# ---------------------------------------------------------------------------
# 4D internal helpers
# ---------------------------------------------------------------------------

def _db_path(n_verts: int, db_dir: Path) -> Path:
    return db_dir / f"polytopes-4d-{n_verts:02d}-vertices.parquet"


def _all_vertex_counts(db_dir: Path) -> list[int]:
    """Return all vertex counts for which a local 4D Parquet file exists."""
    return [n for n in range(5, 37) if _db_path(n, db_dir).exists()]


def _build_arrow_filter(
    h11: int | None,
    h12: int | None,
    chi: int | None,
    n_facets: int | None,
    n_points: int | None,
    n_dual_points: int | None,
) -> list[list[tuple]] | None:
    """
    Build a DNF filter list for ``pq.read_table(filters=...)``.

    Each tuple is ``(column, "=", value)``; all constraints are ANDed together
    as a single conjunction (one inner list in DNF form).
    """
    parts = [
        (col, "=", val)
        for val, col in [
            (h11,          "h11"),
            (h12,          "h12"),
            (chi,          "euler_characteristic"),
            (n_facets,     "facet_count"),
            (n_points,     "point_count"),
            (n_dual_points,"dual_point_count"),
        ]
        if val is not None
    ]
    return [parts] if parts else None


def _extract_vertices_from_table(table: pa.Table) -> list[np.ndarray]:
    """
    Convert the Arrow ``list<list<int32>>`` vertices column to a list of 2-D
    numpy arrays, one per row, each shaped ``(n_verts, 4)``.

    When all rows have the same vertex count (guaranteed within a single
    Parquet file), uses the raw int32 buffer for O(1) copies and returns a
    view into a single contiguous 3-D array.  Falls back to safe per-row
    extraction when vertex counts differ (multi-file filter results).
    """
    col = table.column("vertices").combine_chunks()
    outer_off = col.offsets.to_pylist()
    n_rows = len(table)

    n_verts_0 = outer_off[1] - outer_off[0]
    uniform = all(
        outer_off[i + 1] - outer_off[i] == n_verts_0 for i in range(1, n_rows)
    )

    if uniform:
        inner_off = col.values.offsets.to_pylist()
        n_coords = inner_off[1] - inner_off[0]
        buf = col.values.values.buffers()[1]
        flat = np.frombuffer(buf, dtype=np.int32)
        start = outer_off[0] * n_coords
        arr3d = flat[start : start + n_rows * n_verts_0 * n_coords].reshape(
            n_rows, n_verts_0, n_coords
        )
        return [arr3d[i] for i in range(n_rows)]

    result = []
    for i in range(n_rows):
        row = col[i].as_py()
        result.append(np.array(row, dtype=np.int32))
    return result


# Process-level cache: (vertex_counts_tuple, h11, h12, chi, n_facets, n_points,
#                        n_dual_points, n, seed, str(resolved_dir)) → records
_CACHE: dict[tuple, list[PolytopeRecord]] = {}


def _load_table(
    path: Path,
    arrow_filter,
    n: int | None,
    rng: np.random.Generator | None,
    columns: list[str],
) -> pa.Table:
    """
    Read matching rows from one Parquet file.

    When *arrow_filter* is set, delegate to ``pq.read_table`` with native
    predicate pushdown so pyarrow can skip row groups via column statistics.

    When only *n* is set (no filter), use ``iter_batches`` with an early exit
    so we decompress only the minimum number of row groups needed.
    """
    if arrow_filter is not None:
        tbl = pq.read_table(path, columns=columns, filters=arrow_filter)
        return tbl.slice(0, n) if (n is not None and len(tbl) > n) else tbl

    if n is None:
        return pq.read_table(path, columns=columns)

    pf = pq.ParquetFile(path)
    n_rg = pf.metadata.num_row_groups
    order = rng.permutation(n_rg).tolist() if rng is not None else list(range(n_rg))

    batches: list[pa.Table] = []
    collected = 0
    for rg_idx in order:
        need = n - collected
        for batch in pf.iter_batches(batch_size=need, columns=columns, row_groups=[rg_idx]):
            batches.append(pa.Table.from_batches([batch]))
            collected += len(batch)
            break
        if collected >= n:
            break

    if not batches:
        schema = pq.read_schema(path)
        return pa.table({col: pa.array([], type=schema.field(col).type) for col in columns})
    return pa.concat_tables(batches)


def _table_to_records(table: pa.Table) -> list[PolytopeRecord]:
    if len(table) == 0:
        return []
    verts_list = _extract_vertices_from_table(table)
    vc  = table.column("vertex_count").to_numpy(zero_copy_only=False)
    h11 = table.column("h11").to_numpy(zero_copy_only=False)
    h12 = table.column("h12").to_numpy(zero_copy_only=False)
    ec  = table.column("euler_characteristic").to_numpy(zero_copy_only=False)
    return [
        PolytopeRecord(
            polytope=Polytope(verts_list[i]),
            vertex_count=int(vc[i]),
            h11=int(h11[i]),
            h12=int(h12[i]),
            euler_characteristic=int(ec[i]),
        )
        for i in range(len(table))
    ]


# ---------------------------------------------------------------------------
# 5D internal helpers
# ---------------------------------------------------------------------------

def _5d_path(file_idx: int, reflexive: bool, db_dir: Path) -> Path:
    subset = "reflexive" if reflexive else "non-reflexive"
    return db_dir / subset / f"{file_idx:04d}.parquet"


def _all_5d_file_indices(reflexive: bool, db_dir: Path) -> list[int]:
    """Return all file indices for which a local 5D Parquet file exists."""
    subset = "reflexive" if reflexive else "non-reflexive"
    subdir = db_dir / subset
    if not subdir.is_dir():
        return []
    indices = []
    for f in sorted(subdir.glob("*.parquet")):
        try:
            indices.append(int(f.stem))
        except ValueError:
            pass
    return indices


def _build_5d_arrow_filter(
    h11: int | None,
    h12: int | None,
    h13: int | None,
    n_facets: int | None,
    n_points: int | None,
    n_dual_points: int | None,
    reflexive: bool,
) -> list[list[tuple]] | None:
    mapping = [
        (n_facets,      "facet_count"),
        (n_points,      "point_count"),
    ]
    if reflexive:
        mapping += [
            (h11,           "h11"),
            (h12,           "h12"),
            (h13,           "h13"),
            (n_dual_points, "dual_point_count"),
        ]
    parts = [(col, "=", val) for val, col in mapping if val is not None]
    return [parts] if parts else None


_CACHE_5D: dict[tuple, list[PolytopeRecord5D]] = {}


def _table_to_5d_records(table: pa.Table, reflexive: bool) -> list[PolytopeRecord5D]:
    if len(table) == 0:
        return []

    # Extract weight columns and batch-convert to vertex matrices
    weights = np.column_stack([
        table.column(f"weight{i}").to_numpy(zero_copy_only=False) for i in range(6)
    ]).astype(np.int32)                                  # shape (n, 6)
    verts_batch = _weights_to_vertices(weights)          # shape (n, 7, 6)

    vc = table.column("vertex_count").to_numpy(zero_copy_only=False)
    if reflexive:
        h11 = table.column("h11").to_numpy(zero_copy_only=False)
        h12 = table.column("h12").to_numpy(zero_copy_only=False)
        h13 = table.column("h13").to_numpy(zero_copy_only=False)
    else:
        h11 = h12 = h13 = None

    n = len(table)
    return [
        PolytopeRecord5D(
            polytope=Polytope(verts_batch[i]),
            weights=weights[i],
            vertex_count=int(vc[i]),
            h11=int(h11[i]) if h11 is not None else None,
            h12=int(h12[i]) if h12 is not None else None,
            h13=int(h13[i]) if h13 is not None else None,
            reflexive=reflexive,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Public API — 4D
# ---------------------------------------------------------------------------

def load_polytopes(
    n_vertices: int | list[int] | None = None,
    h11: int | None = None,
    h12: int | None = None,
    chi: int | None = None,
    n_facets: int | None = None,
    n_points: int | None = None,
    n_dual_points: int | None = None,
    n: int | None = None,
    seed: int = 42,
    db_dir: Path | str | None = None,
    stream: bool = False,
    hf_token: str | None = None,
) -> list[PolytopeRecord]:
    """
    Load reflexive 4D polytopes from the Kreuzer-Skarke database.

    **Arguments:**
    - `n_vertices`: Restrict to files with this vertex count (int) or list of
        counts. ``None`` searches all available files.
    - `h11`: Filter by Hodge number $h^{1,1}$.
    - `h12`: Filter by Hodge number $h^{1,2}$.
    - `chi`: Filter by Euler characteristic.
    - `n_facets`: Filter by number of facets.
    - `n_points`: Filter by number of lattice points.
    - `n_dual_points`: Filter by number of dual lattice points.
    - `n`: Maximum number of results.  When the filtered set is larger, a
        reproducible random sample of size *n* is returned (controlled by
        *seed*).  ``None`` returns all matching polytopes.
    - `seed`: RNG seed for reproducible sampling (only used when ``n`` is set
        and the result set is larger than ``n``).
    - `db_dir`: Path to the local directory containing the Parquet files.
        Ignored when ``stream=True``.  If omitted, falls back to
        ``$CYTOOLS_DB_DIR``.  A :exc:`ValueError` is raised if neither is set
        and ``stream=False``.
    - `stream`: If ``True``, download files on demand from HuggingFace
        (``calabi-yau-data/polytopes-4d``) instead of reading from a local
        directory.  Requires ``huggingface_hub`` (``pip install
        'cytools[streaming]'``).
    - `hf_token`: HuggingFace API token for authenticated access.  Only used
        when ``stream=True``.  Can also be set via the ``HF_TOKEN`` environment
        variable.

    **Returns:**
    A list of :class:`PolytopeRecord` named tuples.

    **Example:**
    ```python
    from cytools import load_polytopes

    # Local
    recs = load_polytopes(h11=3, n=10, db_dir="/data/polytopes-4d")

    # Streaming
    recs = load_polytopes(h11=3, n=10, stream=True)
    polys = [r.polytope for r in recs]
    ```
    """
    # Resolve local directory once (ignored when streaming)
    resolved_dir = (
        _resolve_dir(db_dir, DB_DIR, "CYTOOLS_DB_DIR", "4D polytope")
        if not stream else None
    )

    # Normalise n_vertices → list
    if n_vertices is None:
        if not stream:
            assert resolved_dir is not None
            counts = _all_vertex_counts(resolved_dir)
        else:
            counts = list(range(5, 37))
    elif isinstance(n_vertices, int):
        counts = [n_vertices]
    else:
        counts = list(n_vertices)

    arrow_filter = _build_arrow_filter(h11, h12, chi, n_facets, n_points, n_dual_points)

    cache_key = (
        tuple(counts), h11, h12, chi, n_facets, n_points, n_dual_points,
        n, seed, stream, str(db_dir) if not stream else None,
    )
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    rng = np.random.default_rng(seed)

    tables: list[pa.Table] = []
    collected = 0
    for vc in counts:
        if stream:
            path = _hf_download(_HF_4D_REPO, _hf_4d_filename(vc), hf_token)
        else:
            assert resolved_dir is not None
            path = _db_path(vc, resolved_dir)
            if not path.exists():
                raise FileNotFoundError(
                    f"Polytope database file not found: {path}\n"
                    f"Set CYTOOLS_DB_DIR to the directory containing the "
                    f".parquet files, or pass stream=True to download from "
                    f"HuggingFace."
                )

        remaining = (n - collected) if n is not None else None
        tbl = _load_table(path, arrow_filter, remaining, rng, _LOAD_COLUMNS)
        tables.append(tbl)
        collected += len(tbl)
        if n is not None and collected >= n:
            break

    full_table = (
        pa.concat_tables(tables) if tables
        else pa.table({col: [] for col in _LOAD_COLUMNS})
    )

    if n is not None and len(full_table) > n:
        idx = rng.choice(len(full_table), size=n, replace=False)
        full_table = full_table.take(idx)

    records = _table_to_records(full_table)
    _CACHE[cache_key] = records
    return records


def load_sample(
    vertex_counts: list[int],
    n: int,
    seed: int = 42,
    db_dir: Path | str | None = None,
    stream: bool = False,
    hf_token: str | None = None,
) -> list[PolytopeRecord]:
    """
    Load *n* polytopes from the given vertex-count files.

    Convenience wrapper around :func:`load_polytopes` that mirrors the
    benchmark fixture's original signature.
    """
    return load_polytopes(
        n_vertices=vertex_counts, n=n, seed=seed,
        db_dir=db_dir, stream=stream, hf_token=hf_token,
    )


def load_tier(
    name: str,
    db_dir: Path | str | None = None,
    stream: bool = False,
    hf_token: str | None = None,
) -> list[PolytopeRecord]:
    """
    Load polytopes for a named complexity tier
    (``"tiny"``, ``"small"``, ``"medium"``, ``"large"``, ``"full"``).

    The ``"full"`` tier samples ``CYTOOLS_BENCH_N_FULL`` (default 100) polytopes
    from every vertex-count file present in the database, covering the complete
    complexity range of the KS 4D reflexive polytope database.
    """
    cfg = TIERS[name]
    vertex_files = cfg["vertex_files"]
    n_per_file = cfg["n"]

    if vertex_files is None:
        # "full" tier: discover all available files and sample n from each
        if stream:
            all_counts = list(range(5, 37))
        else:
            resolved_dir = _resolve_dir(db_dir, DB_DIR, "CYTOOLS_DB_DIR", "4D polytope")
            all_counts = _all_vertex_counts(resolved_dir)
        records = []
        for vc in all_counts:
            records.extend(load_polytopes(
                n_vertices=vc, n=n_per_file,
                db_dir=db_dir, stream=stream, hf_token=hf_token,
            ))
        return records

    return load_sample(
        vertex_files, n_per_file,
        db_dir=db_dir, stream=stream, hf_token=hf_token,
    )


# ---------------------------------------------------------------------------
# Public API — 5D
# ---------------------------------------------------------------------------

def load_5d_polytopes(
    reflexive: bool = True,
    h11: int | None = None,
    h12: int | None = None,
    h13: int | None = None,
    n_facets: int | None = None,
    n_points: int | None = None,
    n_dual_points: int | None = None,
    n: int | None = None,
    seed: int = 42,
    db_dir: Path | str | None = None,
    stream: bool = False,
    hf_token: str | None = None,
) -> list[PolytopeRecord5D]:
    """
    Load 5D polytopes from the Schöller-Skarke weight-system database.

    Weight systems are converted to polytope vertex matrices via a vectorized
    NumPy operation (no per-row PALP calls), so batch loading is fast even for
    large ``n``.

    **Arguments:**
    - `reflexive`: If ``True`` (default), load from the reflexive subset, which
        includes Hodge numbers (h11, h12, h13) and dual point counts.  If
        ``False``, load from the non-reflexive subset (no Hodge data).
    - `h11`: Filter by $h^{1,1}$.  Only valid when ``reflexive=True``.
    - `h12`: Filter by $h^{1,2}$.  Only valid when ``reflexive=True``.
    - `h13`: Filter by $h^{1,3}$.  Only valid when ``reflexive=True``.
    - `n_facets`: Filter by number of facets.
    - `n_points`: Filter by number of lattice points.
    - `n_dual_points`: Filter by number of dual lattice points.  Only valid
        when ``reflexive=True``.
    - `n`: Maximum number of results.  A reproducible random sample is returned
        when the filtered set is larger (controlled by *seed*).
    - `seed`: RNG seed for reproducible sampling.
    - `db_dir`: Path to the local database directory.  Expected layout::

            {db_dir}/reflexive/0000.parquet … 0399.parquet
            {db_dir}/non-reflexive/0000.parquet … 0405.parquet

        Ignored when ``stream=True``.  Falls back to ``$CYTOOLS_5D_DB_DIR``.
    - `stream`: If ``True``, download files on demand from HuggingFace
        (``calabi-yau-data/ws-5d``).  Requires ``huggingface_hub``.
    - `hf_token`: HuggingFace API token.  Only used when ``stream=True``.
        Can also be set via ``HF_TOKEN``.

    **Returns:**
    A list of :class:`PolytopeRecord5D` named tuples.  Each record exposes the
    original weight system (``record.weights``) alongside the constructed
    :class:`~cytools.polytope.Polytope`.

    **Example:**
    ```python
    from cytools import load_5d_polytopes

    # Local reflexive, filtered by h11
    recs = load_5d_polytopes(h11=10, n=5, db_dir="/data/ws-5d")

    # Streaming non-reflexive
    recs = load_5d_polytopes(reflexive=False, n=20, stream=True)
    print(recs[0].polytope)
    ```
    """
    # Guard: Hodge filters require reflexive data
    if not reflexive and any(v is not None for v in (h11, h12, h13)):
        raise ValueError(
            "h11, h12, and h13 filters are only available for reflexive "
            "polytopes. Set reflexive=True or remove Hodge number filters."
        )

    load_cols = (
        _5D_REFLEXIVE_LOAD_COLUMNS if reflexive else _5D_NONREFLEXIVE_LOAD_COLUMNS
    )
    arrow_filter = _build_5d_arrow_filter(
        h11, h12, h13, n_facets, n_points, n_dual_points, reflexive
    )

    # Resolve file list
    if stream:
        n_files = 400 if reflexive else 406
        file_indices = list(range(n_files))
    else:
        resolved_dir = _resolve_dir(
            db_dir, DB_5D_DIR, "CYTOOLS_5D_DB_DIR", "5D polytope"
        )
        file_indices = _all_5d_file_indices(reflexive, resolved_dir)
        if not file_indices:
            subset = "reflexive" if reflexive else "non-reflexive"
            raise FileNotFoundError(
                f"No 5D Parquet files found under {resolved_dir / subset}\n"
                f"Set CYTOOLS_5D_DB_DIR or pass stream=True to download from "
                f"HuggingFace."
            )

    cache_key = (
        reflexive, h11, h12, h13, n_facets, n_points, n_dual_points,
        n, seed, stream, str(db_dir) if not stream else None,
    )
    if cache_key in _CACHE_5D:
        return _CACHE_5D[cache_key]

    rng = np.random.default_rng(seed)
    # Shuffle file order for unbiased random sampling across files
    rng.shuffle(file_indices)

    tables: list[pa.Table] = []
    collected = 0
    for idx in file_indices:
        if stream:
            path = _hf_download(_HF_5D_REPO, _hf_5d_filename(idx, reflexive), hf_token)
        else:
            path = _5d_path(idx, reflexive, resolved_dir)
            if not path.exists():
                continue  # sparse local download — skip missing files

        remaining = (n - collected) if (n is not None and arrow_filter is None) else None
        tbl = _load_table(path, arrow_filter, remaining, rng, load_cols)
        tables.append(tbl)
        collected += len(tbl)
        if n is not None and arrow_filter is None and collected >= n:
            break

    full_table = (
        pa.concat_tables(tables) if tables
        else pa.table({col: [] for col in load_cols})
    )

    if n is not None and len(full_table) > n:
        idx_arr = rng.choice(len(full_table), size=n, replace=False)
        full_table = full_table.take(idx_arr)

    records = _table_to_5d_records(full_table, reflexive)
    _CACHE_5D[cache_key] = records
    return records
