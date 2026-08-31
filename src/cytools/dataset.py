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
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cytools.polytope import Polytope

__all__ = [
    "PolytopeBatch",
    "PolytopeRecord",
    "PolytopeRecord5D",
    "load_5d_polytopes",
    "load_polytopes",
    "scan_batches",
]

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
# Record types
# ---------------------------------------------------------------------------


class PolytopeRecord(NamedTuple):
    """
    One row of the 4D Kreuzer-Skarke Parquet database.

    .. warning::
       The ``h11``, ``h12`` and ``euler_characteristic`` columns are stated in
       the **M-lattice** convention, which is the *opposite* of CYTools'
       ``lattice="N"`` default.  Verified against the database, for every row::

           record.h11                  == polytope.h11(lattice="M")
                                       == polytope.h21(lattice="N")
           record.h12                  == polytope.h11(lattice="N")
           record.euler_characteristic == polytope.chi(lattice="M")
                                       == -polytope.chi(lattice="N")

       So a ``load_polytopes(h11=(50, 100))`` filter selects on the M-lattice
       (mirror) Hodge number, *not* on the ``h11`` that ``polytope.h11()`` or
       ``cy.h11()`` returns.  Use ``h12=`` to select on the N-lattice ``h11``.
    """

    polytope: Polytope
    vertex_count: int
    h11: int  # M-lattice h11  == polytope.h21(lattice="N")
    h12: int  # M-lattice h12  == polytope.h11(lattice="N")
    euler_characteristic: int  # M-lattice chi  == -polytope.chi(lattice="N")


class PolytopeRecord5D(NamedTuple):
    polytope: Polytope
    weights: np.ndarray  # shape (6,) int32 — original weight system
    vertex_count: int
    h11: int | None  # None for non-reflexive polytopes
    h12: int | None
    h13: int | None
    reflexive: bool


# ---------------------------------------------------------------------------
# Internal column definitions
# ---------------------------------------------------------------------------

_LOAD_COLUMNS = [
    "vertices",
    "vertex_count",
    "facet_count",
    "point_count",
    "dual_point_count",
    "h11",
    "h12",
    "euler_characteristic",
]

_5D_WEIGHT_COLUMNS = [f"weight{i}" for i in range(6)]
_5D_REFLEXIVE_LOAD_COLUMNS = _5D_WEIGHT_COLUMNS + [
    "vertex_count",
    "facet_count",
    "point_count",
    "dual_point_count",
    "h11",
    "h12",
    "h13",
]
_5D_NONREFLEXIVE_LOAD_COLUMNS = _5D_WEIGHT_COLUMNS + [
    "vertex_count",
    "facet_count",
    "point_count",
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

    Requires ``huggingface_hub`` (imported lazily so that the local-Parquet
    path never depends on it).
    """
    try:
        from huggingface_hub import hf_hub_download  # ty: ignore[unresolved-import]
    except ImportError as e:
        raise ImportError(
            "Downloading datasets requires huggingface_hub. Install it with "
            "`pip install 'cytools[streaming]'` (or `cytools[notebook]`), or "
            "point CYTOOLS_DB_DIR at a local directory of Parquet files."
        ) from e

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
        )
    )


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
    h11: int | Iterable[int] | None,
    h12: int | Iterable[int] | None,
    chi: int | Iterable[int] | None,
    n_facets: int | Iterable[int] | None,
    n_points: int | Iterable[int] | None,
    n_dual_points: int | Iterable[int] | None,
) -> list[list[tuple]] | None:
    """
    Build a DNF filter list for ``pq.read_table(filters=...)``.

    Each constraint is ``(column, "=", value)`` for a scalar, or
    ``(column, "in", [...])`` for an iterable -- so ``h12=range(50, 100)``
    selects a band rather than a single value. All constraints are ANDed
    together as a single conjunction (one inner list in DNF form).
    """
    parts = []
    for val, col in [
        (h11, "h11"),
        (h12, "h12"),
        (chi, "euler_characteristic"),
        (n_facets, "facet_count"),
        (n_points, "point_count"),
        (n_dual_points, "dual_point_count"),
    ]:
        if val is None:
            continue
        if isinstance(val, (int, np.integer)):
            parts.append((col, "=", int(val)))
        else:
            vals = [int(v) for v in val]
            if not vals:
                raise ValueError(f"empty set of values for filter {col!r}")
            parts.append((col, "in", vals))
    return [parts] if parts else None


def _extract_vertices_from_table(table: pa.Table) -> list[np.ndarray]:
    """
    Convert the Arrow ``list<list<int32>>`` vertices column to a list of 2-D
    numpy arrays, one per row, each shaped ``(n_verts, 4)``.

    A single 4D KS Parquet file has a uniform vertex count (the file name
    encodes it), so the common case can be served by reshaping the flat int32
    child buffer with zero copies.  Uniformity is *verified* with vectorized
    integer compares over the Arrow offset buffers rather than assumed: a
    filtered query may merge rows from several vertex-count files, and a
    mismatched stride there silently mis-slices vertices into the wrong rows.
    The O(n) check is on numpy arrays and costs far less than the per-row
    Python fallback it guards.
    """
    col = table.column("vertices").combine_chunks()
    n_rows = len(table)
    if n_rows == 0:
        return []

    outer_off = col.offsets.to_numpy(zero_copy_only=False)
    strides = np.diff(outer_off)

    inner = col.values
    inner_off = inner.offsets.to_numpy(zero_copy_only=False)
    coord_widths = np.diff(inner_off)

    uniform = (
        strides.size
        and coord_widths.size
        and bool((strides == strides[0]).all())
        and bool((coord_widths == coord_widths[0]).all())
    )

    if uniform:
        n_verts = int(strides[0])
        n_coords = int(coord_widths[0])
        flat = inner.values.to_numpy(zero_copy_only=False)
        first = int(inner_off[int(outer_off[0])])
        arr3d = flat[first : first + n_rows * n_verts * n_coords].reshape(
            n_rows, n_verts, n_coords
        )
        return [arr3d[i] for i in range(n_rows)]

    # Ragged: rows span more than one vertex count.
    return [np.array(col[i].as_py(), dtype=np.int32) for i in range(n_rows)]


# ---------------------------------------------------------------------------
# Batch representation
# ---------------------------------------------------------------------------


# FNV-1a 64-bit constants. A vectorised mix over the coordinate columns, rather
# than a cryptographic digest per row: hashing row by row made ks_id 68% of the
# whole scan (one blake2b, one tobytes and one int.from_bytes per row), which
# defeated the point of a columnar reader.
_FNV_OFFSET = np.uint64(0xCBF29CE484222325)
_FNV_PRIME = np.uint64(0x100000001B3)


def _ks_ids(vertex_values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """A stable identifier per row, derived from the stored vertices.

    Content-derived rather than positional. A row's position depends on how the
    database happens to be partitioned and on whatever filter produced the
    scan, so a positional id would not survive re-partitioning or let two
    differently-filtered scans be joined. Hashing the vertex data gives an id
    that is the same in every scan that sees the row, which is what makes a
    derived-results store joinable against the source.

    The row width is mixed in first, so rows from different vertex-count files
    cannot collide merely by sharing a coordinate prefix.

    64 bits over the full 438M-row database carries a birthday collision
    probability of roughly 5e-3, so this is fine for keying derived results but
    should not be treated as a cryptographic or globally unique identifier.
    """
    n_rows = len(offsets) - 1
    if n_rows <= 0:
        return np.empty(0, dtype=np.int64)

    counts = np.diff(offsets)
    n_coords = vertex_values.shape[1]

    if bool((counts == counts[0]).all()):
        # uniform width: one contiguous block per row, so the whole batch mixes
        # column by column with no Python-level iteration over rows
        width = int(counts[0]) * n_coords
        block = (
            np.ascontiguousarray(vertex_values, dtype=np.int32)
            .reshape(n_rows, width)
            .view(np.uint32)
            .astype(np.uint64)
        )
        return _fnv1a(block, width)

    # ragged batch: group the rows by width so each group can still be
    # vectorised, rather than falling back to a per-row loop
    out = np.empty(n_rows, dtype=np.int64)
    for count in np.unique(counts):
        rows = np.flatnonzero(counts == count)
        width = int(count) * n_coords
        block = np.empty((len(rows), width), dtype=np.uint64)
        for j, i in enumerate(rows):
            block[j] = (
                np.ascontiguousarray(
                    vertex_values[offsets[i] : offsets[i + 1]], dtype=np.int32
                )
                .reshape(-1)
                .view(np.uint32)
                .astype(np.uint64)
            )
        out[rows] = _fnv1a(block, width)
    return out


def _fnv1a(block: np.ndarray, width: int) -> np.ndarray:
    """FNV-1a over the columns of *block*, one vectorised step per column."""
    h = np.full(len(block), _FNV_OFFSET, dtype=np.uint64)
    with np.errstate(over="ignore"):
        h ^= np.uint64(width)
        h *= _FNV_PRIME
        for c in range(width):
            h ^= block[:, c]
            h *= _FNV_PRIME
    return h.view(np.int64)


@dataclass
class PolytopeBatch:
    """A batch of KS polytopes held as flat columnar buffers.

    Deliberately contains no :class:`~cytools.polytope.Polytope` objects. A
    landscape-scale scan constructs one Python object per row only if something
    asks it to, and :meth:`polytope` is the single place that happens. The
    vertices of every row live in one contiguous ``(total_vertices, dim)``
    array addressed by ``vertex_offsets``, so a batch can be handed to a
    vectorised or compiled kernel without being taken apart first.

    See :class:`PolytopeRecord` for the M-lattice convention of the ``h11``,
    ``h12`` and ``euler_characteristic`` columns.
    """

    ks_ids: np.ndarray  # (n,) int64
    vertex_values: np.ndarray  # (total_vertices, dim) int32, contiguous
    vertex_offsets: np.ndarray  # (n+1,) int64
    vertex_count: np.ndarray  # (n,) int
    facet_count: np.ndarray  # (n,) int
    point_count: np.ndarray  # (n,) int
    dual_point_count: np.ndarray  # (n,) int
    h11: np.ndarray  # (n,) int
    h12: np.ndarray  # (n,) int
    euler_characteristic: np.ndarray  # (n,) int

    def __len__(self) -> int:
        return len(self.ks_ids)

    def vertices(self, i: int) -> np.ndarray:
        """A zero-copy view of row *i*'s vertices, shaped ``(n_verts, dim)``.

        In the order stored in the database, which is not necessarily the order
        :meth:`~cytools.polytope.Polytope.vertices` returns -- constructing a
        Polytope canonicalizes them. The two agree as sets.
        """
        return self.vertex_values[self.vertex_offsets[i] : self.vertex_offsets[i + 1]]

    def iter_vertices(self):
        """Yield each row's vertices as a view. No Polytope is constructed."""
        off = self.vertex_offsets
        for i in range(len(self)):
            yield self.vertex_values[off[i] : off[i + 1]]

    def polytope(self, i: int) -> Polytope:
        """Materialize row *i* as a Polytope. The explicit opt-in."""
        return Polytope(self.vertices(i))

    def record(self, i: int) -> PolytopeRecord:
        """Materialize row *i* as a :class:`PolytopeRecord`."""
        return PolytopeRecord(
            polytope=self.polytope(i),
            vertex_count=int(self.vertex_count[i]),
            h11=int(self.h11[i]),
            h12=int(self.h12[i]),
            euler_characteristic=int(self.euler_characteristic[i]),
        )

    def records(self) -> list[PolytopeRecord]:
        """Materialize the whole batch. Constructs one Polytope per row."""
        return [self.record(i) for i in range(len(self))]

    def take(self, indices) -> PolytopeBatch:
        """A new batch with only *indices*, re-packed contiguously."""
        indices = np.asarray(indices, dtype=np.int64)
        blocks = [self.vertices(int(i)) for i in indices]
        counts = np.array([len(b) for b in blocks], dtype=np.int64)
        offsets = np.zeros(len(blocks) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        values = (
            np.concatenate(blocks)
            if blocks
            else np.empty((0, self.vertex_values.shape[1]), dtype=np.int32)
        )
        return PolytopeBatch(
            ks_ids=self.ks_ids[indices],
            vertex_values=values,
            vertex_offsets=offsets,
            vertex_count=self.vertex_count[indices],
            facet_count=self.facet_count[indices],
            point_count=self.point_count[indices],
            dual_point_count=self.dual_point_count[indices],
            h11=self.h11[indices],
            h12=self.h12[indices],
            euler_characteristic=self.euler_characteristic[indices],
        )


def _resolve_4d_path(vc, resolved_dir, stream, hf_token) -> Path:
    """Resolve one Parquet path when its generator is first consumed.

    Keeping this lazy matters for notebooks using ``stream=True``: a small
    capped query can finish without downloading every vertex-count file.
    """
    if stream:
        return _hf_download(_HF_4D_REPO, _hf_4d_filename(vc), hf_token)

    assert resolved_dir is not None
    path = _db_path(vc, resolved_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Polytope database file not found: {path}\n"
            "Set CYTOOLS_DB_DIR to the directory containing the .parquet "
            "files, or pass stream=True to download from HuggingFace."
        )
    return path


def _dnf_constraints(dnf) -> list[tuple]:
    """Flatten the single-conjunction DNF filter into (column, op, value)."""
    if not dnf:
        return []
    # _build_arrow_filter emits exactly one conjunction, of "=" and "in" tuples
    return [(col, op, val) for col, op, val in dnf[0] if op in ("=", "in")]


def _row_group_can_match(metadata, rg_index: int, constraints, col_index) -> bool:
    """Whether a row group could contain a matching row, from its statistics.

    Recovers the row-group pruning that ``pq.read_table(filters=...)`` does
    natively, since ``iter_batches`` takes no filter. Conservative: any missing
    or unusable statistic means the row group is kept.
    """
    if not constraints:
        return True

    rg = metadata.row_group(rg_index)
    for col, op, val in constraints:
        j = col_index.get(col)
        if j is None:
            continue
        stats = rg.column(j).statistics
        if stats is None or not stats.has_min_max:
            continue
        # For a set of wanted values the row group is only excluded when the
        # whole set falls outside [min, max]; a gap inside the range cannot be
        # ruled out from statistics alone.
        lo = val if op == "=" else min(val)
        hi = val if op == "=" else max(val)
        if hi < stats.min or lo > stats.max:
            return False
    return True


def _iter_record_batches(
    *,
    counts,
    h11,
    h12,
    chi,
    n_facets,
    n_points,
    n_dual_points,
    n,
    seed,
    resolved_dir,
    stream,
    hf_token,
    batch_size: int,
):
    """Stream matching rows as Arrow record batches, bounded by *batch_size*.

    The single scan implementation behind both :func:`load_polytopes` and
    :func:`scan_batches`, so filtering, sampling and path resolution cannot
    drift apart between them.

    Memory is bounded by `batch_size`, not by how many rows match. The earlier
    implementation read every matching row into one table and then sliced it,
    which cost about 6.9 kB per row held simultaneously -- roughly 6.9 GB at a
    million rows and far beyond any machine for the full database, so the batch
    API could not actually be used at the scale it exists for.

    Results are not cached. The previous process-level cache retained whole
    query results including live Polytope objects, which is unbounded retention
    at landscape scale, for a saving the filesystem page cache already provides.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if n is not None and n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0:
        return

    counts = list(counts)
    if not counts:
        return

    dnf = _build_arrow_filter(h11, h12, chi, n_facets, n_points, n_dual_points)
    expr = pq.filters_to_expression(dnf) if dnf else None

    def one_file(idx, vc):
        path = _resolve_4d_path(vc, resolved_dir, stream, hf_token)
        file_rng = np.random.default_rng(None if seed is None else seed + idx)
        yield from _stream_one_file(path, dnf, expr, file_rng, batch_size)

    # Generators resolve their files on first use. In particular, streaming a
    # five-row sample from 32 requested vertex counts downloads at most five
    # files unless one of those files has no matching rows.
    gens = [one_file(idx, vc) for idx, vc in enumerate(counts)]

    if n is None:
        # Nothing to apportion; drain every file, interleaved so a consumer that
        # stops early still sees a spread of vertex counts.
        active = list(gens)
        while active:
            still_active = []
            for gen in active:
                try:
                    batch = next(gen)
                except StopIteration:
                    continue
                if batch.num_rows:
                    yield batch
                still_active.append(gen)
            active = still_active
        return

    # A capped scan is apportioned across the requested vertex counts rather
    # than taken from whichever file happens to come first. Without this, a
    # small `n` is served entirely out of paths[0]: the first file yields a
    # full batch_size batch, which is sliced down to n and exhausts the budget,
    # so `n_vertices=[13, 14, 15], n=16` returns sixteen 13-vertex polytopes.
    base, extra = divmod(n, len(gens))
    budgets = [base + (1 if i < extra else 0) for i in range(len(gens))]
    produced = [0] * len(gens)
    exhausted = [False] * len(gens)
    total = 0

    # Rows left over from a batch that was larger than the caller wanted. They
    # must be kept rather than dropped: discarding the tail made *which* rows a
    # capped scan returns depend on batch_size, so the same query with a
    # different batch_size sampled different polytopes and produced different
    # ks_ids.
    pending: list = [None] * len(gens)

    def pull(i, want):
        """Take up to *want* rows from file *i*; None once it is drained."""
        while True:
            batch = pending[i]
            if batch is None:
                try:
                    batch = next(gens[i])
                except StopIteration:
                    exhausted[i] = True
                    return None
            if not batch.num_rows:
                pending[i] = None
                continue
            if batch.num_rows <= want:
                pending[i] = None
                return batch
            pending[i] = batch.slice(want)
            return batch.slice(0, want)

    # First pass: honour each file's share.
    while total < n and not all(exhausted):
        progressed = False
        for i in range(len(gens)):
            if total >= n:
                break
            want = min(budgets[i] - produced[i], n - total)
            if want <= 0 or exhausted[i]:
                continue
            batch = pull(i, want)
            if batch is None:
                continue
            produced[i] += batch.num_rows
            total += batch.num_rows
            progressed = True
            yield batch
        if not progressed:
            break

    # Second pass: some files had fewer matching rows than their share, so top
    # up from whichever still have rows.
    while total < n and not all(exhausted):
        progressed = False
        for i in range(len(gens)):
            if total >= n:
                break
            if exhausted[i]:
                continue
            batch = pull(i, n - total)
            if batch is None:
                continue
            produced[i] += batch.num_rows
            total += batch.num_rows
            progressed = True
            yield batch
        if not progressed:
            break


def _stream_one_file(path: Path, dnf, expr, rng, batch_size: int):
    """Yield record batches from one file with memory bounded by *batch_size*.

    Uses ``ParquetFile.iter_batches`` rather than the dataset Scanner. The row
    groups here hold ~988,000 rows each, and the Scanner decodes a whole row
    group before handing back a batch -- 672 MB of Arrow buffers to produce 500
    rows, even with readahead disabled. ``iter_batches`` slices within a row
    group and holds ~50 MB for the same request.

    The cost is that ``iter_batches`` takes no filter, so the predicate is
    applied per batch and row-group pruning is done from column statistics
    instead.
    """
    pf = pq.ParquetFile(path)
    metadata = pf.metadata
    constraints = _dnf_constraints(dnf)
    col_index = {metadata.schema.column(j).name: j for j in range(metadata.num_columns)}

    candidates = [
        rg
        for rg in range(metadata.num_row_groups)
        if _row_group_can_match(metadata, rg, constraints, col_index)
    ]
    if not candidates:
        return

    # Random row-group order: file order correlates with how the database was
    # generated, so a prefix taken in file order is a biased sample.
    for i in rng.permutation(len(candidates)):
        rg = candidates[int(i)]
        for batch in pf.iter_batches(
            batch_size=batch_size, columns=_LOAD_COLUMNS, row_groups=[rg]
        ):
            if not batch.num_rows:
                continue
            if expr is not None:
                table = pa.Table.from_batches([batch]).filter(expr)
                if not table.num_rows:
                    continue
                for filtered in table.to_batches():
                    if filtered.num_rows:
                        yield filtered
            else:
                yield batch


def _scan_table(
    *,
    counts,
    h11,
    h12,
    chi,
    n_facets,
    n_points,
    n_dual_points,
    n,
    seed,
    resolved_dir,
    stream,
    hf_token,
) -> pa.Table:
    """Collect a scan into one table.

    For :func:`load_polytopes`, which materializes a Polytope per row anyway
    and so is already bounded by the object graph rather than by the Arrow
    buffers. Prefer :func:`scan_batches` at scale.
    """
    batches = list(
        _iter_record_batches(
            counts=counts,
            h11=h11,
            h12=h12,
            chi=chi,
            n_facets=n_facets,
            n_points=n_points,
            n_dual_points=n_dual_points,
            n=n,
            seed=seed,
            resolved_dir=resolved_dir,
            stream=stream,
            hf_token=hf_token,
            batch_size=4096,
        )
    )
    if not batches:
        return pa.table({col: [] for col in _LOAD_COLUMNS})
    return pa.Table.from_batches(batches)


def scan_batches(
    n_vertices: int | Iterable[int] | None = None,
    h11: int | Iterable[int] | None = None,
    h12: int | Iterable[int] | None = None,
    chi: int | Iterable[int] | None = None,
    n_facets: int | Iterable[int] | None = None,
    n_points: int | Iterable[int] | None = None,
    n_dual_points: int | Iterable[int] | None = None,
    n: int | None = None,
    batch_size: int = 4096,
    seed: int = 42,
    db_dir: Path | str | None = None,
    stream: bool = False,
    hf_token: str | None = None,
):
    """
    **Description:**
    Scan the 4D Kreuzer-Skarke database and yield :class:`PolytopeBatch`
    objects, without constructing a `Polytope` for any row.

    This is the batch-native counterpart to :func:`load_polytopes`. That
    function materializes one Python `Polytope` per row eagerly, which is fine
    interactively but is the dominant cost of a landscape-scale scan: the
    Parquet read hands back contiguous integer buffers, and building an object
    per row immediately discards that layout. Use `scan_batches` when the work
    is "apply a computation to many rows", and `load_polytopes` when you want
    objects to poke at.

    Filtering arguments match :func:`load_polytopes`, including the M-lattice
    convention of `h11`/`h12`/`chi` documented on :class:`PolytopeRecord`.

    **Arguments:**
    - `n_vertices`: Vertex-count file(s) to scan. `None` scans all available.
    - `h11`, `h12`, `chi`, `n_facets`, `n_points`, `n_dual_points`: Column
        filters, pushed down to Parquet.
    - `n`: Total rows to sample. `None` scans everything matching.
    - `batch_size`: Rows per yielded batch.
    - `seed`: Sampling seed.
    - `db_dir`: Local database directory. Defaults to `CYTOOLS_DB_DIR`.
    - `stream`: Download from HuggingFace instead of reading locally.
    - `hf_token`: HuggingFace token, when streaming.

    **Returns:**
    *(generator)* Yields :class:`PolytopeBatch`.

    **Example:**
    Compute an invariant over many geometries without building the objects.
    ```python {4}
    from cytools.dataset import scan_batches
    total = 0
    for batch in scan_batches(n_vertices=[13, 14], n=10000):
        for verts in batch.iter_vertices():
            total += len(verts)
    ```
    """
    # Validate before touching local configuration or the network. In
    # particular, an empty notebook query should be a harmless empty iterator.
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if n is not None and n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0:
        return

    resolved_dir = (
        _resolve_dir(db_dir, DB_DIR, "CYTOOLS_DB_DIR", "4D polytope")
        if not stream
        else None
    )

    if n_vertices is None:
        if stream:
            counts = list(range(5, 37))
        else:
            assert resolved_dir is not None
            counts = _all_vertex_counts(resolved_dir)
    elif isinstance(n_vertices, (int, np.integer)):
        counts = [int(n_vertices)]
    else:
        counts = list(n_vertices)

    for record_batch in _iter_record_batches(
        counts=counts,
        h11=h11,
        h12=h12,
        chi=chi,
        n_facets=n_facets,
        n_points=n_points,
        n_dual_points=n_dual_points,
        n=n,
        seed=seed,
        resolved_dir=resolved_dir,
        stream=stream,
        hf_token=hf_token,
        batch_size=batch_size,
    ):
        yield _table_to_batch(pa.Table.from_batches([record_batch]))


def _batch_buffers(table: pa.Table):
    """The vertices column as one flat ``(total_vertices, dim)`` array + offsets.

    Arrow already stores a list<list<int32>> column as a single contiguous child
    buffer plus offsets, which is exactly the layout PolytopeBatch wants. The
    uniform case therefore reshapes that buffer in place instead of splitting it
    into per-row views and concatenating them back together -- a full copy of
    data that was already contiguous.
    """
    n_rows = len(table)
    col = table.column("vertices").combine_chunks()
    outer = col.offsets.to_numpy(zero_copy_only=False)
    inner = col.values
    inner_off = inner.offsets.to_numpy(zero_copy_only=False)

    strides = np.diff(outer)
    coord_widths = np.diff(inner_off)

    offsets = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(strides, out=offsets[1:])

    uniform = (
        strides.size
        and coord_widths.size
        and bool((coord_widths == coord_widths[0]).all())
    )

    if uniform:
        n_coords = int(coord_widths[0])
        flat = inner.values.to_numpy(zero_copy_only=False)
        first = int(inner_off[int(outer[0])])
        total = int(offsets[-1])
        values = flat[first : first + total * n_coords].reshape(total, n_coords)
        return values, offsets

    # ragged coordinate widths: nothing to reuse, rebuild row by row
    blocks = _extract_vertices_from_table(table)
    values = np.ascontiguousarray(np.concatenate(blocks), dtype=np.int32)
    return values, offsets


def _table_to_batch(table: pa.Table) -> PolytopeBatch:
    """Build a PolytopeBatch from an Arrow table without materializing rows."""
    n_rows = len(table)
    if not n_rows:
        empty_i = np.empty(0, dtype=np.int64)
        return PolytopeBatch(
            ks_ids=empty_i,
            vertex_values=np.empty((0, 4), dtype=np.int32),
            vertex_offsets=np.zeros(1, dtype=np.int64),
            vertex_count=empty_i,
            facet_count=empty_i,
            point_count=empty_i,
            dual_point_count=empty_i,
            h11=empty_i,
            h12=empty_i,
            euler_characteristic=empty_i,
        )

    values, offsets = _batch_buffers(table)

    col = lambda name: table.column(name).to_numpy(zero_copy_only=False)  # noqa: E731

    return PolytopeBatch(
        ks_ids=_ks_ids(values, offsets),
        vertex_values=values,
        vertex_offsets=offsets,
        vertex_count=col("vertex_count"),
        facet_count=col("facet_count"),
        point_count=col("point_count"),
        dual_point_count=col("dual_point_count"),
        h11=col("h11"),
        h12=col("h12"),
        euler_characteristic=col("euler_characteristic"),
    )


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
        for batch in pf.iter_batches(
            batch_size=need, columns=columns, row_groups=[rg_idx]
        ):
            batches.append(pa.Table.from_batches([batch]))
            collected += len(batch)
            break
        if collected >= n:
            break

    if not batches:
        schema = pq.read_schema(path)
        return pa.table(
            {col: pa.array([], type=schema.field(col).type) for col in columns}
        )
    return pa.concat_tables(batches)


def _table_to_records(table: pa.Table) -> list[PolytopeRecord]:
    if not len(table):
        return []
    verts_list = _extract_vertices_from_table(table)
    vc = table.column("vertex_count").to_numpy(zero_copy_only=False)
    h11 = table.column("h11").to_numpy(zero_copy_only=False)
    h12 = table.column("h12").to_numpy(zero_copy_only=False)
    ec = table.column("euler_characteristic").to_numpy(zero_copy_only=False)
    return [
        PolytopeRecord(
            polytope=Polytope(verts),
            vertex_count=int(v),
            h11=int(h),
            h12=int(h2),
            euler_characteristic=int(e),
        )
        for verts, v, h, h2, e in zip(verts_list, vc, h11, h12, ec)
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
            # Not an index shard -- the directory may hold other parquet files.
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
        (n_facets, "facet_count"),
        (n_points, "point_count"),
    ]
    if reflexive:
        mapping += [
            (h11, "h11"),
            (h12, "h12"),
            (h13, "h13"),
            (n_dual_points, "dual_point_count"),
        ]
    parts = [(col, "=", val) for val, col in mapping if val is not None]
    return [parts] if parts else None


_CACHE_5D: dict[tuple, list[PolytopeRecord5D]] = {}


def _table_to_5d_records(table: pa.Table, reflexive: bool) -> list[PolytopeRecord5D]:
    if not len(table):
        return []

    # Extract weight columns and batch-convert to vertex matrices
    weights = np.column_stack(
        [table.column(f"weight{i}").to_numpy(zero_copy_only=False) for i in range(6)]
    ).astype(np.int32)  # shape (n, 6)
    verts_batch = _weights_to_vertices(weights)  # shape (n, 7, 6)

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
    n_vertices: int | Iterable[int] | None = None,
    h11: int | Iterable[int] | None = None,
    h12: int | Iterable[int] | None = None,
    chi: int | Iterable[int] | None = None,
    n_facets: int | Iterable[int] | None = None,
    n_points: int | Iterable[int] | None = None,
    n_dual_points: int | Iterable[int] | None = None,
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
    - `n`: Maximum number of results. When the filtered set is larger, rows
        are drawn reproducibly across files and shuffled row groups, controlled
        by *seed*. This is a bounded-memory stratified sample, not a uniform
        sample over every matching row. ``None`` returns all matches.
    - `seed`: RNG seed for reproducible row-group ordering and file
        interleaving (only used when ``n`` is set).
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
    if n is not None and n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0:
        return []

    # Resolve local directory once (ignored when streaming)
    resolved_dir = (
        _resolve_dir(db_dir, DB_DIR, "CYTOOLS_DB_DIR", "4D polytope")
        if not stream
        else None
    )

    # Normalise n_vertices → list
    if n_vertices is None:
        if not stream:
            assert resolved_dir is not None
            counts = _all_vertex_counts(resolved_dir)
        else:
            counts = list(range(5, 37))
    elif isinstance(n_vertices, (int, np.integer)):
        counts = [int(n_vertices)]
    else:
        counts = list(n_vertices)

    table = _scan_table(
        counts=counts,
        h11=h11,
        h12=h12,
        chi=chi,
        n_facets=n_facets,
        n_points=n_points,
        n_dual_points=n_dual_points,
        n=n,
        seed=seed,
        resolved_dir=resolved_dir,
        stream=stream,
        hf_token=hf_token,
    )
    return _table_to_records(table)


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
    - `n`: Maximum number of results. When the filtered set is larger, rows
        are drawn reproducibly across files and shuffled row groups, controlled
        by *seed*. This is a bounded-memory stratified sample, not a uniform
        sample over every matching row.
    - `seed`: RNG seed for reproducible row-group ordering and file
        interleaving.
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
        reflexive,
        h11,
        h12,
        h13,
        n_facets,
        n_points,
        n_dual_points,
        n,
        seed,
        stream,
        str(db_dir) if not stream else None,
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

        remaining = (
            (n - collected) if (n is not None and arrow_filter is None) else None
        )
        tbl = _load_table(path, arrow_filter, remaining, rng, load_cols)
        tables.append(tbl)
        collected += len(tbl)
        if n is not None and arrow_filter is None and collected >= n:
            break

    full_table = (
        pa.concat_tables(tables) if tables else pa.table({col: [] for col in load_cols})
    )

    if n is not None and len(full_table) > n:
        idx_arr = rng.choice(len(full_table), size=n, replace=False)
        full_table = full_table.take(idx_arr)

    records = _table_to_5d_records(full_table, reflexive)
    _CACHE_5D[cache_key] = records
    return records
