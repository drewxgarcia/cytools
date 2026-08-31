# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# =============================================================================
"""
A Parquet-backed store for quantities derived from the polytope database.

The point is to change what "compute" means. Asking for the Mori cones of
48,000 geometries should not mean *calculate 48,000 Mori cones*; it should mean
*materialize the ones that are missing*. That makes a landscape-scale scan
resumable by construction: interrupt it, run it again, and it picks up the
work that is left rather than starting over.

Everything is keyed on the stable ``ks_id`` that
:class:`~cytools.dataset.PolytopeBatch` carries, so derived datasets are
relational against the source and against each other::

    <root>/<quantity>/v<version>/part-<write-time>-<hex>.parquet

        ks_id  int64      the join key
        ...               one column per field the payload returns

The layout is deliberately boring, because it is meant to stay readable after
someone has generated terabytes against it. A quantity is a directory, an
algorithm version is a directory, and a write is a new file -- never an
in-place edit -- so a crashed write cannot corrupt what was already there.

Usage::

    from cytools.dataset import scan_batches
    from cytools.store import DerivedStore, materialize

    store = DerivedStore("~/ks-derived")

    def hodge(vertices):
        from cytools import Polytope
        cy = Polytope(vertices).triangulate().get_toric_variety().get_cy()
        return {"h11": cy.h11(), "h21": cy.h21(), "chi": cy.chi()}

    summary = materialize(
        "hodge", hodge, store=store,
        scan=scan_batches(n_vertices=[13, 14], n=10000),
    )
    # {'requested': 10000, 'computed': 10000, 'skipped': 0,
    #  'unsupported': 0, 'failed': 0}

    # run it again: nothing left to do
    summary = materialize(...)
    # {'requested': 10000, 'computed': 0, 'skipped': 10000,
    #  'unsupported': 0, 'failed': 0}

    table = store.read("hodge")          # everything computed so far
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "ERROR_COLUMN",
    "UNSUPPORTED_COLUMN",
    "DerivedStore",
    "Unsupported",
    "materialize",
]

_ID_COLUMN = "ks_id"

# The two status columns a payload can produce instead of results. Both are
# public: consumers need to recognise them to tell a status row from a
# computed one, and exporting only one of the pair invites the other to be
# hard-coded at the call site.
ERROR_COLUMN = "error"
UNSUPPORTED_COLUMN = "unsupported"


class Unsupported(Exception):
    """Raised by a payload when a geometry cannot support the requested work.

    Distinct from an error: a non-favorable polytope has no Calabi-Yau
    hypersurface in the non-experimental regime, which is a fact about the
    geometry rather than a bug. `materialize` counts these apart from failures
    and still writes the row, so the scan does not retry it on the next run.

    Named for the outcome it produces. `materialize` reports four
    mutually exclusive outcomes per row -- `computed`, `skipped`, `unsupported`
    and `failed` -- and this exception is how a payload asks for the third.
    Note in particular that it is *not* `skipped`, which means the row was
    already in the store.
    """


def _resolve_root(root: Path | str | None) -> Path:
    """Locate the store, defaulting to a user cache directory.

    Nothing should have to be configured before a result can be computed, so an
    unset `CYTOOLS_DERIVED_DIR` falls back to the platform cache location
    rather than raising.
    """
    if root is None:
        env = os.environ.get("CYTOOLS_DERIVED_DIR")
        if env:
            root = env
        else:
            from platformdirs import user_cache_dir

            root = Path(user_cache_dir("cytools")) / "derived"
    return Path(root).expanduser()


class DerivedStore:
    """
    **Description:**
    A directory of derived quantities, keyed by `ks_id`.

    **Arguments:**
    - `root`: Directory to hold the store. Defaults to `CYTOOLS_DERIVED_DIR`,
        then the platform's CYTools user-cache directory.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = _resolve_root(root)

    # -- layout ------------------------------------------------------------

    def quantity_dir(self, quantity: str, version: int = 1) -> Path:
        """The directory holding one quantity at one algorithm version."""
        if not quantity or "/" in quantity or quantity.startswith("."):
            raise ValueError(f"Invalid quantity name: {quantity!r}")
        if int(version) < 0:
            raise ValueError(f"Invalid version: {version!r}")
        return self.root / quantity / f"v{int(version)}"

    def quantities(self) -> list[str]:
        """Quantity names present in the store."""
        if not self.root.exists():
            return []
        return sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    def versions(self, quantity: str) -> list[int]:
        """Algorithm versions present for *quantity*."""
        d = self.root / quantity
        if not d.exists():
            return []
        out = []
        for p in d.iterdir():
            if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit():
                out.append(int(p.name[1:]))
        return sorted(out)

    def _parts(self, quantity: str, version: int = 1) -> list[Path]:
        d = self.quantity_dir(quantity, version)
        if not d.exists():
            return []
        return sorted(d.glob("part-*.parquet"), key=self._part_order)

    @staticmethod
    def _part_order(path: Path) -> tuple[int, str]:
        """Sort parts by write order, including stores made by older versions.

        New part names carry a nanosecond timestamp. Legacy names contain only
        a UUID, so their modification time is the best available ordering. The
        filename is a deterministic tie-breaker for filesystems with coarse
        timestamps.
        """
        pieces = path.stem.split("-", 2)
        if len(pieces) >= 3 and pieces[1].isdigit():
            written = int(pieces[1])
        else:
            try:
                written = path.stat().st_mtime_ns
            except OSError:
                written = 0
        return written, path.name

    @staticmethod
    def _part_path(directory: Path) -> Path:
        """Return a collision-resistant part name that preserves write order."""
        return directory / f"part-{time.time_ns():020d}-{uuid.uuid4().hex}.parquet"

    def _part_ranges(self, quantity: str, version: int = 1):
        """Yield ``(path, min_id, max_id)`` per part, from Parquet footers only.

        Reading the footer is cheap and independent of the part's size, which
        is what makes it worth doing before touching any data.

        A caveat worth stating plainly, because it was measured rather than
        assumed: ks_id is a content hash, so ids are spread uniformly over the
        int64 range, and a query for an arbitrary *set* of ids spans nearly
        that whole range. Over a 40-part store, both a 50-id and a 4096-id
        query passed the range test on 40 of 40 parts. This prunes nothing for
        scattered ids and only helps if a caller ever queries a genuinely
        narrow id range.
        """
        for path in self._parts(quantity, version):
            try:
                metadata = pq.read_metadata(path)
                col = metadata.schema.names.index(_ID_COLUMN)
            except Exception:
                yield path, None, None
                continue

            lo = hi = None
            for rg in range(metadata.num_row_groups):
                stats = metadata.row_group(rg).column(col).statistics
                if stats is None or not stats.has_min_max:
                    lo = hi = None
                    break
                lo = stats.min if lo is None else min(lo, stats.min)
                hi = stats.max if hi is None else max(hi, stats.max)
            yield path, lo, hi

    # -- reading -----------------------------------------------------------

    def known_ids(self, quantity: str, version: int = 1) -> np.ndarray:
        """The `ks_id`s already stored for *quantity*.

        Reads only the id column, so this stays cheap as the store grows.
        """
        parts = self._parts(quantity, version)
        if not parts:
            return np.empty(0, dtype=np.int64)

        chunks = []
        for p in parts:
            try:
                tbl = pq.read_table(p, columns=[_ID_COLUMN])
            except Exception:
                # a partially written file from a killed run; a later write of
                # the same ids will supersede it
                continue
            chunks.append(tbl.column(_ID_COLUMN).to_numpy(zero_copy_only=False))

        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(chunks).astype(np.int64))

    def missing(self, quantity: str, ks_ids, version: int = 1) -> np.ndarray:
        """
        **Description:**
        Which of *ks_ids* are not yet stored, in the order given.

        Works one part at a time, so peak memory tracks a single part rather
        than the whole store. On an 8,000,000-row store: `known_ids` costs
        613 MB, this costs 46 MB. `known_ids` remains the cheaper choice in I/O
        when every id will be needed anyway, which is why `materialize` uses
        it, but it scales with the store and this does not.

        The scan stops as soon as every queried id is accounted for, so a query
        whose ids sit in an early part short-circuits: 1.5 ms against 46 ms for
        the same query landing in the last of 40 parts.
        """
        ids = np.asarray(ks_ids, dtype=np.int64)
        if not len(ids):
            return ids

        found = np.zeros(len(ids), dtype=bool)
        lo = int(ids.min())
        hi = int(ids.max())

        for path, part_lo, part_hi in self._part_ranges(quantity, version):
            if part_lo is not None and (part_hi < lo or part_lo > hi):
                continue
            try:
                column = pq.read_table(path, columns=[_ID_COLUMN])
            except Exception:
                continue
            part_ids = column.column(_ID_COLUMN).to_numpy(zero_copy_only=False)
            found |= np.isin(ids, part_ids.astype(np.int64))
            if found.all():
                break

        return ids[~found]

    def read(
        self,
        quantity: str,
        version: int = 1,
        ks_ids=None,
    ) -> pa.Table:
        """
        **Description:**
        Read stored results for *quantity*, optionally restricted to `ks_ids`.

        Duplicate ids -- possible after recomputation or concurrent runs -- are
        reduced to their most recently written occurrence, so a successful
        recomputation deterministically supersedes the prior value.
        """
        parts = self._parts(quantity, version)
        if not parts:
            return pa.table({_ID_COLUMN: pa.array([], type=pa.int64())})

        tables = []
        for p in parts:
            try:
                tables.append(pq.read_table(p))
            except Exception:
                continue
        if not tables:
            return pa.table({_ID_COLUMN: pa.array([], type=pa.int64())})

        table = pa.concat_tables(tables, promote_options="default")

        ids = table.column(_ID_COLUMN).to_numpy(zero_copy_only=False).astype(np.int64)
        # Parts are concatenated oldest to newest. Select from the reversed id
        # array so every duplicate resolves to the latest completed write.
        _, latest_from_end = np.unique(ids[::-1], return_index=True)
        keep = np.sort(len(ids) - 1 - latest_from_end)

        if ks_ids is not None:
            wanted = np.asarray(ks_ids, dtype=np.int64)
            keep = keep[np.isin(ids[keep], wanted)]

        return table.take(keep)

    def stats(self, quantity: str, version: int = 1) -> dict:
        """Row and file counts for one quantity/version."""
        parts = self._parts(quantity, version)
        return {
            "quantity": quantity,
            "version": version,
            "n_rows": int(len(self.known_ids(quantity, version))),
            "n_parts": len(parts),
            "bytes": sum(p.stat().st_size for p in parts),
            "path": str(self.quantity_dir(quantity, version)),
        }

    # -- writing -----------------------------------------------------------

    def compact(self, quantity: str, version: int = 1) -> Path | None:
        """
        **Description:**
        Merge a quantity's part files into one, de-duplicating ids.

        Every `materialize` batch appends a part file, so a store written over
        many runs accumulates small files and `known_ids` slows down in
        proportion. Compaction is safe to run at any time: the merged file is
        written under a temporary name and renamed into place before the old
        parts are removed, so an interruption leaves a readable store either
        way -- at worst with the merged file present alongside the originals,
        which de-duplicates on read.

        **Returns:**
        The merged part file, or `None` if there was nothing to compact.
        """
        parts = self._parts(quantity, version)
        if len(parts) <= 1:
            return None

        table = self.read(quantity, version)
        if not table.num_rows:
            return None
        table = table.sort_by(_ID_COLUMN)

        d = self.quantity_dir(quantity, version)
        final = self._part_path(d)
        tmp = final.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        tmp.replace(final)

        for p in parts:
            try:
                p.unlink()
            except OSError:
                # Best-effort cleanup: the merged table is already durable, so
                # a leftover part is wasted space, not lost or corrupt data.
                pass
        return final

    def write(
        self,
        quantity: str,
        ks_ids,
        results: list[dict],
        version: int = 1,
    ) -> Path | None:
        """
        **Description:**
        Append one shard of results.

        Writes a new part file rather than editing anything, via a temporary
        name that is renamed into place, so an interrupted write leaves the
        store as it was.

        **Arguments:**
        - `quantity`: Name of the derived quantity.
        - `ks_ids`: One id per entry of *results*.
        - `results`: Dicts of scalars and/or numeric sequences. Keys need not
            agree between entries; missing keys become nulls.
        - `version`: Algorithm version.

        **Returns:**
        The part file written, or `None` if there was nothing to write.
        """
        ids = np.asarray(ks_ids, dtype=np.int64)
        if len(ids) != len(results):
            raise ValueError(f"got {len(ids)} ks_ids for {len(results)} results")
        if not len(ids):
            return None

        columns: dict[str, list] = {}
        for r in results:
            for k in r:
                if k != _ID_COLUMN:
                    columns.setdefault(k, [])

        data: dict[str, list] = {_ID_COLUMN: ids.tolist()}
        for k in columns:
            col = []
            for r in results:
                v = r.get(k)
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                elif isinstance(v, np.generic):
                    v = v.item()
                col.append(v)
            data[k] = col

        table = pa.table(data)

        # Sort by the join key, so parts have tight per-row-group min/max
        # statistics and the file layout is deterministic.
        #
        # Measured honestly: for this store's ids this buys nothing on its own.
        # ks_id is a content hash, so ids are uniform over int64; sorting left
        # a 400,000-row store at exactly the same 4.59 MB on disk, and every
        # query timing was unchanged. It is kept because it is nearly free and
        # is the precondition for range pruning to ever be useful, not because
        # it currently pays.
        table = table.sort_by(_ID_COLUMN)

        d = self.quantity_dir(quantity, version)
        d.mkdir(parents=True, exist_ok=True)
        final = self._part_path(d)
        tmp = final.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        tmp.replace(final)
        return final


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _check_parallel_payload(fn) -> None:
    """Fail early and legibly on payloads that cannot cross a process boundary.

    Worker processes are started with `spawn`, so the payload is sent by
    reference and re-imported on the other side. A lambda, a closure or a
    function defined inside another function cannot be, and the resulting
    failure surfaces as `BrokenProcessPool` from deep inside concurrent.futures
    with nothing pointing at the cause.

    The related trap -- a calling script whose top-level code is not guarded by
    `if __name__ == "__main__":`, so each spawned child re-executes it -- is not
    checked here. There is no way to inspect the caller for the guard, and
    multiprocessing already raises its own explicit RuntimeError for it.
    """
    import pickle

    try:
        pickle.dumps(fn)
    except Exception as e:
        name = getattr(fn, "__qualname__", repr(fn))
        raise ValueError(
            f"The payload {name!r} cannot be sent to worker processes ({e}). "
            "With workers > 1 it must be a module-level function, not a "
            "lambda, a closure, or a function defined inside another function. "
            "Either move it to module level or call materialize(..., workers=1)."
        ) from e


def _make_pool(workers, fn):
    """Create the one process pool shared by a scan's nonempty batches.

    Spawned workers re-import the library. Keeping the pool alive across source
    batches pays that startup cost once per materialization rather than once per
    batch.
    """
    if workers <= 1:
        return None

    _check_parallel_payload(fn)

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    # fork is unsafe here: the compiled dependencies segfault in a forked child
    ctx = mp.get_context("spawn")
    return ProcessPoolExecutor(max_workers=workers, mp_context=ctx)


def _run_batch(vertex_list, fn, chunksize, pool=None):
    if pool is None:
        return [_safe(fn, v) for v in vertex_list]
    return list(pool.map(_Task(fn), vertex_list, chunksize=chunksize))


class _Task:
    """Picklable wrapper so the payload can be a module-level function."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, vertices):
        return _safe(self.fn, vertices)


def _safe(fn, vertices):
    try:
        out = fn(vertices)
    except Unsupported as e:
        # A fact about the geometry, not a failure. Recorded so the row is not
        # retried, and counted apart from errors.
        return {UNSUPPORTED_COLUMN: str(e) or "unsupported"}
    except Exception as e:  # noqa: BLE001 - one bad geometry must not stop a scan
        return {ERROR_COLUMN: f"{type(e).__name__}: {e}"}
    if not isinstance(out, dict):
        raise TypeError(
            f"payload must return a dict of scalars/sequences, got {type(out).__name__}"
        )
    return out


def materialize(
    quantity: str,
    fn,
    *,
    store: DerivedStore,
    scan,
    version: int = 1,
    workers: int = 1,
    chunksize: int = 8,
    recompute: bool = False,
    store_errors: bool = True,
    on_progress=None,
) -> dict:
    """
    **Description:**
    Compute *quantity* for every geometry in *scan* that the store is missing.

    This is the "materialize the missing column" operation. Rows already
    present are skipped, so an interrupted run resumes where it stopped and a
    completed run is a no-op.

    **Arguments:**
    - `quantity`: Name of the derived quantity.
    - `fn`: Callable `(vertices) -> dict`. Must be importable by name when
        `workers > 1`, since it is sent to worker processes.

        :::warning
        With `workers > 1`, worker processes are started with `spawn` and
        therefore **re-import the calling module**. Any top-level side effect in
        that module runs again in every worker. A script that clears its output
        directory at module level will have each worker clear it too, silently
        destroying results as the run proceeds -- and Python raises no error,
        because the child never starts a further process. Put everything except
        imports and definitions inside `if __name__ == "__main__":`.
        :::
    - `store`: The :class:`DerivedStore` to read and write.
    - `scan`: Iterable of :class:`~cytools.dataset.PolytopeBatch`, e.g. from
        :func:`~cytools.dataset.scan_batches`.
    - `version`: Algorithm version. Bump it to recompute under a new
        implementation without discarding the old results.
    - `workers`: Worker processes. 1 runs in-process.
    - `chunksize`: Geometries per task when parallel.
    - `recompute`: Compute every row, ignoring what is stored.
    - `store_errors`: Record failed geometries so they are not retried on the
        next run. Set `False` to leave them to be retried.
    - `on_progress`: Optional `(summary_dict) -> None`, called per batch.

    **Returns:**
    *(dict)* Counts: `requested`, `computed`, `skipped`, `unsupported`,
    `failed`. `skipped` counts rows already present in the store;
    `unsupported` counts rows whose payload raised :class:`Unsupported`,
    i.e. the geometry cannot support the requested work. The four outcomes are
    mutually exclusive and sum to `requested`.
    """
    totals = {
        "requested": 0,
        "computed": 0,
        "skipped": 0,
        "unsupported": 0,
        "failed": 0,
    }

    # One read of the id column per run rather than per batch. This is
    # I/O-optimal when most ids will be tested anyway, but it holds the whole
    # store's ids in memory -- about 8 bytes per stored row, so ~600 MB at
    # 8,000,000 rows. For a store far larger than that, or to check a handful
    # of ids interactively, use DerivedStore.missing(), which works one part at
    # a time.
    known = (
        np.empty(0, dtype=np.int64) if recompute else store.known_ids(quantity, version)
    )

    pool = None
    try:
        for batch in scan:
            ids = np.asarray(batch.ks_ids, dtype=np.int64)
            totals["requested"] += len(ids)

            if len(known):
                # known_ids() is sorted. Binary search avoids asking np.isin to
                # preprocess a many-million-row cache index for every 2k-row
                # source batch.
                positions = np.searchsorted(known, ids)
                present = positions < len(known)
                present[present] = known[positions[present]] == ids[present]
                todo = np.flatnonzero(~present)
            else:
                todo = np.arange(len(ids))

            totals["skipped"] += len(ids) - len(todo)
            if not len(todo):
                if on_progress:
                    on_progress(dict(totals))
                continue

            if workers > 1 and pool is None:
                pool = _make_pool(workers, fn)
            vertex_list = [batch.vertices(int(i)) for i in todo]
            results = _run_batch(vertex_list, fn, chunksize, pool=pool)

            ok_ids, ok_results = [], []
            for i, res in zip(todo, results):
                failed = ERROR_COLUMN in res
                if failed:
                    totals["failed"] += 1
                    if not store_errors:
                        continue
                elif UNSUPPORTED_COLUMN in res:
                    # Written regardless of store_errors: an unsupported geometry
                    # will still be unsupported next run, so retrying is waste.
                    totals["unsupported"] += 1
                else:
                    totals["computed"] += 1
                ok_ids.append(int(ids[i]))
                ok_results.append(res)

            if ok_ids:
                store.write(quantity, ok_ids, ok_results, version)
                # Do not grow the in-memory id index on a cold sweep. KS scan ids
                # are unique, and repeatedly unioning every batch made both time
                # and memory grow with the output (quadratic copying over a long
                # run). Duplicate ids from a custom scan are harmless: append-only
                # parts retain them and read() deterministically keeps the latest.

            if on_progress:
                on_progress(dict(totals))
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    return totals
