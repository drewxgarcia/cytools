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

    <root>/<quantity>/v<version>/part-<hex>.parquet

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
    # {'requested': 10000, 'computed': 10000, 'skipped': 0, 'failed': 0}

    # run it again: nothing left to do
    summary = materialize(...)
    # {'requested': 10000, 'computed': 0, 'skipped': 10000, 'failed': 0}

    table = store.read("hodge")          # everything computed so far
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["DerivedStore", "materialize"]

_ID_COLUMN = "ks_id"
_ERROR_COLUMN = "error"


def _resolve_root(root: Path | str | None) -> Path:
    if root is None:
        env = os.environ.get("CYTOOLS_DERIVED_DIR")
        if not env:
            raise ValueError(
                "No derived-results directory configured. Pass root= or set the "
                "CYTOOLS_DERIVED_DIR environment variable."
            )
        root = env
    return Path(root).expanduser()


class DerivedStore:
    """
    **Description:**
    A directory of derived quantities, keyed by `ks_id`.

    **Arguments:**
    - `root`: Directory to hold the store. Defaults to `CYTOOLS_DERIVED_DIR`.
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
            p.name for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")
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
        return sorted(d.glob("part-*.parquet"))

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
        """Which of *ks_ids* are not yet stored, in the order given."""
        ids = np.asarray(ks_ids, dtype=np.int64)
        known = self.known_ids(quantity, version)
        if not len(known):
            return ids
        return ids[~np.isin(ids, known)]

    def read(
        self,
        quantity: str,
        version: int = 1,
        ks_ids=None,
    ) -> pa.Table:
        """
        **Description:**
        Read stored results for *quantity*, optionally restricted to `ks_ids`.

        Duplicate ids -- possible if two runs computed the same row -- are
        reduced to their first occurrence, so the result is one row per id.
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
        _, first = np.unique(ids, return_index=True)
        keep = np.sort(first)

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

        d = self.quantity_dir(quantity, version)
        final = d / f"part-{uuid.uuid4().hex}.parquet"
        tmp = final.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        tmp.replace(final)

        for p in parts:
            try:
                p.unlink()
            except OSError:
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
            raise ValueError(
                f"got {len(ids)} ks_ids for {len(results)} results"
            )
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

        d = self.quantity_dir(quantity, version)
        d.mkdir(parents=True, exist_ok=True)
        final = d / f"part-{uuid.uuid4().hex}.parquet"
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


def _run_batch(vertex_list, fn, workers, chunksize):
    if workers <= 1:
        return [_safe(fn, v) for v in vertex_list]

    _check_parallel_payload(fn)

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    # fork is unsafe here: the compiled dependencies segfault in a forked child
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        return list(ex.map(_Task(fn), vertex_list, chunksize=chunksize))


class _Task:
    """Picklable wrapper so the payload can be a module-level function."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, vertices):
        return _safe(self.fn, vertices)


def _safe(fn, vertices):
    try:
        out = fn(vertices)
    except Exception as e:  # noqa: BLE001 - one bad geometry must not stop a scan
        return {_ERROR_COLUMN: f"{type(e).__name__}: {e}"}
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
    *(dict)* Counts: `requested`, `computed`, `skipped`, `failed`.
    """
    totals = {"requested": 0, "computed": 0, "skipped": 0, "failed": 0}

    # one read of the id column per run rather than per batch
    known = (
        np.empty(0, dtype=np.int64)
        if recompute
        else store.known_ids(quantity, version)
    )

    for batch in scan:
        ids = np.asarray(batch.ks_ids, dtype=np.int64)
        totals["requested"] += len(ids)

        if len(known):
            todo = np.flatnonzero(~np.isin(ids, known))
        else:
            todo = np.arange(len(ids))

        totals["skipped"] += len(ids) - len(todo)
        if not len(todo):
            if on_progress:
                on_progress(dict(totals))
            continue

        vertex_list = [batch.vertices(int(i)) for i in todo]
        results = _run_batch(vertex_list, fn, workers, chunksize)

        ok_ids, ok_results = [], []
        for i, res in zip(todo, results):
            failed = _ERROR_COLUMN in res
            if failed:
                totals["failed"] += 1
                if not store_errors:
                    continue
            else:
                totals["computed"] += 1
            ok_ids.append(int(ids[i]))
            ok_results.append(res)

        if ok_ids:
            store.write(quantity, ok_ids, ok_results, version)
            known = np.union1d(known, np.asarray(ok_ids, dtype=np.int64))

        if on_progress:
            on_progress(dict(totals))

    return totals
