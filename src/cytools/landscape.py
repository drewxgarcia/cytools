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
"""
Landscape scans: name the columns you want, get a DataFrame.

    from cytools import scan

    scan(["h11", "chi"], n=10_000)                 # a Parquet read; no Polytope
    scan(["h11", "n_intnums"], n=10_000)           # triangulates; builds no CY
    scan(["divisor_volumes"], h11=range(50, 300), n=1000)  # full chain to tip

Everything below exists so that those three lines are the whole interface.

Why columns rather than payload functions
-----------------------------------------
A :class:`Geometry` computes each step of the
``Polytope -> Triangulation -> CalabiYau -> cones`` chain lazily and at most
once. Asking for a set of columns therefore computes exactly their dependency
closure and nothing else, which means **there is no slow path available to
ask for**:

- Requesting only database-backed columns never constructs a `Polytope`.
- Requesting `n_intnums`, `n_simplices` or `n_points` never constructs a
  `CalabiYau`, so those are available for a non-favorable polytope too.
- Requesting a column that genuinely needs the threefold -- `n_cy_intnums`,
  `divisor_volumes`, `cy_volume`, `tip` -- costs a favorability test and no
  more on the ~48% of the database that has no Calabi-Yau (see below).

Note that `n_intnums` counts the *ambient toric variety's* intersection
numbers and `n_cy_intnums` the *threefold's*. They are different quantities
(742 against 178 on one measured geometry), and only the latter is what the
axion literature means by "the intersection numbers".

None of that is policy the caller opts into; it falls out of what was asked
for. The scan is also parallel, resumable and persisted without being told to
be: `batch_size`, `chunksize` and worker count are measured optima supplied
here rather than parameters, and results are keyed by `ks_id` in a
:class:`~cytools.store.DerivedStore` that defaults to a user cache directory.

Favorability is a precondition, not a filter
--------------------------------------------
`CalabiYau.__init__` already refuses a non-favorable polytope -- it just checks
*after* the triangulation has been built. :attr:`Geometry.cy` performs the same
check before triangulating and raises :class:`~cytools.store.Unsupported`, which
`materialize` records as `unsupported` rather than as an error. So the ~48% of
the database that has no Calabi-Yau hypersurface costs a favorability test
instead of a full pipeline, and only the columns that actually need a CY are
affected.

The M-lattice trap
------------------
The database's `h11`/`h12`/`euler_characteristic` columns are M-lattice, the
opposite of CYTools' ``lattice="N"`` default. This module speaks N-lattice
throughout -- both the `h11`/`h21`/`chi` columns and the filters of the same
name -- and performs the inversion internally. The convention never reaches
the caller.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

from cytools.dataset import scan_batches
from cytools.store import (
    ERROR_COLUMN,
    UNSUPPORTED_COLUMN,
    DerivedStore,
    Unsupported,
    materialize,
)

__all__ = [
    "Geometry",
    "Unsupported",
    "quantity",
    "quantities",
    "scan",
    "status",
    "sweep",
]

# Measured optima, not preferences. batch_size 2048 vs 512 is 95 vs 62
# geoms/s; chunksize 8 and (cores - 2) workers hold 91% scaling efficiency at
# 4 workers. Deliberately not parameters.
_BATCH_SIZE = 2048
_CHUNKSIZE = 8
_MAX_AUTO_WORKERS = 8

# `scan` is the interactive verb and returns a DataFrame. Past this many rows
# that is the wrong tool, and silently truncating would be worse than saying so.
_MAX_COLLECT = 2_000_000

_ID = "ks_id"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Quantity:
    name: str
    fn: Callable
    source: str  # "database" (also derivable) or "computed"
    doc: str


_QUANTITIES: dict[str, _Quantity] = {}
_ModuliMode = Literal["tip", "sampled"]


def quantity(fn=None, *, name: str | None = None, source: str = "computed"):
    """
    **Description:**
    Register a storable column on :class:`Geometry`.

    The decorated function takes a `Geometry` and returns a scalar or array --
    anything a Parquet column can hold. It becomes a memoized attribute, so
    other quantities may depend on it freely without recomputation, and its
    name becomes requestable from :func:`scan` and :func:`sweep`.

    Built-in and user-defined computed columns use the same decorator. Direct
    Parquet mappings are reserved for CYTools because they also require a
    matching batch-buffer implementation.

    **Arguments:**
    - `fn`: The function to register. Usually supplied by decoration.
    - `name`: Column name. Defaults to the function's name.
    - `source`: `"computed"` for user quantities. `"database"` is reserved for
        CYTools' built-in mappings to the Kreuzer-Skarke Parquet schema.

    **Returns:**
    *(Callable)* The function, unchanged.

    **Example:**
    ```python {1}
    @quantity
    def max_vertex_coordinate(g):
        "Largest absolute coordinate among the vertices."
        return abs(g.polytope.vertices()).max()

    scan(["h11", "max_vertex_coordinate"], n=100)
    ```
    """

    def register(f):
        qname = name or f.__name__
        if not isinstance(qname, str) or not qname.isidentifier():
            raise ValueError(
                f"quantity names must be valid Python identifiers, got {qname!r}"
            )
        if source not in {"computed", "database"}:
            raise ValueError(
                f"quantity source must be 'computed' or 'database', got {source!r}"
            )
        external = getattr(f, "__module__", None) != __name__
        if source == "database" and external:
            raise ValueError(
                "source='database' is reserved for CYTools' built-in Parquet "
                "mappings; notebook quantities must use source='computed'."
            )
        existing = _QUANTITIES.get(qname)
        if (
            existing is not None
            and getattr(existing.fn, "__module__", None) == __name__
            and external
        ):
            raise ValueError(f"cannot replace built-in quantity {qname!r}")
        _QUANTITIES[qname] = _Quantity(
            name=qname,
            fn=f,
            source=source,
            doc=(f.__doc__ or "").strip().split("\n")[0],
        )
        return f

    return register if fn is None else register(fn)


def quantities():
    """
    **Description:**
    Every registered column, as a DataFrame -- what :func:`scan` will accept.

    **Arguments:**
    None.

    **Returns:**
    *(pandas.DataFrame)* Columns `name`, `source`, `parallel_safe`,
    `description`.
    """
    import pandas as pd

    rows = [
        {
            "name": q.name,
            "source": q.source,
            "parallel_safe": getattr(q.fn, "__module__", None) == __name__,
            "description": q.doc,
        }
        for q in sorted(_QUANTITIES.values(), key=lambda q: (q.source, q.name))
    ]
    return pd.DataFrame(
        rows, columns=["name", "source", "parallel_safe", "description"]
    )


def status(derived_dir=None):
    """
    **Description:**
    What is already computed and cached, as a DataFrame.

    **Arguments:**
    - `derived_dir`: Store location. Defaults to `CYTOOLS_DERIVED_DIR`, then to
        a user cache directory.

    **Returns:**
    *(pandas.DataFrame)* Columns `columns`, `version`, `n_rows`, `megabytes`.
    """
    import pandas as pd

    store = DerivedStore(derived_dir)
    rows = []
    for q in store.quantities() or []:
        for v in store.versions(q):
            st = store.stats(q, v)
            rows.append(
                {
                    "columns": q,
                    "version": v,
                    "n_rows": st["n_rows"],
                    "megabytes": round(st["bytes"] / 1e6, 3),
                }
            )
    return pd.DataFrame(rows, columns=["columns", "version", "n_rows", "megabytes"])


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class Geometry:
    """
    **Description:**
    One Kreuzer-Skarke geometry, built lazily and at most once per step.

    The single place the `Polytope -> Triangulation -> CalabiYau -> cones`
    chain is written. Intermediates are `cached_property`; storable columns are
    registered with :func:`quantity` and resolved through `__getattr__`, which
    memoizes into the instance dict on first access.

    A polytope generally admits many fine, regular, star triangulations, and
    each one is a different Calabi-Yau. `triangulation_seed` picks which:
    `None` takes the canonical `Polytope.triangulate()`, and any integer draws
    a reproducible random FRST. Two `Geometry` objects with the same vertices
    and the same seed are the same geometry, which is what lets a scan over
    many triangulations per polytope stay resumable.

    **Arguments:**
    - `vertices`: The polytope's vertices, as accepted by
        :class:`~cytools.polytope.Polytope`.
    - `triangulation_seed`: `None` for the canonical triangulation, or an
        integer seeding a reproducible random FRST.
    - `moduli_seed`: `None` to evaluate volumes at the tip of the stretched
        Kähler cone, or an integer seeding a reproducible interior direction.

    **Example:**
    ```python {2}
    g = Geometry(vertices)
    g.n_intnums        # triangulates and builds the ambient variety on demand
    g.n_intnums        # cached

    Geometry(vertices, triangulation_seed=7).divisor_volumes   # a different CY
    ```
    """

    def __init__(
        self,
        vertices,
        *,
        triangulation_seed: int | None = None,
        moduli_seed: int | None = None,
    ):
        self._vertices = np.asarray(vertices)
        self._triangulation_seed = triangulation_seed
        self._moduli_seed = moduli_seed

    def __repr__(self):
        seed = self._triangulation_seed
        suffix = "" if seed is None else f", triangulation_seed={seed}"
        return f"Geometry({len(self._vertices)} vertices{suffix})"

    def __getattr__(self, name):
        # Only reached when normal lookup fails, so cached values and
        # cached_property intermediates shadow this permanently.
        q = _QUANTITIES.get(name)
        if q is None:
            raise AttributeError(
                f"{name!r} is not a registered quantity. "
                f"Available: {', '.join(sorted(_QUANTITIES))}"
            )
        value = q.fn(self)
        self.__dict__[name] = value
        return value

    # -- intermediates: objects, not columns -------------------------------

    @cached_property
    def polytope(self):
        """The `Polytope`."""
        from cytools.polytope import Polytope

        return Polytope(self._vertices)

    @cached_property
    def triangulation(self):
        """A fine, regular, star triangulation.

        `Polytope.triangulate()` does not memoize, so this is what keeps a
        payload wanting both a triangulation and a CY from building two.

        With a `triangulation_seed` this draws a random FRST instead of the
        canonical one. The seed is passed straight through to the sampler, so
        the same seed always yields the same triangulation -- the property the
        store relies on to skip work it has already done.
        """
        if self._triangulation_seed is None:
            return self.polytope.triangulate(verbosity=0)
        return next(
            self.polytope.random_triangulations_fast(
                N=1,
                seed=int(self._triangulation_seed),
            )
        )

    @cached_property
    def toric_variety(self):
        """The ambient toric variety."""
        return self.triangulation.get_toric_variety()

    @cached_property
    def cy(self):
        """The Calabi-Yau hypersurface.

        Checks favorability *before* triangulating. `CalabiYau.__init__`
        imposes the same requirement but only after the triangulation exists,
        so honoring it here is what makes a non-favorable geometry cost a
        favorability test rather than a full pipeline.
        """
        if not self.is_favorable:
            raise Unsupported(
                "non-favorable polytope: no CY outside experimental features"
            )
        return self.triangulation.get_cy()

    @cached_property
    def intersection_numbers(self):
        """Ambient intersection numbers in the current basis; a dict, not a column.

        The *toric variety's*, so requesting `n_intnums` does not construct a
        `CalabiYau` and works for a non-favorable polytope. For the threefold's
        own triple intersections -- a different and strictly smaller set -- see
        :attr:`cy_intersection_numbers`.
        """
        return self.toric_variety.intersection_numbers(in_basis=True)

    @cached_property
    def cy_intersection_numbers(self):
        """Triple intersection numbers *on the Calabi-Yau*, dense and symmetrised.

        Distinct from :attr:`intersection_numbers`: these live on the threefold
        rather than the ambient fourfold, and there are far fewer of them (224
        against 745 on one measured 14-vertex geometry). These are the ones that
        set the Kahler potential, so they are what the type IIB axion
        literature means by "the intersection numbers".

        Kept as the sparse dict from `cy.intersection_numbers(in_basis=True)`.
        The fan can supply the same numbers as a dense array -- verified equal
        entry for entry (0 mismatches across 497/451/511/512 entries) -- and it
        looked like the two routes were duplicating work. They are not: the
        dense form is `(h11, h11, h11)`, which `compute_divisor_volumes` needs
        and the fan caches, while this sparse form holds only ~2,100 entries.
        Routing this through the dense tensor measured 0.89x, because scanning
        2.7e7 entries for nonzeros costs more than `len()` on a dict.
        """
        return self.cy.intersection_numbers(in_basis=True)

    @cached_property
    def mori_cone(self):
        """The Mori cone in the current basis."""
        return self.cy.toric_mori_cone(in_basis=True)

    @cached_property
    def moduli_point(self):
        """Where in Kahler moduli space the volume columns are evaluated.

        With no `moduli_seed` this is the tip of the stretched Kahler cone: a
        specific distinguished point and the historical default. With a seed it
        is a reproducible random direction, suitable for ensemble studies that
        should not evaluate every geometry at the same distinguished point.

        Divisor volumes are homogeneous under rescaling the Kahler parameters,
        so only the direction matters; sampling is done on the bounded slice
        `{x in K : g.x = 1}` for a grading vector `g`, which is positive on the
        cone and therefore makes the slice a polytope rather than a cone. The
        sampled ray is then scaled to the same `min(H.x) = 1` stretched-cone
        convention as :attr:`tip`.
        """
        if self._moduli_seed is None:
            return self.tip
        return _sample_kahler_direction(
            self.kahler_cone, int(self._moduli_seed), start=self.tip
        )

    @cached_property
    def kahler_cone(self):
        """The Kahler cone inferred from toric geometry."""
        return self.cy.toric_kahler_cone()


# ---------------------------------------------------------------------------
# Built-in columns
# ---------------------------------------------------------------------------
#
# Registered exactly like a user's own. The `source="database"` ones are served
# from the scan's buffers; their bodies run only when a Geometry is used
# directly.


@quantity(source="database")
def h11(g):
    """N-lattice Hodge number h^{1,1}."""
    return int(g.polytope.h11(lattice="N"))


@quantity(source="database")
def h21(g):
    """N-lattice Hodge number h^{2,1}."""
    return int(g.polytope.h21(lattice="N"))


@quantity(source="database")
def chi(g):
    """N-lattice Euler characteristic."""
    return int(g.polytope.chi(lattice="N"))


@quantity(source="database")
def n_vertices(g):
    """Number of vertices of the polytope."""
    return len(g.polytope.vertices())


@quantity
def is_favorable(g):
    """Whether the polytope is favorable in the N lattice."""
    return bool(g.polytope.is_favorable(lattice="N"))


@quantity(source="database")
def n_points(g):
    """Number of lattice points of the polytope."""
    return len(g.polytope.points())


@quantity(source="database")
def n_facets(g):
    """Number of facets of the polytope."""
    return len(g.polytope.facets())


@quantity(source="database")
def n_dual_points(g):
    """Number of lattice points in the dual polytope."""
    return len(g.polytope.dual().points())


@quantity
def triangulation_hash(g):
    """Content hash of the triangulation, for spotting repeated draws.

    Sampling N triangulations of a polytope does not guarantee N *distinct*
    ones -- the sampler can return the same FRST twice -- and the papers that
    scan "10 triangulations per polytope" mean distinct geometries. Recording
    this makes duplicates detectable after the fact rather than silently
    inflating an ensemble.
    """
    simplices = np.asarray(g.triangulation.simplices(), dtype=np.int64)
    simplices = simplices[np.lexsort(simplices.T[::-1])]  # order-independent
    return _hash_bytes(simplices.tobytes())


@quantity
def n_simplices(g):
    """Number of simplices in the triangulation."""
    return len(g.triangulation.simplices())


@quantity
def n_intnums(g):
    """Number of nonzero intersection numbers of the ambient toric variety."""
    return len(g.intersection_numbers)


@quantity
def n_cy_intnums(g):
    """Number of nonzero triple intersection numbers on the Calabi-Yau."""
    return len(g.cy_intersection_numbers)


@quantity
def n_mori_rays(g):
    """Number of generating rays of the Mori cone."""
    return len(g.mori_cone.rays())


@quantity
def tip(g):
    """Tip of the stretched Kahler cone, or None if the solver failed.

    A `None` here is a recorded outcome rather than an exception: the geometry
    is fine, the optimizer did not converge. See `tip_backend` for the accuracy
    of a tip that was found.
    """
    t = g.kahler_cone.tip_of_stretched_cone(1, show_hints=False)
    return None if t is None else np.asarray(t, dtype=float)


@quantity
def tip_backend(g):
    """Which optimizer produced `tip`, and hence whether it is exact.

    Mirrors the backend choice in `Cone.tip_of_stretched_cone`. Recorded
    because at ambient dimension >= 25 the tip is an LP *approximation* unless
    Mosek is licensed, and results should state their own accuracy rather than
    leave it to be inferred from the h11 of each row.
    """
    import cytools.config as config

    if g.kahler_cone.ambient_dim() < 25:
        return "osqp"  # exact QP
    return "mosek" if config.mosek_is_activated() else "highs"  # LP approximation


@quantity
def kahler_point(g):
    """Kähler parameters used to evaluate the volume columns."""
    return g.moduli_point


@quantity
def divisor_volumes(g):
    """Prime-toric-divisor volumes at the selected Kähler-moduli point.

    The central quantity of the type IIB axion literature: divisor volumes set
    instanton actions, and hence axion masses and decay constants.
    """
    if g.moduli_point is None:
        return None
    return np.asarray(g.cy.compute_divisor_volumes(g.moduli_point), dtype=float)


@quantity
def cy_volume(g):
    """Volume of the Calabi-Yau at the selected Kähler-moduli point."""
    if g.moduli_point is None:
        return None
    return float(g.cy.compute_cy_volume(g.moduli_point))


# ---------------------------------------------------------------------------
# The payload that crosses the process boundary
# ---------------------------------------------------------------------------


class _Payload:
    """Picklable by construction: it holds column names, nothing else.

    Quantities are looked up in the registry inside the worker, so no function
    object is ever pickled. That is what removes the spawn/pickle trap from the
    public surface -- there is no user-supplied callable to fail to serialize.
    """

    def __init__(self, columns, moduli: _ModuliMode = "tip"):
        self.columns = tuple(columns)
        self.moduli = moduli

    def __call__(self, item):
        """Compute each column independently, keeping whatever succeeds.

        *item* is either a vertex array or a `(vertices, triangulation_seed)`
        pair, which is how a multi-triangulation scan tells the worker which
        of a polytope's Calabi-Yaus this row is.

        `Unsupported` is handled per column rather than per row: a non-favorable
        polytope still has a point count and a favorability flag, and throwing
        those away because some *other* column needed a CY would lose real
        data. A genuine exception is left to propagate, so `_safe` marks the
        whole row an error -- an error means a bug worth seeing, unlike a
        geometry that simply does not support the question.
        """
        if isinstance(item, tuple):
            vertices, seed = item
        else:
            vertices, seed = item, None

        # Deterministic from the vertices, so a resumed scan reproduces the
        # same moduli point rather than drawing a fresh one.
        moduli_seed = None
        if self.moduli == "sampled":
            # Canonical little-endian representation keeps the sample stable
            # across NumPy integer widths and machine architectures.
            seed_material = np.asarray(vertices, dtype="<i8").tobytes(order="C")
            if seed is not None:
                seed_material += int(seed).to_bytes(8, "little", signed=True)
            moduli_seed = _hash_bytes(seed_material) & 0x7FFFFFFF

        g = Geometry(vertices, triangulation_seed=seed, moduli_seed=moduli_seed)
        out, reason = {}, None

        for c in self.columns:
            try:
                out[c] = getattr(g, c)
            except Unsupported as e:
                reason = reason or str(e)

        if reason is not None:
            out[UNSUPPORTED_COLUMN] = reason
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_columns(columns) -> list[str]:
    if isinstance(columns, str):
        columns = [columns]
    cols = list(dict.fromkeys(columns))  # de-duplicate, keep order
    unknown = [c for c in cols if c not in _QUANTITIES]
    if unknown:
        raise ValueError(
            f"unknown column(s) {unknown}. "
            f"Available: {', '.join(sorted(_QUANTITIES))}. "
            "Register your own with @cytools.quantity."
        )
    if not cols:
        raise ValueError("no columns requested")
    return cols


def _scan_kwargs(*, n, batch_size, db_dir, filters) -> dict:
    """Translate N-lattice filters to the database's M-lattice columns."""
    known = {
        "n_vertices",
        "h11",
        "h21",
        "chi",
        "n_facets",
        "n_points",
        "n_dual_points",
        "seed",
        "stream",
        "hf_token",
    }
    unknown = set(filters) - known
    if unknown:
        raise TypeError(f"unexpected filter argument(s): {sorted(unknown)}")

    out = {"n": n, "batch_size": batch_size}
    if db_dir is not None:
        out["db_dir"] = db_dir
    for k in (
        "n_vertices",
        "n_facets",
        "n_points",
        "n_dual_points",
        "seed",
        "stream",
        "hf_token",
    ):
        if k in filters:
            out[k] = filters[k]

    # N-lattice in, M-lattice out. See the module docstring.
    if "h11" in filters:
        out["h12"] = filters["h11"]
    if "h21" in filters:
        out["h11"] = filters["h21"]
    if "chi" in filters:
        c = filters["chi"]
        out["chi"] = (
            -int(c) if isinstance(c, (int, np.integer)) else [-int(v) for v in c]
        )
    return out


class _ExpandedBatch:
    """A `PolytopeBatch` seen as *n_triangulations* geometries per polytope.

    Presents exactly the surface `materialize` consumes -- `ks_ids`, `len()`
    and `vertices(i)` -- so multiplicity needs no change to the store contract.
    `vertices(i)` returns a `(vertices, seed)` pair, and the ids are derived so
    that triangulation 0 keeps the polytope's own `ks_id`; a single-
    triangulation scan is therefore byte-identical to one that never knew about
    multiplicity.
    """

    def __init__(self, batch, n_triangulations: int):
        self._batch = batch
        self._n = int(n_triangulations)
        base = np.asarray(batch.ks_ids, dtype=np.int64)
        self._rows = np.repeat(np.arange(len(base)), self._n)
        self._k = np.tile(np.arange(self._n), len(base))
        self.ks_ids = np.array(
            [_mix(int(base[r]), int(k)) for r, k in zip(self._rows, self._k)],
            dtype=np.int64,
        )
        self.polytope_ids = base[self._rows]

    def __len__(self):
        return len(self.ks_ids)

    def vertices(self, i: int):
        row, k = int(self._rows[i]), int(self._k[i])
        seed = None if k == 0 else _seed(int(self.polytope_ids[i]), k)
        return self._batch.vertices(row), seed

    def source_column(self, name):
        """A per-polytope database column, repeated across its triangulations."""
        return np.asarray(getattr(self._batch, name))[self._rows]

    @property
    def triangulation_indices(self):
        return self._k


# Requestable name -> the database column it comes from, and whether the
# N-lattice value is the negation of it. See the module docstring.
_DB_SOURCE = {
    "h11": ("h12", False),
    "h21": ("h11", False),
    "chi": ("euler_characteristic", True),
    "n_vertices": ("vertex_count", False),
    "n_facets": ("facet_count", False),
    "n_points": ("point_count", False),
    "n_dual_points": ("dual_point_count", False),
}


def _db_columns(batch, cols) -> dict:
    """Database-backed columns straight from the batch buffers, N-lattice.

    Works for a plain `PolytopeBatch` and for an `_ExpandedBatch`, which
    repeats each polytope's values across its triangulations.
    """
    fetch = getattr(batch, "source_column", None)
    if fetch is None:

        def fetch(name):
            return np.asarray(getattr(batch, name))

    out = {_ID: np.asarray(batch.ks_ids, dtype=np.int64)}

    # Provenance for a multi-triangulation scan: which polytope each row came
    # from (the join key back to the source) and which of its triangulations.
    if isinstance(batch, _ExpandedBatch):
        out["polytope_id"] = np.asarray(batch.polytope_ids, dtype=np.int64)
        out["triangulation_index"] = np.asarray(
            batch.triangulation_indices, dtype=np.int64
        )

    for c in cols:
        try:
            source, negate = _DB_SOURCE[c]
        except KeyError:  # pragma: no cover - guarded by _Quantity.source
            raise KeyError(c) from None
        values = np.asarray(fetch(source), dtype=np.int64)
        out[c] = -values if negate else values
    return out


_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3


def _hash_bytes(data: bytes) -> int:
    """A stable 64-bit content hash, as a signed int.

    `hashlib` rather than a Python-level FNV loop: a triangulation at
    h11 ~ 500 is ~100 kB of simplices, and hashing that byte by byte in Python
    would cost more than computing the triangulation did.
    """
    import hashlib

    digest = hashlib.blake2b(data, digest_size=8).digest()
    h = int.from_bytes(digest, "little")
    return h - (1 << 64) if h >= (1 << 63) else h


def _sample_kahler_direction(cone, seed: int, start=None, burn: int = 30):
    """A reproducible random direction inside *cone*, by hit-and-run.

    A grading vector is strictly positive on a pointed cone, so the slice
    `{x : g.x = 1}` is bounded and every hit-and-run interval is finite. Steps
    are projected onto `g.d = 0` so the walk stays on the slice. Returns None
    if no interior starting point is available. The selected ray is rescaled
    onto the `min(H.x) = 1` stretched-cone boundary before it is returned.
    """
    H = np.asarray(cone.hyperplanes(), dtype=float)
    grading = cone.find_grading_vector()
    if grading is None:
        return None
    g = np.asarray(grading, dtype=float)
    norm_squared = float(g @ g)
    if not np.isfinite(norm_squared) or norm_squared <= 0:
        return None

    x = start if start is not None else cone.tip_of_stretched_cone(1, show_hints=False)
    if x is None:
        return None
    x = np.asarray(x, dtype=float)
    denom = float(g @ x)
    if not np.isfinite(denom) or denom <= 0:
        return None
    x = x / denom

    rng = np.random.default_rng(seed)
    for _ in range(burn):
        d = rng.normal(size=len(x))
        d -= g * (g @ d) / norm_squared
        Hx, Hd = H @ x, H @ d
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = -Hx / Hd
        lo = ratios[Hd > 0].max() if np.any(Hd > 0) else -np.inf
        hi = ratios[Hd < 0].min() if np.any(Hd < 0) else np.inf
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            continue
        pad = 1e-9 * (hi - lo)
        x = x + rng.uniform(lo + pad, hi - pad) * d

    min_slack = float(np.min(H @ x))
    if not np.isfinite(min_slack) or min_slack <= 0:
        return None
    return x / min_slack


def _mix(ks_id: int, k: int) -> int:
    """Derive a geometry id for the k-th triangulation of one polytope.

    `k == 0` returns `ks_id` unchanged, so a scan that asks for a single
    triangulation keys its rows exactly as before -- existing stores stay
    valid, and the common case carries no extra machinery.
    """
    if k == 0:
        return int(ks_id)
    h = (int(ks_id) & 0xFFFFFFFFFFFFFFFF) ^ _FNV_OFFSET
    h = ((h * _FNV_PRIME) ^ (k & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    h = (h * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h - (1 << 64) if h >= (1 << 63) else h


def _seed(ks_id: int, k: int) -> int:
    """A reproducible sampler seed for the k-th triangulation of a polytope."""
    return abs(_mix(ks_id, k + 1)) % (2**31 - 1)


def _store_key(columns) -> str:
    """A readable, stable store key for one computed column set."""
    joined = "-".join(sorted(columns))
    if len(joined) <= 80:
        return joined
    import hashlib

    digest = hashlib.sha1(joined.encode()).hexdigest()[:8]
    return f"{joined[:70]}-{digest}"


def _resolve_workers(workers, columns) -> int:
    """Worker count, falling back to 1 when a column cannot survive `spawn`.

    Worker processes re-import the library but not the caller's notebook, so a
    quantity defined interactively does not exist on the other side. Rather
    than failing deep inside `concurrent.futures`, run in-process.
    """
    local = [
        c for c in columns if getattr(_QUANTITIES[c].fn, "__module__", None) != __name__
    ]
    if local:
        if workers is not None and int(workers) > 1:
            warnings.warn(
                "Notebook-defined quantities run in the notebook process, so "
                f"workers={workers} was reduced to 1 for {local}.",
                RuntimeWarning,
                stacklevel=3,
            )
        return 1
    if workers is not None:
        return max(1, int(workers))
    return min(_MAX_AUTO_WORKERS, max(1, (os.cpu_count() or 2) - 2))


def _progress_callback(progress, total):
    """Resolve the `progress` argument into a (callback, bar) pair.

    `False` silences it, a callable replaces it, and the default is a tqdm bar
    driven by materialize's per-batch summary.
    """
    if progress is False:
        return None, None
    if callable(progress):
        return progress, None

    from tqdm.auto import tqdm

    bar = tqdm(total=total, unit="geom")
    state = {"seen": 0}

    def cb(summary):
        seen = summary["requested"]
        bar.update(seen - state["seen"])
        state["seen"] = seen
        bar.set_postfix(
            computed=summary["computed"],
            cached=summary["skipped"],
            unsupported=summary["unsupported"],
            failed=summary["failed"],
            refresh=False,
        )

    return cb, bar


def _run(
    columns,
    *,
    n,
    workers,
    version,
    recompute,
    progress,
    db_dir,
    derived_dir,
    collect,
    max_rows,
    triangulations,
    moduli,
    filters,
):
    """Shared body of `scan` and `sweep`."""
    import pandas as pd

    cols = _resolve_columns(columns)
    version = int(version)
    if version < 0:
        raise ValueError(f"version must be non-negative, got {version}")
    triangulations = int(triangulations)
    if triangulations < 1:
        raise ValueError(f"triangulations must be at least 1, got {triangulations}")
    if moduli not in ("tip", "sampled"):
        raise ValueError(f"moduli must be 'tip' or 'sampled', got {moduli!r}")
    db_cols = [c for c in cols if _QUANTITIES[c].source == "database"]
    computed = [c for c in cols if _QUANTITIES[c].source != "database"]

    if triangulations > 1 and not computed:
        raise ValueError(
            "triangulations > 1 was requested but every column asked for comes "
            "straight from the database, so each triangulation would return an "
            "identical row. Ask for a column that depends on the "
            "triangulation, e.g. n_simplices or divisor_volumes."
        )

    kwargs = _scan_kwargs(n=n, batch_size=_BATCH_SIZE, db_dir=db_dir, filters=filters)

    # Nothing to compute: a Parquet read. No store, no workers, no Polytope.
    if not computed:
        frames, seen = [], 0
        for batch in scan_batches(**kwargs):
            seen += len(batch)
            if collect and seen > max_rows:
                raise ValueError(_too_many(seen, max_rows))
            if collect:
                frames.append(pd.DataFrame(_db_columns(batch, db_cols)))
        if not collect:
            return {
                "requested": seen,
                "computed": 0,
                "skipped": 0,
                "unsupported": 0,
                "failed": 0,
            }
        df = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=[_ID] + db_cols)
        )
        df.attrs["cytools"] = {
            "columns": cols,
            "version": version,
            "moduli": moduli,
            "requested": seen,
        }
        return df

    # Database-backed columns are read directly and must not fragment the
    # cache: `scan(["is_favorable"])` and
    # `scan(["h11", "is_favorable"])` reuse the same materialization.
    # The mode is part of the key: sampled volumes are a different quantity
    # from tip volumes and must never be served from the same cache.
    key = _store_key(computed) + ("" if moduli == "tip" else "-sampled")
    store = DerivedStore(derived_dir)
    workers = _resolve_workers(workers, computed)

    # Tee the source columns and ids out of the scan as materialize consumes
    # it, so they can be joined back without a second pass over Parquet.
    teed: list[dict] = []
    guard = {"seen": 0}

    def source():
        for batch in scan_batches(**kwargs):
            if triangulations > 1:
                batch = _ExpandedBatch(batch, triangulations)
            guard["seen"] += len(batch)
            if collect and guard["seen"] > max_rows:
                raise ValueError(_too_many(guard["seen"], max_rows))
            if collect:
                teed.append(_db_columns(batch, db_cols))
            yield batch

    cb, bar = _progress_callback(progress, n)
    try:
        summary = materialize(
            key,
            _Payload(computed, moduli=moduli),
            store=store,
            scan=source(),
            version=version,
            workers=workers,
            chunksize=_CHUNKSIZE,
            recompute=recompute,
            on_progress=cb,
        )
    finally:
        if bar is not None:
            bar.close()

    if not collect:
        return summary

    ids = (
        np.concatenate([t[_ID] for t in teed]) if teed else np.empty(0, dtype=np.int64)
    )
    table = store.read(key, version=version, ks_ids=ids)
    computed_df = table.to_pandas()
    src = (
        pd.concat([pd.DataFrame(t) for t in teed], ignore_index=True)
        if teed
        else pd.DataFrame(columns=[_ID] + db_cols)
    )
    # A left join preserves scan order, including when no source columns were
    # requested and the store's physical part order differs from the scan.
    df = src.merge(computed_df, on=_ID, how="left")
    for column in cols:
        if column not in df:
            df[column] = pd.NA
    status_cols = [c for c in (UNSUPPORTED_COLUMN, ERROR_COLUMN) if c in df]
    # Provenance is not "requested" but a multi-triangulation result cannot be
    # read without it: which polytope a row belongs to, and which of its
    # triangulations. Kept ahead of the requested columns.
    provenance = [c for c in ("polytope_id", "triangulation_index") if c in df]
    df = df[[_ID] + provenance + cols + status_cols]
    df.attrs["cytools"] = {
        "columns": cols,
        "version": version,
        "triangulations": triangulations,
        "moduli": moduli,
        **summary,
    }
    return df


def _too_many(seen, max_rows) -> str:
    return (
        f"this scan reached {seen:,} rows, past the {max_rows:,}-row limit for "
        "collecting into a DataFrame. Narrow it with n= or a filter, or use "
        "cytools.sweep(...) to compute and store without collecting."
    )


# ---------------------------------------------------------------------------
# The two verbs
# ---------------------------------------------------------------------------


def scan(
    columns,
    *,
    n: int | None = None,
    workers: int | None = None,
    version: int = 1,
    triangulations: int = 1,
    moduli: _ModuliMode = "tip",
    recompute: bool = False,
    progress=None,
    db_dir=None,
    derived_dir=None,
    max_rows: int = _MAX_COLLECT,
    **filters,
):
    """
    **Description:**
    Compute *columns* over the Kreuzer-Skarke database and return a DataFrame.

    Parallel, resumable and cached without being asked: results are keyed by
    `ks_id`, so re-running the same scan recomputes nothing. Only the
    dependency closure of the requested columns is evaluated, so asking for
    less genuinely costs less.

    **Arguments:**
    - `columns`: Column name or list of names. See :func:`quantities`.
    - `n`: Maximum geometries to scan. `None` scans everything matching.
    - `workers`: Worker processes. Defaults to at most 8, and to 1 when a
        requested column was defined outside `cytools.landscape` (as in a
        notebook cell).
    - `version`: Cache schema/algorithm version. Bump this when the meaning or
        implementation of a custom quantity changes.
    - `triangulations`: How many fine, regular, star triangulations to draw
        per polytope. A polytope generally admits many, and each is a distinct
        Calabi-Yau; the published ensembles sample 10-1000 of them. Index 0 is
        always the canonical `Polytope.triangulate()`, so raising this reuses
        rather than invalidates what a previous scan computed. With more than
        one, the result carries `polytope_id` and `triangulation_index`, and
        `triangulation_hash` is worth requesting to spot repeated draws.
    - `moduli`: `"tip"` evaluates volume columns at the tip of the stretched
        Kähler cone. `"sampled"` uses one deterministic interior direction per
        geometry, rescales it to the same minimum curve-volume convention, and
        keeps those results in a separate cache. Request `kahler_point` to
        retain the coordinates used.
    - `recompute`: Ignore cached results and compute every row again.
    - `progress`: `False` to silence it, or a callable taking the running
        summary dict to report progress your own way. Defaults to a `tqdm` bar.
    - `db_dir`: Database directory. Defaults to `CYTOOLS_DB_DIR`.
    - `derived_dir`: Store directory. Defaults to `CYTOOLS_DERIVED_DIR`, then a
        user cache directory.
    - `max_rows`: Refuse to collect more than this many rows.
    - `**filters`: `n_vertices`, `h11`, `h21`, `chi`, `n_facets`, `n_points`,
        `n_dual_points`. Each takes a value or an iterable of values. The Hodge
        filters are **N-lattice**, matching the columns of the same name.

    **Returns:**
    *(pandas.DataFrame)* One row per geometry scanned, indexed by position and
    carrying `ks_id`. Geometries that could not support the requested columns
    carry an `unsupported` reason; ones that failed carry an `error`.

    **Example:**
    ```python {3}
    from cytools import scan

    scan(["h11", "n_intnums"], n=1000)
    scan(["divisor_volumes", "tip_backend"], h11=range(50, 80))
    ```
    """
    return _run(
        columns,
        n=n,
        workers=workers,
        version=version,
        recompute=recompute,
        progress=progress,
        db_dir=db_dir,
        derived_dir=derived_dir,
        collect=True,
        max_rows=max_rows,
        triangulations=triangulations,
        moduli=moduli,
        filters=filters,
    )


def sweep(
    columns,
    *,
    n: int | None = None,
    workers: int | None = None,
    version: int = 1,
    triangulations: int = 1,
    moduli: _ModuliMode = "tip",
    recompute: bool = False,
    progress=None,
    db_dir=None,
    derived_dir=None,
    **filters,
):
    """
    **Description:**
    Like :func:`scan`, but for landscape-scale runs: compute and store without
    collecting, and return counts.

    Interrupting is safe. Results are committed per batch and keyed by `ks_id`,
    so re-running the same call resumes rather than restarting. Read the
    results afterwards with the same :func:`scan` call, or from
    :class:`~cytools.store.DerivedStore` directly.

    **Arguments:**
    As :func:`scan`, without `max_rows`.

    **Returns:**
    *(dict)* Counts: `requested`, `computed`, `skipped` (already cached),
    `unsupported` (no such geometry, e.g. non-favorable), `failed`.

    **Example:**
    ```python {3}
    from cytools import sweep

    sweep(["divisor_volumes"], n=1_000_000)
    ```
    """
    return _run(
        columns,
        n=n,
        workers=workers,
        version=version,
        recompute=recompute,
        progress=progress,
        db_dir=db_dir,
        derived_dir=derived_dir,
        collect=False,
        max_rows=None,
        triangulations=triangulations,
        moduli=moduli,
        filters=filters,
    )
