#!/usr/bin/env python
"""
North-star throughput harness: the scoreboard the architecture work is judged on.

This is deliberately *not* a pytest-benchmark module. Those measure per-call
latency on warm, pre-built objects, which is the wrong instrument for the
question that matters here:

    How many Kreuzer-Skarke geometries per second per core can we take from
    local Parquet all the way to a result Parquet, and how much memory does
    that cost?

What it reports, per run:

    geometries/sec              end-to-end throughput
    geometries/sec/core         the number to compare across worker counts
    CPU-seconds/geometry        work done, independent of parallelism
    peak RSS                    high-water mark, parent + children
    Polytope objects/geometry   how often the scan collapses into the Python
                                object model -- the architectural metric. A
                                batch-native scan should drive this toward 0
                                for geometries that are never materialized.
    scaling efficiency          throughput/core relative to the 1-worker run

Usage
-----
    CYTOOLS_DB_DIR=~/Downloads/polytopes-4d \\
        python benchmarks/pipeline_throughput.py --n 200 --workers 1,2,4,8

    # write results and compare later
    ... --json benchmarks/artifacts/throughput.json
    ... --compare benchmarks/artifacts/throughput.json

Notes
-----
Worker counts above the physical core count are reported but flagged, since
throughput/core stops being meaningful once workers contend.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------

_POLYTOPE_COUNT_ENV = "CYTOOLS_COUNT_POLYTOPES"

# set once, when Polytope.__init__ is first wrapped
_COUNTER_GET = None


def install_polytope_counter():
    """Count Polytope.__init__ calls in this process.

    Returns a zero-argument callable giving the current count. Wrapping
    __init__ rather than sampling means the count is exact, including
    Polytopes built internally (duals, faces, subpolytopes).
    """
    global _COUNTER_GET

    from cytools.polytope import Polytope

    if _COUNTER_GET is not None:
        return _COUNTER_GET

    original = Polytope.__init__
    state = {"n": 0}

    def counting_init(self, *args, **kwargs):
        state["n"] += 1
        return original(self, *args, **kwargs)

    Polytope.__init__ = counting_init  # ty: ignore[invalid-assignment]
    _COUNTER_GET = lambda: state["n"]
    return _COUNTER_GET


def peak_rss_bytes() -> int:
    """High-water RSS for this process and all reaped children."""
    scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss units differ
    me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    return max(me, kids)


def cpu_seconds() -> float:
    r_self = resource.getrusage(resource.RUSAGE_SELF)
    r_kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (
        r_self.ru_utime + r_self.ru_stime + r_kids.ru_utime + r_kids.ru_stime
    )


# ---------------------------------------------------------------------------
# The payload
# ---------------------------------------------------------------------------


def compute_one(vertices):
    """Full pipeline for one geometry, from raw vertices to scalar invariants.

    Takes vertices rather than a Polytope so that the worker boundary carries
    plain arrays. Any Polytope construction here is counted by the harness and
    is exactly what the batch-native refactor is meant to remove from the scan
    path.
    """
    from cytools import Polytope

    p = Polytope(vertices)
    t = p.triangulate()
    tv = t.get_toric_variety()

    intnums = tv.intersection_numbers(in_basis=True)
    mori_rays = len(tv.mori_cone(in_basis=True).rays())

    return {
        "n_intnums": len(intnums),
        "n_mori_rays": mori_rays,
        "n_simplices": len(t.simplices()),
        "n_points": len(p.points()),
    }


_worker_counter = None


def worker_init():
    """Runs once per worker process. Installs the Polytope counter there.

    Without this, `Polytope objects/geometry` only ever reflects the parent and
    reads 0.00 for any parallel run -- which is exactly the metric the
    batch-native work needs to move, so it has to survive the process boundary.
    """
    global _worker_counter
    _worker_counter = install_polytope_counter()


def compute_one_safe(vertices):
    global _worker_counter
    if _worker_counter is None:
        _worker_counter = install_polytope_counter()

    before = _worker_counter()
    try:
        out = compute_one(vertices)
    except Exception as e:  # noqa: BLE001 - a failed geometry must not kill the scan
        out = {"error": f"{type(e).__name__}: {e}"}
    out["_n_polytopes"] = _worker_counter() - before
    return out


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@dataclass
class Result:
    n_geometries: int
    n_ok: int
    workers: int
    wall_s: float
    startup_s: float
    cpu_s: float
    geoms_per_s: float
    geoms_per_s_per_core: float
    cpu_s_per_geom: float
    peak_rss_mb: float
    polytopes_per_geom: float
    oversubscribed: bool


def run_once(vertex_arrays, workers: int, chunksize: int = 8) -> Result:
    """Measure steady-state throughput, with pool startup excluded.

    Startup is timed separately rather than folded in. A cold `import cytools`
    costs O(1.5 s) per worker process, so at small N that fixed cost swamps the
    measurement and makes healthy parallelism look broken -- an earlier version
    of this harness reported 15% scaling efficiency for exactly that reason.
    Both numbers are reported so neither can hide.
    """
    n = len(vertex_arrays)
    n_cores = os.cpu_count() or 1

    if workers == 1:
        worker_init()
        startup = 0.0
        cpu0 = cpu_seconds()
        t0 = time.perf_counter()
        results = [compute_one_safe(v) for v in vertex_arrays]
        wall = time.perf_counter() - t0
        cpu = cpu_seconds() - cpu0
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("spawn")  # fork segfaults with the compiled deps
        ex = ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, initializer=worker_init
        )
        try:
            # force every worker up, and pay the import cost, before timing
            t_start = time.perf_counter()
            list(ex.map(_ping, range(workers * 4), chunksize=1))
            startup = time.perf_counter() - t_start

            cpu0 = cpu_seconds()
            t0 = time.perf_counter()
            results = list(
                ex.map(compute_one_safe, vertex_arrays, chunksize=chunksize)
            )
            wall = time.perf_counter() - t0
        finally:
            # RUSAGE_CHILDREN only accounts for *reaped* children, so worker CPU
            # time is invisible until the pool is torn down. Sample after.
            ex.shutdown(wait=True)
        cpu = cpu_seconds() - cpu0

    n_ok = sum(1 for r in results if r and "error" not in r)
    polys = sum(r.get("_n_polytopes", 0) for r in results if r)

    return Result(
        n_geometries=n,
        n_ok=n_ok,
        workers=workers,
        wall_s=wall,
        startup_s=startup,
        cpu_s=cpu,
        geoms_per_s=n / wall if wall else float("nan"),
        geoms_per_s_per_core=(n / wall / workers) if wall else float("nan"),
        cpu_s_per_geom=cpu / n if n else float("nan"),
        peak_rss_mb=peak_rss_bytes() / 1e6,
        polytopes_per_geom=polys / n if n else float("nan"),
        oversubscribed=workers > n_cores,
    )


def _ping(_):
    return 1


def load_vertices(n: int, vertex_counts, db_dir):
    """Scan the local Parquet and hand back raw vertex arrays.

    Deliberately returns arrays, not Polytopes: the point of the harness is to
    measure what the *compute* side costs, and to expose how much Python object
    construction the current scan path forces.
    """
    from cytools.dataset import load_polytopes

    t0 = time.perf_counter()
    records = load_polytopes(n_vertices=vertex_counts, n=n, db_dir=db_dir or None)
    scan_s = time.perf_counter() - t0

    verts = [r.polytope.vertices() for r in records]
    return verts, scan_s, len(records)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(results, scan_s, baseline=None):
    n_cores = os.cpu_count() or 1
    print(f"\nmachine: {n_cores} cores, {sys.platform}")
    print(f"scan (Parquet -> vertex arrays): {scan_s * 1000:.0f} ms")

    head = (
        f"\n{'workers':>7}  {'geoms/s':>9}  {'/s/core':>8}  {'cpu-s/geom':>10}  "
        f"{'peak RSS':>9}  {'Polytope/geom':>13}  {'scaling':>7}  {'startup':>8}"
    )
    print(head)
    print("-" * (len(head) - 1))

    one = next((r for r in results if r.workers == 1), None)
    for r in results:
        scaling = (
            r.geoms_per_s_per_core / one.geoms_per_s_per_core
            if one and one.geoms_per_s_per_core
            else float("nan")
        )
        flag = " *" if r.oversubscribed else ""
        print(
            f"{r.workers:>7}  {r.geoms_per_s:>9.2f}  {r.geoms_per_s_per_core:>8.2f}  "
            f"{r.cpu_s_per_geom:>10.3f}  {r.peak_rss_mb:>7.0f}MB  "
            f"{r.polytopes_per_geom:>13.2f}  {scaling:>6.0%}{flag}  "
            f"{r.startup_s:>7.2f}s"
        )

    # a run must be long enough that per-pool startup is not the measurement
    for r in results:
        if r.workers > 1 and r.wall_s < 4 * r.startup_s:
            print(
                f"\n  WARNING: workers={r.workers} ran for {r.wall_s:.1f}s against "
                f"{r.startup_s:.1f}s of pool startup. Raise --n; at this size the "
                "throughput figure is mostly process spin-up, not geometry."
            )

    if any(r.oversubscribed for r in results):
        print("\n  * more workers than cores; throughput/core is not meaningful there")

    failed = [r for r in results if r.n_ok != r.n_geometries]
    for r in failed:
        print(
            f"\n  note: workers={r.workers} completed {r.n_ok}/{r.n_geometries} "
            "geometries; the rest raised (see --show-errors)"
        )

    if baseline:
        print("\nvs baseline:")
        base_by_w = {b["workers"]: b for b in baseline["results"]}
        for r in results:
            b = base_by_w.get(r.workers)
            if not b:
                continue
            delta = r.geoms_per_s / b["geoms_per_s"] if b["geoms_per_s"] else float("nan")
            arrow = "faster" if delta >= 1 else "SLOWER"
            print(
                f"  workers={r.workers:>2}: {b['geoms_per_s']:.2f} -> "
                f"{r.geoms_per_s:.2f} geoms/s  ({delta:.2f}x {arrow})"
            )


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[1])
    ap.add_argument("--n", type=int, default=200, help="geometries to process")
    ap.add_argument(
        "--vertex-counts",
        default="13,14,15,16,17",
        help="KS vertex-count files to sample (13-17 is 65.9%% of the database)",
    )
    ap.add_argument("--workers", default="1,2,4,8", help="comma-separated worker counts")
    ap.add_argument("--db-dir", default=None)
    ap.add_argument("--json", default=None, help="write results here")
    ap.add_argument("--compare", default=None, help="compare against a results file")
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--show-errors", action="store_true")
    args = ap.parse_args(argv)

    vertex_counts = [int(x) for x in args.vertex_counts.split(",")]
    worker_counts = [int(x) for x in args.workers.split(",")]

    verts, scan_s, n_loaded = load_vertices(args.n, vertex_counts, args.db_dir)
    if not verts:
        print("no geometries loaded; is CYTOOLS_DB_DIR set?", file=sys.stderr)
        return 1
    print(f"loaded {n_loaded} geometries from vertex counts {vertex_counts}")

    results = [run_once(verts, w, args.chunksize) for w in worker_counts]

    baseline = None
    if args.compare:
        baseline = json.loads(Path(args.compare).read_text())

    report(results, scan_s, baseline)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "n_cores": os.cpu_count(),
                    "platform": sys.platform,
                    "vertex_counts": vertex_counts,
                    "scan_s": scan_s,
                    "results": [asdict(r) for r in results],
                },
                indent=2,
            )
        )
        print(f"\nwrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
