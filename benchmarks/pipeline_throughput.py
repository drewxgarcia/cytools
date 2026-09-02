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
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
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

    # A deliberate, process-lifetime monkeypatch: the counter must observe every
    # Polytope built by the run, so it is installed rather than scoped. Assigning
    # a differently-typed callable to `__init__` is exactly what a type checker
    # should object to, so the objection is acknowledged here rather than dodged
    # via `setattr`, which would hide the same thing without saying so.
    Polytope.__init__ = counting_init

    def _counter_get():
        return state["n"]

    return _counter_get


def peak_rss_bytes() -> int:
    """High-water RSS for this process and all reaped children.

    A *lifetime* maximum that only ever ratchets up. Reported for continuity,
    but it is the wrong number to judge memory by, for three reasons that all
    bit in practice:

    - It includes the scan. `load_vertices` runs before any timing, and the
      Arrow decode buffers it allocates reached 274 MB at `batch_size=32768`
      while `pyarrow.total_allocated_bytes()` was 0 afterwards -- entirely
      transient, and nothing to do with the compute pipeline.
    - It never resets between worker counts, so every row after the first
      reports the maximum of all rows before it. The committed 2026-08-27
      baseline shows `peak_rss_mb` as exactly 675.67616 for workers=1, 2, 4
      *and* 8: one number, repeated, conveying nothing per row.
    - It ratchets with allocator high-water as N grows, in a way that tracks
      churn rather than footprint. Measured across two library generations
      whose steady-state RSS was 614 MB and 613 MB, this metric read 666 MB
      and 708 MB and looked like a 6% regression.

    `steady_rss_mb` is the number that matters, because `landscape` divides
    available memory by a per-worker footprint to cap worker count.
    """
    scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss units differ
    me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    return max(me, kids)


def tree_rss_bytes() -> int:
    """*Current* RSS of this process plus its direct children.

    Current, not high-water. Children are included because with a pool the
    workers hold the footprint, and `ps` is used rather than psutil to avoid
    adding a dependency to the harness.

    Honest about what this does and does not isolate. Worker processes are
    destroyed at pool teardown, so their footprint genuinely leaves the tree
    and does not contaminate a later row -- which is the contamination
    `ru_maxrss` cannot avoid. The *parent's* own arenas are stickier: CPython
    does not promptly return freed memory to the OS, so freeing 100 MB inside
    this process leaves RSS unchanged (measured). Read this as "what the
    machine is holding for this run", not as an allocation total.
    """
    me = os.getpid()
    out = subprocess.run(
        ["ps", "-Ao", "rss=,pid=,ppid="], capture_output=True, text=True, timeout=10
    )
    total = 0
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            rss, pid, ppid = (int(x) for x in parts)
        except ValueError:
            continue
        if pid == me or ppid == me:
            total += rss * 1024  # ps reports KiB on both platforms
    return total


def arrow_pool_peak_bytes() -> float:
    """Arrow's own allocation high-water, or nan when pyarrow is absent.

    Recorded separately so the scan's decode buffers are attributable instead
    of silently inflating a number labelled as the pipeline's memory.
    """
    try:
        import pyarrow as pa

        return float(pa.default_memory_pool().max_memory())
    except Exception:
        return float("nan")


class RssSampler(threading.Thread):
    """Samples process-tree RSS on an interval, for a steady-state figure.

    A median over the timed region, not a maximum: one transient spike is not
    the footprint that decides how many workers fit on a machine.
    """

    def __init__(self, interval: float = 0.1) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.samples.append(tree_rss_bytes())
            except Exception:  # sampling must never break a measurement
                pass

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=5)

    def median_bytes(self) -> float:
        return statistics.median(self.samples) if self.samples else float("nan")

    def max_bytes(self) -> float:
        return float(max(self.samples)) if self.samples else float("nan")


def performance_cores() -> int:
    """Cores worth scheduling a worker on, not every core the OS reports.

    `os.cpu_count()` counts efficiency cores too. On this M3 Pro that is 11
    against 5 performance cores, and the difference is not cosmetic: it decides
    whether a row is oversubscribed. The 1500-geometry sweep ran 8 workers on 5
    performance cores and reported `oversubscribed=False`, so a 67% scaling
    figure looked like a parallelism defect rather than the expected result of
    asking for more workers than there are fast cores.

    `CYTOOLS_MAX_WORKERS` overrides, matching the environment variable the
    library honours, so a constrained CI runner can state its own budget.
    """
    override = os.environ.get("CYTOOLS_MAX_WORKERS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            perf = int(out.stdout.strip())
            if perf > 0:
                return perf
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return os.cpu_count() or 1


def cpu_seconds() -> float:
    r_self = resource.getrusage(resource.RUSAGE_SELF)
    r_kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r_self.ru_utime + r_self.ru_stime + r_kids.ru_utime + r_kids.ru_stime


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
    except Exception as e:  # a failed geometry must not kill the scan
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
    #: Median process-tree RSS during the timed region. The headline memory
    #: number: it is per-row comparable and is what caps worker count.
    steady_rss_mb: float
    #: Maximum of the same samples, to show how spiky the run was.
    sampled_max_rss_mb: float
    #: Steady footprint attributable to one worker. This is the quantity
    #: `landscape._resolve_workers` divides available memory by when it caps
    #: the pool, so it is the memory figure with a decision attached to it.
    rss_per_worker_mb: float
    #: Lifetime `ru_maxrss`. Ratchets across rows; see `peak_rss_bytes`.
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
    n_cores = performance_cores()

    sampler = RssSampler()
    # Tree RSS before the run, so the per-worker figure is an increment rather
    # than including the parent's already-resident scan buffers.
    rss_before = tree_rss_bytes()

    if workers == 1:
        worker_init()
        startup = 0.0
        cpu0 = cpu_seconds()
        sampler.start()
        t0 = time.perf_counter()
        results = [compute_one_safe(v) for v in vertex_arrays]
        wall = time.perf_counter() - t0
        sampler.stop()
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
            sampler.start()
            t0 = time.perf_counter()
            results = list(ex.map(compute_one_safe, vertex_arrays, chunksize=chunksize))
            wall = time.perf_counter() - t0
            # Stopped before shutdown, so the samples cover a period when the
            # workers still existed. After teardown their RSS is gone and the
            # figure would describe the parent alone.
            sampler.stop()
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
        steady_rss_mb=sampler.median_bytes() / 1e6,
        sampled_max_rss_mb=sampler.max_bytes() / 1e6,
        rss_per_worker_mb=max(sampler.median_bytes() - rss_before, 0.0) / workers / 1e6,
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


def report(results, scan_s, baseline=None, scan_memory=None):
    n_perf = performance_cores()
    n_all = os.cpu_count() or 1
    detail = f"{n_perf} performance cores of {n_all} logical"
    print(f"\nmachine: {detail}, {sys.platform}")
    print(f"scan (Parquet -> vertex arrays): {scan_s * 1000:.0f} ms")

    # Attributed separately, because it lands in `peak RSS` below and is not
    # the pipeline's memory: Arrow's decode buffers are transient, and their
    # size is set by `batch_size` rather than by any geometry.
    if scan_memory:
        print(
            f"scan memory: {scan_memory['rss_mb']:.0f} MB RSS after scan"
            f"  (arrow pool high-water {scan_memory['arrow_mb']:.0f} MB,"
            f" live {scan_memory['arrow_live_mb']:.1f} MB)"
        )

    head = (
        f"\n{'workers':>7}  {'geoms/s':>9}  {'/s/core':>8}  {'cpu-s/geom':>10}  "
        f"{'steady RSS':>11}  {'MB/worker':>10}  {'lifetime':>9}  "
        f"{'Polytope/geom':>13}  {'scaling':>7}  {'startup':>8}"
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
            f"{r.cpu_s_per_geom:>10.3f}  {r.steady_rss_mb:>9.0f}MB  "
            f"{r.rss_per_worker_mb:>10.0f}  {r.peak_rss_mb:>7.0f}MB  "
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
        print(
            f"\n  * more workers than the {n_perf} performance cores; throughput/core "
            "is expected to fall there and is not a parallelism defect"
        )

    if len({round(r.peak_rss_mb) for r in results}) == 1 and len(results) > 1:
        print(
            "\n  note: 'lifetime' is ru_maxrss and only ratchets, so it is identical "
            "in every row here and includes the scan. Judge memory by 'steady RSS'."
        )

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
            delta = (
                r.geoms_per_s / b["geoms_per_s"] if b["geoms_per_s"] else float("nan")
            )
            arrow = "faster" if delta >= 1 else "SLOWER"
            print(
                f"  workers={r.workers:>2}: {b['geoms_per_s']:.2f} -> "
                f"{r.geoms_per_s:.2f} geoms/s  ({delta:.2f}x {arrow})"
            )
            # `.get`, because baselines written before steady-state sampling
            # existed carry only the ratcheting lifetime figure. Comparing
            # against that is what produced a phantom 39% memory regression,
            # so it is deliberately not compared at all.
            base_steady = b.get("steady_rss_mb")
            if base_steady:
                print(
                    f"                memory {base_steady:.0f} -> "
                    f"{r.steady_rss_mb:.0f} MB steady"
                )
            else:
                print(
                    "                memory: not comparable; the baseline predates "
                    "steady-state sampling and its peak_rss_mb includes the scan"
                )


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[1])
    ap.add_argument("--n", type=int, default=200, help="geometries to process")
    ap.add_argument(
        "--vertex-counts",
        default="13,14,15,16,17",
        help="KS vertex-count files to sample (13-17 is 65.9%% of the database)",
    )
    ap.add_argument(
        "--workers", default="1,2,4,8", help="comma-separated worker counts"
    )
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

    # Snapshot memory at the scan/compute boundary, so the scan's share of the
    # lifetime high-water is attributable rather than folded into every row.
    try:
        import pyarrow as pa

        arrow_live_mb = pa.total_allocated_bytes() / 1e6
    except Exception:
        arrow_live_mb = float("nan")
    scan_memory = {
        "rss_mb": tree_rss_bytes() / 1e6,
        "peak_rss_mb": peak_rss_bytes() / 1e6,
        "arrow_mb": arrow_pool_peak_bytes() / 1e6,
        "arrow_live_mb": arrow_live_mb,
    }

    results = [run_once(verts, w, args.chunksize) for w in worker_counts]

    baseline = None
    if args.compare:
        baseline = json.loads(Path(args.compare).read_text())

    report(results, scan_s, baseline, scan_memory)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "n_cores": performance_cores(),
                    "n_logical_cores": os.cpu_count(),
                    "platform": sys.platform,
                    "vertex_counts": vertex_counts,
                    "scan_s": scan_s,
                    "scan_memory": scan_memory,
                    "results": [asdict(r) for r in results],
                },
                indent=2,
            )
        )
        print(f"\nwrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
