#!/usr/bin/env python
"""
A real landscape sweep: scan the local KS database, compute invariants, persist.

Not a benchmark. This is the thing the batch scan and the derived store exist
to make possible, run at a scale where the numbers collected from N=2000 stop
being predictive. It is safe to interrupt: results are keyed by ks_id and
committed per batch, so re-running resumes rather than restarting.

    CYTOOLS_DB_DIR=~/Downloads/polytopes-4d \\
    CYTOOLS_DERIVED_DIR=~/ks-derived \\
        python benchmarks/sweep.py --n 100000 --workers 6

    # same command again picks up whatever is left
    # --status prints what has been computed without computing anything

What it reports while running: geometries/sec, ETA, failure rate, resident
memory, and the size of the store. Those are the numbers that were guesses
before: whether throughput holds over hours, whether known_ids stays cheap as
the store grows, and what fraction of real KS geometries the pipeline actually
fails on.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

# The payload runs in worker processes, so it must be importable by name and
# this module must keep every side effect under __main__. See the warning on
# cytools.store.materialize.


def invariants(vertices):
    """Vertices -> scalar invariants. One geometry's worth of work."""
    from cytools import Polytope

    p = Polytope(vertices)
    triang = p.triangulate()
    tv = triang.get_toric_variety()

    intnums = tv.intersection_numbers(in_basis=True)
    mori = tv.mori_cone(in_basis=True)

    return {
        "n_points": len(p.points()),
        "n_simplices": len(triang.simplices()),
        "n_intnums": len(intnums),
        "n_mori_rays": len(mori.rays()),
        "h11_N": int(p.h11(lattice="N")),
        "h21_N": int(p.h21(lattice="N")),
        "chi_N": int(p.chi(lattice="N")),
        "is_favorable": bool(p.is_favorable(lattice="N")),
    }


def rss_mb() -> float:
    scale = 1 if sys.platform == "darwin" else 1024
    me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    return max(me, kids) / 1e6


def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=100_000, help="geometries to target")
    ap.add_argument(
        "--vertex-counts",
        default="13,14,15,16,17",
        help="KS vertex-count files to sample (13-17 is 65.9%% of the database)",
    )
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--quantity", default="invariants")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--derived-dir", default=None)
    ap.add_argument("--db-dir", default=None)
    ap.add_argument(
        "--compact-every",
        type=int,
        default=50,
        help="compact the store every N batches (0 disables)",
    )
    ap.add_argument("--status", action="store_true", help="report and exit")
    args = ap.parse_args(argv)

    from cytools.dataset import scan_batches
    from cytools.store import DerivedStore, materialize

    store = DerivedStore(args.derived_dir)

    if args.status:
        for q in store.quantities() or []:
            for v in store.versions(q):
                st = store.stats(q, v)
                print(
                    f"  {q} v{v}: {st['n_rows']:,} rows, {st['n_parts']} parts, "
                    f"{st['bytes']/1e6:.1f} MB"
                )
        if not store.quantities():
            print("  store is empty")
        return 0

    vertex_counts = [int(x) for x in args.vertex_counts.split(",")]
    already = len(store.known_ids(args.quantity, args.version))

    print(
        f"sweep: {args.n:,} geometries from vertex counts {vertex_counts}\n"
        f"  workers={args.workers} batch_size={args.batch_size} "
        f"chunksize={args.chunksize}\n"
        f"  store={store.root}  quantity={args.quantity} v{args.version}\n"
        f"  already computed: {already:,}\n"
    )

    scan_kwargs = {"n_vertices": vertex_counts, "n": args.n, "batch_size": args.batch_size}
    if args.db_dir:
        scan_kwargs["db_dir"] = args.db_dir

    t_start = time.perf_counter()
    state = {"batches": 0, "last_report": t_start, "last_computed": 0}

    def on_progress(summary):
        state["batches"] += 1
        now = time.perf_counter()
        elapsed = now - t_start
        done = summary["computed"] + summary["failed"]

        # compaction keeps known_ids cheap as part files accumulate
        if args.compact_every and state["batches"] % args.compact_every == 0:
            store.compact(args.quantity, args.version)

        if now - state["last_report"] < 10 and summary["requested"] < args.n:
            return
        state["last_report"] = now

        rate = done / elapsed if elapsed and done else 0.0
        remaining = args.n - summary["requested"]
        eta = remaining / rate if rate else float("inf")
        fail_pct = 100.0 * summary["failed"] / max(done, 1)

        print(
            f"  [{human_time(elapsed):>6}] "
            f"seen {summary['requested']:>8,}/{args.n:,}  "
            f"computed {summary['computed']:>8,}  "
            f"skipped {summary['skipped']:>7,}  "
            f"failed {summary['failed']:>6,} ({fail_pct:4.1f}%)  "
            f"{rate:6.1f} g/s  RSS {rss_mb():6.0f}MB  "
            f"ETA {human_time(eta)}"
        )

    summary = materialize(
        args.quantity,
        invariants,
        store=store,
        scan=scan_batches(**scan_kwargs),
        version=args.version,
        workers=args.workers,
        chunksize=args.chunksize,
        on_progress=on_progress,
    )

    elapsed = time.perf_counter() - t_start
    done = summary["computed"] + summary["failed"]

    if args.compact_every:
        store.compact(args.quantity, args.version)
    st = store.stats(args.quantity, args.version)

    print(
        f"\ndone in {human_time(elapsed)}\n"
        f"  {summary}\n"
        f"  throughput  : {done/elapsed:.1f} geoms/s over {args.workers} workers\n"
        f"                {done/elapsed/args.workers:.1f} geoms/s/worker\n"
        f"  peak RSS    : {rss_mb():.0f} MB\n"
        f"  store       : {st['n_rows']:,} rows, {st['n_parts']} parts, "
        f"{st['bytes']/1e6:.1f} MB "
        f"({st['bytes']/max(st['n_rows'],1):.0f} bytes/geometry)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
