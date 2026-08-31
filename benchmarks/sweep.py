#!/usr/bin/env python
"""
A real landscape sweep, from the command line.

The sweep itself lives in the library now -- `cytools.sweep` -- because it is a
tool rather than a benchmark. What remains here is the shell around it: argument
parsing, and the operational reporting that a multi-hour run needs and a library
call should not impose (resident memory, ETA, store growth).

    CYTOOLS_DB_DIR=~/Downloads/polytopes-4d \\
    CYTOOLS_DERIVED_DIR=~/ks-derived \\
        python benchmarks/sweep.py --columns h11,is_favorable,n_intnums \\
                                  --n 100000 --workers 6

    # the same command again picks up whatever is left
    # --status prints what has been computed without computing anything
    # --list prints every column that can be requested

The numbers it reports -- whether throughput holds over hours, whether
`known_ids` stays cheap as the store grows, what fraction of real KS geometries
the pipeline fails on -- were guesses before this existed.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time


def rss_mb() -> float:
    scale = 1 if sys.platform == "darwin" else 1024
    me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    return max(me, kids) / 1e6


def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[1])
    ap.add_argument(
        "--columns",
        default="h11,h21,chi,is_favorable,n_points,n_simplices,n_intnums,n_mori_rays",
        help="comma-separated columns to compute (see --list)",
    )
    ap.add_argument("--n", type=int, default=100_000, help="geometries to target")
    ap.add_argument(
        "--vertex-counts",
        default="13,14,15,16,17",
        help="KS vertex-count files to sample (13-17 is 65.9%% of the database)",
    )
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--version", type=int, default=1, help="cache schema version")
    ap.add_argument("--derived-dir", default=None)
    ap.add_argument("--db-dir", default=None)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--status", action="store_true", help="report and exit")
    ap.add_argument("--list", action="store_true", help="list columns and exit")
    args = ap.parse_args(argv)

    from cytools import quantities, status, sweep

    if args.list:
        print(quantities().to_string(index=False))
        return 0

    if args.status:
        st = status(args.derived_dir)
        print("  store is empty" if st.empty else st.to_string(index=False))
        return 0

    columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    vertex_counts = [int(x) for x in args.vertex_counts.split(",")]

    print(
        f"sweep: {args.n:,} geometries from vertex counts {vertex_counts}\n"
        f"  columns={columns}\n"
        f"  workers={args.workers or 'auto'}\n"
    )

    t_start = time.perf_counter()
    last = {"t": t_start}

    def on_progress(summary):
        now = time.perf_counter()
        if now - last["t"] < 10 and summary["requested"] < args.n:
            return
        last["t"] = now
        elapsed = now - t_start
        done = summary["computed"] + summary["unsupported"] + summary["failed"]
        rate = done / elapsed if elapsed and done else 0.0
        eta = (args.n - summary["requested"]) / rate if rate else float("inf")
        print(
            f"  [{human_time(elapsed):>6}] "
            f"seen {summary['requested']:>8,}/{args.n:,}  "
            f"computed {summary['computed']:>8,}  "
            f"cached {summary['skipped']:>7,}  "
            f"unsupported {summary['unsupported']:>7,}  "
            f"failed {summary['failed']:>6,}  "
            f"{rate:6.1f} g/s  RSS {rss_mb():6.0f}MB  "
            f"ETA {human_time(eta)}"
        )

    summary = sweep(
        columns,
        n=args.n,
        n_vertices=vertex_counts,
        workers=args.workers,
        version=args.version,
        recompute=args.recompute,
        db_dir=args.db_dir,
        derived_dir=args.derived_dir,
        progress=on_progress,
    )

    elapsed = time.perf_counter() - t_start
    done = summary["computed"] + summary["unsupported"] + summary["failed"]
    workers = args.workers or min(8, max(1, (os.cpu_count() or 2) - 2))

    print(
        f"\ndone in {human_time(elapsed)}\n"
        f"  {summary}\n"
        f"  throughput  : {done / elapsed:.1f} geoms/s over {workers} workers\n"
        f"                {done / elapsed / workers:.1f} geoms/s/worker\n"
        f"  peak RSS    : {rss_mb():.0f} MB"
    )
    st = status(args.derived_dir)
    if not st.empty:
        print("\n" + st.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
