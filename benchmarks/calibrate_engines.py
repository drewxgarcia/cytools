"""Measure the engine crossovers that `cytools._backends.crossovers` asserts.

The thresholds that used to sit inline in the domain code -- `ambient_dim >= 25`
in five places in `cone.py`, `1 <= dim <= 4` in `polytope.py` -- were inherited
folklore with no derivation anywhere in the tree. This script derives them, and
`tests/test_crossovers.py` fails when a recorded value stops matching what the
installed engines actually do.

Every measurement runs in a **subprocess**. That is not defensiveness: PALP
calls `abort()` when a configuration exceeds one of its compile-time limits
(`CEQ_Nmax`, `EQUA_Nmax`) or contains duplicate points, which terminates the
interpreter with SIGABRT and no Python traceback. A calibration harness that
ran engines in-process would die partway through and silently report a
truncated sweep. Isolating each cell also lets the harness *record* which
(engine, problem) pairs abort, which is itself the measurement that determines
where PALP may be registered as usable.

Usage:

    .venv/bin/python benchmarks/calibrate_engines.py --task convex_hull
    ... --task interior_point
    ... --task all

Output is a JSON report under `benchmarks/artifacts/`, plus a human summary.
Engines whose dependency is absent are reported as skipped rather than
silently dropped, so a run on a machine without Mosek cannot be mistaken for
evidence about Mosek.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ARTIFACTS = Path(__file__).parent / "artifacts"

ABORTED = "aborted"
SKIPPED = "skipped"
FAILED = "failed"
TIMEOUT = "timeout"
NA = "n/a"


# problem generators
# ------------------
# Shared by the parent (to enumerate cells) and the worker (to rebuild the
# exact same problem from its descriptor). Seeded and duplicate-free, so a
# rerun measures the same thing and no cell trips PALP's duplicate-point abort
# for reasons unrelated to size.
def hull_problem(dim: int, extra: int) -> np.ndarray:
    rng = np.random.default_rng(1)
    seen = {
        tuple(int(x) for x in p)
        for p in np.vstack([np.eye(dim, dtype=int), -np.ones((1, dim), dtype=int)])
    }
    target = dim + 1 + extra
    # The sampling box must hold more lattice points than we ask for, or the
    # rejection loop below never terminates -- (2b+1)**dim distinct points are
    # available, and at dim=1 the default box holds only 9.
    bound = 4
    while (2 * bound + 1) ** dim < 4 * target:
        bound += 1
    while len(seen) < target:
        seen.add(tuple(int(x) for x in rng.integers(-bound, bound + 1, size=dim)))
    return np.array(sorted(seen))


def cone_problem(dim: int, n: int) -> np.ndarray:
    """Hyperplanes of a solid cone, built around a known interior point.

    Solidity is guaranteed by construction; calibrating on infeasible problems
    would time the failure path instead of the one that matters.
    """
    rng = np.random.default_rng(2)
    interior = np.ones(dim)
    rows = []
    while len(rows) < n:
        h = rng.integers(-5, 6, size=dim)
        if h.dot(interior) > 0:
            rows.append(h)
    return np.array(rows)


# worker
# ------
def _worker(task: str, engine_name: str, repeat: int, params: list[int]) -> None:
    """Time one (engine, problem) cell and print a JSON line. Runs isolated."""
    from cytools._backends.engines import CONVEX_HULL, INTERIOR_POINT, STRETCHED_TIP

    registries = {
        "convex_hull": CONVEX_HULL,
        "interior_point": INTERIOR_POINT,
        "stretched_tip": STRETCHED_TIP,
    }
    engine = registries[task][engine_name]
    if not engine.available():
        print(json.dumps({"status": SKIPPED}))
        return

    # Do not consult the registry's measured crossover here: this benchmark is
    # the evidence that defines that predicate, so doing so would make the
    # measurement circular. Only exclude engines with a hard mathematical or
    # library-domain restriction.
    dim = params[0]
    if task == "convex_hull" and (
        (engine_name == "interval" and dim != 1) or (engine_name == "qhull" and dim < 2)
    ):
        print(json.dumps({"status": NA}))
        return

    if task == "convex_hull":
        problem = (hull_problem(*params),)
    elif task == "interior_point":
        dim = params[0]
        problem = (cone_problem(dim, params[1]), 1.0, dim, None, False)
    else:
        problem = (cone_problem(params[0], params[1]), 1.0)

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        try:
            engine.run(*problem)
        except Exception as exc:  # a real exception is a usable outcome
            print(
                json.dumps({"status": FAILED, "error": f"{type(exc).__name__}: {exc}"})
            )
            return
        samples.append(time.perf_counter() - start)
    print(json.dumps({"status": "ok", "seconds": statistics.median(samples)}))


def _measure(task: str, engine: str, repeat: int, params: list[int]) -> dict:
    """Run one cell in a subprocess. A killed process is recorded as ABORTED."""
    try:
        proc = subprocess.run(
            [
                sys.executable,
                __file__,
                "--worker",
                "--task",
                task,
                "--engine",
                engine,
                "--repeat",
                str(repeat),
                "--params",
                *[str(p) for p in params],
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"status": TIMEOUT}
    for line in reversed(proc.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"status": ABORTED, "returncode": proc.returncode}


# report formatting
# -----------------
def _cell(result: dict, scale: float, unit: str) -> str:
    status = result["status"]
    if status == "ok":
        return f"{result['seconds'] * scale:8.1f}{unit}"
    return f"{status:>8s} "


def _sweep(task: str, engines, cells, repeat, scale, unit, label) -> dict:
    results = {}
    for params in cells:
        row = {e: _measure(task, e, repeat, list(params)) for e in engines}
        key = "x".join(str(p) for p in params)
        results[key] = row
        ok = [(r["seconds"], e) for e, r in row.items() if r["status"] == "ok"]
        winner = min(ok)[1] if ok else "none"
        print(
            f"  {label}={key:<8s} "
            + "  ".join(f"{e}={_cell(r, scale, unit)}" for e, r in row.items())
            + f"   -> {winner}"
        )
    return results


def calibrate_convex_hull(repeat: int) -> dict:
    """V-to-H: which exact engine wins, and where each stops being usable."""
    cells = [(d, e) for d in (1, 2, 3, 4, 5, 6, 8, 9, 10) for e in (0, 12, 30)]
    return _sweep(
        "convex_hull",
        ("interval", "ppl", "palp", "qhull"),
        cells,
        repeat,
        1e6,
        "us",
        "dim,extra",
    )


def calibrate_interior_point(repeat: int) -> dict:
    """LP feasibility across ambient dimension."""
    cells = [(d, 6 * d) for d in (4, 8, 12, 16, 20, 25, 30, 40, 60, 80)]
    return _sweep(
        "interior_point",
        ("highs", "glop", "scip", "cpsat"),
        cells,
        repeat,
        1e3,
        "ms",
        "dim,rows",
    )


def calibrate_stretched_tip(repeat: int) -> dict:
    """The QP across ambient dimension."""
    cells = [(d, 4 * d) for d in (4, 8, 12, 16, 20, 25, 30, 40)]
    return _sweep(
        "stretched_tip",
        ("mosek", "osqp", "cvxopt"),
        cells,
        repeat,
        1e3,
        "ms",
        "dim,rows",
    )


TASKS = {
    "convex_hull": calibrate_convex_hull,
    "interior_point": calibrate_interior_point,
    "stretched_tip": calibrate_stretched_tip,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="all", choices=[*TASKS, "all"])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--engine", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--params", nargs="*", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        _worker(args.task, args.engine, args.repeat, args.params or [])
        return

    chosen = list(TASKS) if args.task == "all" else [args.task]
    report = {
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repeat": args.repeat,
        "python": sys.version.split()[0],
        "tasks": {},
    }
    for name in chosen:
        print(f"\n{name}")
        report["tasks"][name] = TASKS[name](args.repeat)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = args.out or ARTIFACTS / "engine_crossovers.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
