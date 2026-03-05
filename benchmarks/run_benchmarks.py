#!/usr/bin/env python
"""
Convenience runner for the CYTools benchmark suite.

Usage:
    # Quick run — tiny/small tiers only, no slow tests, JSON output
    python benchmarks/run_benchmarks.py

    # Full run including medium/large tiers
    python benchmarks/run_benchmarks.py --all

    # Run a specific module
    python benchmarks/run_benchmarks.py --module polytope

    # Compare against a saved baseline
    python benchmarks/run_benchmarks.py --compare baselines/baseline.json

    # Save results as a new baseline
    python benchmarks/run_benchmarks.py --save baselines/my_baseline.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent


def build_pytest_args(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, "-m", "pytest",
        str(HERE),
        "--benchmark-only",
        "--benchmark-columns=min,mean,stddev,rounds,iterations",
        "--benchmark-sort=name",
        "-v",
    ]

    if not args.all:
        cmd += ["-m", "not slow"]

    if args.module:
        # Run only bench_<module>.py
        target = HERE / f"bench_{args.module}.py"
        if not target.exists():
            print(f"ERROR: {target} does not exist", file=sys.stderr)
            sys.exit(1)
        # Replace the directory target with the specific file
        cmd[3] = str(target)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cmd += [f"--benchmark-save={save_path.stem}", f"--benchmark-storage={save_path.parent}"]

    if args.compare:
        compare_path = Path(args.compare)
        if not compare_path.exists():
            print(f"ERROR: baseline file not found: {compare_path}", file=sys.stderr)
            sys.exit(1)
        cmd += [f"--benchmark-compare={compare_path}"]

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        cmd += [f"--benchmark-json={json_path}"]

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run CYTools benchmarks")
    parser.add_argument("--all", action="store_true", help="Include slow (medium/large) tiers")
    parser.add_argument("--module", metavar="NAME", help="Run only bench_<NAME>.py")
    parser.add_argument("--save", metavar="PATH", help="Save results to PATH (JSON)")
    parser.add_argument("--compare", metavar="PATH", help="Compare against baseline at PATH")
    parser.add_argument("--json", metavar="PATH", help="Write JSON report to PATH")
    args = parser.parse_args()

    cmd = build_pytest_args(args)
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
