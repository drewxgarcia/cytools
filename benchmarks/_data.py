"""Shared datasets for the benchmark suite.

Keeping these fixtures here prevents benchmark sample policy and environment
variables from leaking into the installed ``cytools.dataset`` module.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from cytools import Polytope
from cytools.dataset import PolytopeRecord, load_polytopes

POLY_5V = Polytope(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
)
POLY_6V = Polytope(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, -1, -3, -6],
        [-1, -1, -1, -1],
    ]
)


@dataclass(frozen=True)
class Tier:
    vertex_counts: tuple[int, ...] | None
    size_env: str
    default_size: int


TIERS = {
    "tiny": Tier((5,), "CYTOOLS_BENCH_N_TINY", 20),
    "small": Tier((6, 7), "CYTOOLS_BENCH_N_SMALL", 20),
    "medium": Tier((9, 10), "CYTOOLS_BENCH_N_MEDIUM", 20),
    "bulk": Tier((13, 14, 15, 16, 17), "CYTOOLS_BENCH_N_BULK", 20),
    "full": Tier(None, "CYTOOLS_BENCH_N_FULL", 100),
}

_VERTEX_FILE = re.compile(r"polytopes-4d-(\d+)-vertices\.parquet$")


def _local_vertex_counts(db_dir: str | Path | None) -> list[int]:
    root_value = db_dir or os.environ.get("CYTOOLS_DB_DIR")
    if root_value is None:
        raise ValueError("Set CYTOOLS_DB_DIR to run database-backed benchmarks.")
    counts = []
    for path in Path(root_value).glob("polytopes-4d-*-vertices.parquet"):
        match = _VERTEX_FILE.match(path.name)
        if match:
            counts.append(int(match.group(1)))
    return sorted(counts)


def load_tier(
    name: str,
    *,
    db_dir: str | Path | None = None,
    stream: bool = False,
    hf_token: str | None = None,
) -> list[PolytopeRecord]:
    """Load the benchmark sample described by ``name``."""
    try:
        tier = TIERS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(TIERS))
        raise ValueError(f"unknown benchmark tier {name!r}; choose {choices}") from exc

    size = int(os.environ.get(tier.size_env, tier.default_size))
    if tier.vertex_counts is not None:
        return load_polytopes(
            n_vertices=tier.vertex_counts,
            n=size,
            db_dir=db_dir,
            stream=stream,
            hf_token=hf_token,
        )

    counts = list(range(5, 37)) if stream else _local_vertex_counts(db_dir)
    records: list[PolytopeRecord] = []
    for vertex_count in counts:
        records.extend(
            load_polytopes(
                n_vertices=vertex_count,
                n=size,
                db_dir=db_dir,
                stream=stream,
                hf_token=hf_token,
            )
        )
    return records


def load_h11_sample(
    values: range,
    n: int,
    *,
    seed: int = 42,
) -> list[PolytopeRecord]:
    """Sample evenly across N-lattice/CY ``h11`` values."""
    h11_values = list(values)
    if not h11_values or n <= 0:
        return []
    per_value = max(1, math.ceil(n / len(h11_values)))
    records: list[PolytopeRecord] = []
    for h11 in h11_values:
        # The Parquet database stores the mirror/M-lattice convention, so its
        # h12 column is CYTools' default N-lattice h11.
        records.extend(load_polytopes(h12=h11, n=per_value, seed=seed))
    return records[:n]


__all__ = ["POLY_5V", "POLY_6V", "load_h11_sample", "load_tier"]
