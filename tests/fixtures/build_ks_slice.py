"""Regenerate the committed Kreuzer-Skarke slice used by the test suite.

    CYTOOLS_DB_DIR=/path/to/polytopes-4d python tests/fixtures/build_ks_slice.py

The slice holds **every** 4D reflexive polytope with `h11(lattice="N") <= 5`,
which is 6,472 of the 473,800,776 in the database. Completeness is the point:
the published counts this fixture pins are counts, so a sample would reproduce
none of them, while a complete stratum reproduces all of them exactly. The
figure is self-validating -- 6,472 is the paper's 6,366 favorable plus 106
non-favorable -- and `tests/test_repro_published.py` re-derives the split.

Those polytopes occupy vertex counts 5 through 9 only, so the slice is five
files and a few hundred kilobytes, small enough to commit and therefore small
enough for continuous integration to use as a real database.

The layout and schema deliberately match the real database file-for-file, so
`cytools.dataset` reads the slice through exactly the code path it uses in
production rather than through a test-only branch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

MAX_H11 = 5
VERTEX_COUNTS = range(5, 10)
DESTINATION = Path(__file__).parent / "ks-slice"


def main() -> int:
    source = os.environ.get("CYTOOLS_DB_DIR")
    if not source:
        print("set CYTOOLS_DB_DIR to the full database directory", file=sys.stderr)
        return 1

    DESTINATION.mkdir(parents=True, exist_ok=True)
    total = 0
    for n_vertices in VERTEX_COUNTS:
        name = f"polytopes-4d-{n_vertices:02d}-vertices.parquet"
        parquet = pq.ParquetFile(Path(source) / name)
        column = parquet.schema_arrow.names.index("h12")

        batches = []
        for group in range(parquet.metadata.num_row_groups):
            # `h12` is the N-lattice h11. Row-group statistics prune the vast
            # majority of the database before anything is decoded.
            statistics = parquet.metadata.row_group(group).column(column).statistics
            if statistics is not None and statistics.min > MAX_H11:
                continue
            table = parquet.read_row_group(group)
            h12 = table.column("h12").to_numpy()
            selected = table.filter((h12 >= 1) & (h12 <= MAX_H11))
            if selected.num_rows:
                batches.append(selected)

        if not batches:
            continue
        combined = pa.concat_tables(batches)
        pq.write_table(combined, DESTINATION / name, compression="zstd")
        total += combined.num_rows
        print(f"{name}: {combined.num_rows} rows")

    print(f"total: {total} rows in {DESTINATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
