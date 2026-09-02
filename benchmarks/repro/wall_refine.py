"""Refine the h11=3 upper bound on the "# CYs" count.

A box search over GL(3,Z) leaves 188 classes against the paper's 186, i.e. two
equivalences unfound. Rather than enlarge the box (cost grows as bound^9), solve
for Lambda with constraint propagation, the paper's approach in miniature:

  * c_2 is a linear form, so Lambda^T c2 = c2' is 3 linear equations on the 9
    unknowns -- massively constraining, and cheap to enumerate solutions of.
  * for each Lambda satisfying the c_2 constraint (and |det| = 1), check the
    cubic kappa exactly.

This searches a far larger effective region than an affordable box, but it is
still bounded: a match with the paper is reproduction evidence, not a proof
that no transformation exists outside the search.
"""

import itertools
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

os.environ.setdefault("CYTOOLS_DB_DIR", os.path.expanduser("~/Downloads/polytopes-4d"))
from benchmarks.repro.wall_classes import (  # after CYTOOLS_DB_DIR is set above
    gl_matrices,
    invariants,
    transform,
    wall_data,
)

BOUND_FAST = 2
BOUND_DEEP = 6


def deep_equivalent(a, b, bound=BOUND_DEEP):
    """Is there Lambda in GL(3,Z), entries in [-bound,bound], with a -> b?

    Enumerate Lambda column-block-wise under the c_2 constraint instead of
    enumerating all bound^9 matrices.
    """
    ka, ca = a[2], a[3]
    kb, cb = b[2], b[3]
    n = ka.shape[0]
    rng = range(-bound, bound + 1)

    # rows of Lambda: (Lambda^T ca)_i = sum_a L[a,i] ca[a] = cb[i]
    # so column i of Lambda must satisfy  ca . L[:,i] == cb[i]
    cols = []
    for i in range(n):
        ok = [
            v
            for v in itertools.product(rng, repeat=n)
            if int(np.dot(ca, v)) == int(cb[i])
        ]
        if not ok:
            return False
        cols.append(ok)

    for combo in itertools.product(*cols):
        L = np.array(combo, dtype=np.int64).T  # columns
        d = int(round(np.linalg.det(L.astype(np.float64))))
        if abs(d) != 1:
            continue
        k, c = transform(ka, ca, L)
        if np.array_equal(k, kb) and np.array_equal(c, cb):
            return True
    return False


def main(h11):
    from cytools.dataset import load_polytopes

    recs = load_polytopes(h12=h11)
    vl = [r.polytope.vertices().tolist() for r in recs]
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn")) as ex:
        geoms = [g for sub in ex.map(wall_data, vl, chunksize=8) for g in sub]

    mats = gl_matrices(h11, BOUND_FAST)
    buckets = {}
    for g in geoms:
        buckets.setdefault(invariants(*g), []).append(g)

    t0 = time.time()
    total = 0
    unresolved = []
    for key, group in buckets.items():
        reps = []
        for g in group:
            hit = False
            for rep in reps:
                for L in mats:
                    k, c = transform(g[2], g[3], L)
                    if np.array_equal(k, rep[2]) and np.array_equal(c, rep[3]):
                        hit = True
                        break
                if hit:
                    break
            if not hit:
                reps.append(g)
        total += len(reps)
        if len(reps) > 1:
            unresolved.append((key, reps))
    print(
        f"h11={h11}: box search (bound={BOUND_FAST}) -> {total} classes "
        f"({time.time() - t0:.0f}s); {len(unresolved)} buckets hold >1 rep"
    )

    # deep pass, only on pairs the box search left separate
    merged = 0
    t0 = time.time()
    for _key, reps in unresolved:
        keep = []
        for g in reps:
            if any(deep_equivalent(g, r) for r in keep):
                merged += 1
            else:
                keep.append(g)
    print(
        f"  deep c2-constrained pass (bound={BOUND_DEEP}) merged {merged} more "
        f"({time.time() - t0:.0f}s)"
    )
    final = total - merged
    exp = {1: 5, 2: 29, 3: 186, 4: 1186}.get(h11)
    print(
        f"  Wall equivalence classes = {final}   paper = {exp}   "
        f"{'MATCH' if final == exp else 'DIFF'}"
    )


if __name__ == "__main__":
    for k in (int(x) for x in (sys.argv[1:] or ["3"])):
        main(k)
