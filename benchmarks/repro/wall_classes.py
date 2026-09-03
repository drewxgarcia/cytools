"""Reproduce the "# CYs" column of arXiv:2310.06820 Table 1 -- the paper's
headline result -- by counting equivalence classes of *Wall data*.

Wall's theorem: a simply connected CY threefold with torsion-free homology is
classified up to diffeomorphism by (h11, h21, kappa_ijk, c_2). Two are
equivalent iff some Lambda in GL(h11,Z) maps one pair into the other:

    kappa'_ijk = Lambda^a_i Lambda^b_j Lambda^c_k kappa_abc
    c'_2,i     = Lambda^a_i c_2,a

Strategy (the paper's, in miniature): bucket by cheap basis-independent
invariants first, then search for an explicit Lambda inside each bucket. A
bounded search proves equivalence when it exhibits a transformation, but it
cannot prove inequivalence. The resulting class count is therefore an upper
bound; stability as the bound grows is strong evidence, not a proof.

Paper Table 1, "# CYs with pi_1 = 0" + "# ECs with pi_1 != 0":
    h11=1 -> 4 + 1     h11=2 -> 27 + 2
so the total number of Wall classes is 5 and 29 respectively. This script
counts *Wall classes* (it does not compute pi_1), so the target is the total.
"""

import argparse
import itertools
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

os.environ.setdefault("CYTOOLS_DB_DIR", os.path.expanduser("~/Downloads/polytopes-4d"))

PAPER_TOTAL_WALL_CLASSES = {
    1: 5,
    2: 29,
    3: 186,
    4: 1186,
}  # (pi_1=0)+(pi_1!=0), favorable


def wall_data(verts, favorable_only=True):
    """One CY per two-face class; return its (kappa, c2) in the CY basis."""
    from cytools import Polytope

    p = Polytope(verts)
    if favorable_only and not p.is_favorable(lattice="N"):
        return []
    out = []
    seen = set()
    for t in p.all_triangulations():
        key = frozenset(frozenset(s) for s in t.simplex_set(on_faces_dim=2))
        if key in seen:
            continue
        seen.add(key)
        cy = t.get_cy()
        n = cy.h11()
        kappa = np.zeros((n, n, n), dtype=np.int64)
        for idx, v in cy.intersection_numbers(in_basis=True).items():
            for perm in set(itertools.permutations(idx)):
                kappa[perm] = int(round(v))
        c2 = np.asarray(cy.second_chern_class(in_basis=True), dtype=np.int64)
        out.append((int(cy.h11()), int(cy.h12()), kappa, c2))
    return out


def transform(kappa, c2, L):
    """Push (kappa, c2) through Lambda. Indices are lowered, so contract with L."""
    k = np.einsum("ai,bj,ck,abc->ijk", L, L, L, kappa)
    return k, L.T @ c2


def invariants(h11, h21, kappa, c2):
    """Bucket key using ONLY basis-independent quantities.

    Careful: sorted kappa_iii, kappa.sum(), c2.sum() and friends are *not*
    invariant under GL(n,Z) -- a change of basis mixes the indices, so using
    them here would split genuinely equivalent geometries and inflate the
    count. What is invariant:
      * the Hodge numbers;
      * the integral content of the cubic form and of the linear form c_2
        (gcd of coefficients is preserved by a unimodular substitution);
    Equivalence inside a bucket is always decided by exhibiting an explicit
    Lambda.
    """

    def content(values):
        result = 0
        for value in np.asarray(values).flat:
            result = math.gcd(result, abs(int(value)))
        return result

    g_k = content(kappa)
    g_c = content(c2)
    return (h11, h21, g_k, g_c)


def gl_matrices(n, bound):
    """All Lambda in GL(n,Z) with entries in [-bound, bound], vectorized.

    A Python loop over (2*bound+1)**(n*n) candidates is hopeless at n=3,
    bound=2 (~2e6 iterations each building a numpy array); build the whole
    grid at once and filter on |det| == 1 in one shot.
    """
    vals = np.arange(-bound, bound + 1, dtype=np.int64)
    grid = np.stack(np.meshgrid(*([vals] * (n * n)), indexing="ij"), axis=-1)
    cand = grid.reshape(-1, n, n)
    dets = np.rint(np.linalg.det(cand.astype(np.float64))).astype(np.int64)
    return list(cand[np.abs(dets) == 1])


def equivalent(a, b, mats):
    ka, ca = a[2], a[3]
    kb, cb = b[2], b[3]
    for L in mats:
        k, c = transform(ka, ca, L)
        if np.array_equal(k, kb) and np.array_equal(c, cb):
            return True
    return False


def run(h11, bound):
    from cytools.dataset import load_polytopes

    t0 = time.time()
    recs = load_polytopes(h12=h11)
    vl = [r.polytope.vertices().tolist() for r in recs]
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn")) as ex:
        geoms = [g for sub in ex.map(wall_data, vl, chunksize=8) for g in sub]
    t_build = time.time() - t0

    mats = gl_matrices(h11, bound)

    # bucket by invariants, then resolve inside buckets with explicit Lambda
    buckets = {}
    for g in geoms:
        buckets.setdefault(invariants(*g), []).append(g)

    t0 = time.time()
    n_classes = 0
    for group in buckets.values():
        reps = []
        for g in group:
            if not any(equivalent(g, rep, mats) for rep in reps):
                reps.append(g)
        n_classes += len(reps)
    t_cls = time.time() - t0

    exp = PAPER_TOTAL_WALL_CLASSES.get(h11)
    print(
        "  NOTE: a bounded GL search proves equivalence, never inequivalence,"
        " so this is an UPPER bound; stability as --bound grows is evidence,"
        " not proof."
    )
    print(
        f"h11={h11}: FRST classes(geometries)={len(geoms)}  "
        f"invariant buckets={len(buckets)}  |GL({h11},Z)| searched={len(mats)}"
    )
    print(
        f"  Wall equivalence classes = {n_classes}   paper total = {exp}   "
        f"{'MATCH' if n_classes == exp else 'DIFF'}"
    )
    print(f"  (build {t_build:.1f}s, classify {t_cls:.1f}s)")
    sys.stdout.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h11", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--bound", type=int, default=3)
    a = ap.parse_args()
    for k in a.h11:
        run(k, a.bound)
