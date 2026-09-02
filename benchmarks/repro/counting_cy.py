"""Reproduce Table 1 of arXiv:2310.06820 (Gendler, MacFadden, McAllister, Moritz,
Nally, Schachner, Stillman -- "Counting Calabi-Yau Threefolds") with this fork.

Columns reproduced: # polytopes, # FRSTs, # FRST classes -- each split into
favorable + non-favorable, at fixed h^{1,1} of the CY threefold.

Definitions used (matched to the paper):
  * polytopes at fixed h11: 4d reflexive polytopes with poly.h11(lattice="N")==k.
    In the local Parquet mirror of Kreuzer-Skarke this is the ``h12`` column
    (the shipped Hodge columns are M-lattice; see PolytopeRecord).
  * FRST: fine regular star triangulation, points interior to facets excluded
    (CYTools default for reflexive polytopes).
  * FRST class: orbit of the *induced two-face triangulation* under the
    automorphism group of the polytope.  The two-face restriction alone is what
    cytools.ntfe computes; the Aut(P) quotient is the extra step the paper takes
    and is required to reproduce its counts.
"""

import argparse
import os
import sys
import time
from collections import Counter

os.environ.setdefault("CYTOOLS_DB_DIR", os.path.expanduser("~/Downloads/polytopes-4d"))

PAPER = {
    # h11: (polys_fav, polys_nonfav, frst_fav, frst_nonfav, cls_fav, cls_nonfav)
    1: (5, 0, 5, 0, 5, 0),
    2: (36, 0, 48, 0, 36, 0),
    3: (243, 1, 525, 1, 274, 1),
    4: (1185, 12, 5330, 18, 1760, 14),
    5: (4897, 93, 56714, 336, 11713, 134),
}
PAPER_FAV_ONLY = {6: (16608, 584281, 74503), 7: (48221, 5990333, 467283)}


def twoface_key(triang):
    return frozenset(
        frozenset(s) for s in triang.simplices(on_faces_dim=2, as_np_array=False)
    )


def analyze(verts):
    """FRST count and Aut-quotiented FRST-class count for one polytope."""
    from cytools import Polytope

    p = Polytope(verts)
    fav = p.is_favorable(lattice="N")
    keys = set()
    n_frst = 0
    for t in p.all_triangulations():
        n_frst += 1
        keys.add(twoface_key(t))
    autos = p.automorphisms(as_dictionary=True)
    seen, n_cls = set(), 0
    for k in keys:
        if k in seen:
            continue
        n_cls += 1
        for a in autos:
            seen.add(frozenset(frozenset(a[i] for i in s) for s in k))
    return fav, n_frst, n_cls


def _worker(verts):
    try:
        return analyze(verts)
    except Exception as e:  # keep a scan alive; report at the end
        return ("ERR", repr(e), None)


def run(h11, workers, limit=None):
    from cytools.dataset import load_polytopes

    t0 = time.time()
    recs = load_polytopes(h12=h11)
    vertlists = [r.polytope.vertices().tolist() for r in recs]
    if limit:
        vertlists = vertlists[:limit]
    t_load = time.time() - t0

    t0 = time.time()
    if workers == 1:
        results = [_worker(v) for v in vertlists]
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            results = list(ex.map(_worker, vertlists, chunksize=8))
    t_calc = time.time() - t0

    errs = [r for r in results if r[0] == "ERR"]
    good = [r for r in results if r[0] != "ERR"]
    agg = Counter()
    for fav, n_frst, n_cls in good:
        tag = "fav" if fav else "non"
        agg[f"polys_{tag}"] += 1
        agg[f"frst_{tag}"] += n_frst
        agg[f"cls_{tag}"] += n_cls

    print(
        f"\n=== h11 = {h11} ===  ({len(vertlists)} polytopes, "
        f"load {t_load:.1f}s, compute {t_calc:.1f}s, workers={workers})"
    )
    if errs:
        print(f"  !! {len(errs)} errors, first: {errs[0][1]}")
    mine = (
        agg["polys_fav"],
        agg["polys_non"],
        agg["frst_fav"],
        agg["frst_non"],
        agg["cls_fav"],
        agg["cls_non"],
    )
    labels = [
        "# polys fav",
        "# polys non",
        "# FRSTs fav",
        "# FRSTs non",
        "# FRST cls fav",
        "# FRST cls non",
    ]
    if h11 in PAPER and not limit:
        print(f"  {'quantity':<16} {'this fork':>10} {'paper':>10}   status")
        allok = True
        for lab, m, p in zip(labels, mine, PAPER[h11], strict=True):
            ok = m == p
            allok &= ok
            print(f"  {lab:<16} {m:>10} {p:>10}   {'MATCH' if ok else 'DIFF'}")
        print(f"  --> h11={h11}: {'ALL MATCH' if allok else 'MISMATCH'}")
    else:
        if h11 in PAPER_FAV_ONLY and not limit:
            pp, pf, pc = PAPER_FAV_ONLY[h11]
            print(f"  {'quantity':<16} {'this fork':>10} {'paper(fav)':>10}   status")
            for lab, m, p in (
                ("# polys fav", agg["polys_fav"], pp),
                ("# FRSTs fav", agg["frst_fav"], pf),
                ("# FRST cls fav", agg["cls_fav"], pc),
            ):
                print(f"  {lab:<16} {m:>10} {p:>10}   {'MATCH' if m == p else 'DIFF'}")
            print(
                f"  (non-fav, not in paper: polys={agg['polys_non']} "
                f"FRSTs={agg['frst_non']} cls={agg['cls_non']})"
            )
        else:
            for lab, m in zip(labels, mine, strict=True):
                print(f"  {lab:<16} {m:>10}")
    sys.stdout.flush()
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h11", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    for k in a.h11:
        run(k, a.workers, a.limit)
