"""The "# FRST classes" column of arXiv:2310.06820, via cytools.ntfe instead of
enumerate-then-dedup.

`ntfe_frsts` returns one FRST per two-face equivalence class directly, so it
skips enumerating the (large) redundancy: at h11=7 the paper counts 5,990,333
FRSTs collapsing to 467,283 classes. Verified to give byte-identical class sets
to `all_triangulations` + dedup, and 13x/69x faster at h11=6/7.
The Aut(P) orbit quotient on top is what matches the paper -- see the README.
"""

import argparse
import os
import sys
import time

os.environ.setdefault("CYTOOLS_DB_DIR", os.path.expanduser("~/Downloads/polytopes-4d"))

PAPER_FAV_CLASSES = {1: 5, 2: 36, 3: 274, 4: 1760, 5: 11713, 6: 74503, 7: 467283}


def classes_for(verts):
    from cytools import Polytope

    p = Polytope(verts)
    fav = p.is_favorable(lattice="N")
    keys = {
        frozenset(frozenset(s) for s in t.simplices(on_faces_dim=2, as_np_array=False))
        for t in p.ntfe_frsts()
    }
    autos = p.automorphisms(as_dictionary=True)
    seen, n = set(), 0
    for k in keys:
        if k in seen:
            continue
        n += 1
        for a in autos:
            seen.add(frozenset(frozenset(a[i] for i in s) for s in k))
    return fav, n


def _w(v):
    try:
        return classes_for(v)
    except Exception as e:
        return ("ERR", repr(e))


def run(h11, workers):
    from cytools.dataset import load_polytopes

    t0 = time.time()
    verts = [r.polytope.vertices().tolist() for r in load_polytopes(h12=h11)]
    t_load = time.time() - t0
    t0 = time.time()
    if workers == 1:
        out = [_w(v) for v in verts]
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as ex:
            out = list(ex.map(_w, verts, chunksize=8))
    t_calc = time.time() - t0
    errs = [o for o in out if o[0] == "ERR"]
    fav = sum(n for f, n in (o for o in out if o[0] != "ERR") if f)
    non = sum(n for f, n in (o for o in out if o[0] != "ERR") if not f)
    exp = PAPER_FAV_CLASSES.get(h11)
    print(
        f"h11={h11}: polys={len(verts)} load={t_load:.1f}s compute={t_calc:.1f}s "
        f"workers={workers}"
    )
    print(
        f"  # FRST classes favorable = {fav}   paper = {exp}   "
        f"{'MATCH' if fav == exp else 'DIFF'}    (non-favorable = {non})"
    )
    if errs:
        print(f"  !! {len(errs)} errors, first: {errs[0][1]}")
    sys.stdout.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h11", type=int, nargs="+", default=[6])
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args()
    for k in a.h11:
        run(k, a.workers)
