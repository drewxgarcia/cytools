"""Reproduce the central efficiency claim of arXiv:2507.12516
(Cheng & Gendler, "Universality in the Axiverse").

The paper argues that axion decay constants -- normally obtained from the
eigenvalues of the h11 x h11 Kahler metric K_ij -- can be approximated using
*only divisor volumes*:

    eig(K_ij)  ~  1 / ( tau_max^{3/2} * sqrt(tau_i) )

with tau_i the divisor volumes. It calls this "a nearly instantaneous
approximation" replacing "computationally expensive scans". This script checks
both halves of that: whether the approximation tracks the exact spectrum, and
what it actually saves.
"""

import argparse
import os
import sys
import time

import numpy as np

os.environ.setdefault("CYTOOLS_DB_DIR", os.path.expanduser("~/Downloads/polytopes-4d"))


def one(verts, c=1.0):
    from cytools import Polytope

    p = Polytope(verts)
    if not p.is_favorable(lattice="N"):
        return None
    cy = p.triangulate().get_cy()
    tloc = cy.toric_kahler_cone().tip_of_stretched_cone(c)

    # --- exact: build K_ij, then its eigenvalues
    t0 = time.perf_counter()
    K = np.asarray(cy.compute_kahler_metric(tloc), dtype=float)
    eig_exact = np.sort(np.linalg.eigvalsh(K))[::-1]
    t_exact = time.perf_counter() - t0

    # --- approximation: divisor volumes only
    t0 = time.perf_counter()
    tau = np.asarray(cy.compute_divisor_volumes(tloc, in_basis=True), dtype=float)
    tau = np.abs(tau)
    tau[tau == 0] = np.finfo(float).tiny
    eig_approx = np.sort(1.0 / (tau.max() ** 1.5 * np.sqrt(tau)))[::-1]
    t_approx = time.perf_counter() - t0

    return cy.h11(), eig_exact, eig_approx, t_exact, t_approx


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / d) if d else float("nan")


def run(h11, n):
    from cytools.dataset import load_polytopes

    verts = [
        r.polytope.vertices().tolist() for r in load_polytopes(h12=h11, n=n + 1, seed=7)
    ]
    # burn one geometry: the first compute_kahler_metric pays one-time costs
    # (BLAS/solver init) that would otherwise be charged to the exact path
    for v in verts:
        try:
            if one(v) is not None:
                break
        except Exception:
            continue
    verts = verts[1:]
    Te = Ta = 0.0
    rows = []
    for v in verts:
        try:
            r = one(v)
        except Exception as e:
            print(f"    skip: {type(e).__name__}: {str(e)[:70]}", file=sys.stderr)
            continue
        if r is None:
            continue
        _, ee, ea, te, ta = r
        Te += te
        Ta += ta
        # compare as *distributions* (the paper's claim), via log-spectrum
        # correlation and median decades of offset
        le, la = np.log10(ee), np.log10(ea)
        rows.append((spearman(le, la), np.median(la - le), len(ee)))
    if not rows:
        print(f"h11={h11}: no usable geometries")
        return
    rho = np.mean([r[0] for r in rows])
    off = np.median([r[1] for r in rows])
    print(
        f"h11={h11:>4}  geoms={len(rows):>3}  modes/geom={rows[0][2]:>4}  "
        f"exact={Te:>7.3f}s  approx={Ta:>7.4f}s  speedup={Te / max(Ta, 1e-9):>7.1f}x  "
        f"rank-corr(log spectrum)={rho:+.3f}  median log10 offset={off:+.2f}"
    )
    sys.stdout.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h11", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    for k in a.h11:
        run(k, a.n)
