# Reproducing Naomi Gendler's CYTools papers with this fork

## Papers surveyed (13 physics papers, arXiv author search)
| arXiv | Title | Uses CYTools |
|---|---|---|
| 2112.04503 | PQ Axiverse | yes |
| 2204.06566 | Superpotentials from Singular Divisors | (bib only) |
| 2212.10573 | Moduli Space Reconstruction and Weak Gravity | yes |
| 2309.01831 | Axion minima in string theory | yes |
| 2309.13145 | Glimmers from the Axiverse | yes |
| **2310.06820** | **Counting Calabi-Yau Threefolds** | **yes — reproduced** |
| 2404.15414 | Axions in the Dark Dimension | no |
| 2407.07143 | QCD Axion Dark Matter in String Theory | yes |
| 2412.12012 | Fuzzy Axions and Associated Relics | yes |
| 2507.03535 | Constraining the axiverse with reionization | yes |
| 2507.12516 | Universality in the Axiverse | yes |
| **2603.11173** | **Holes in Calabi-Yau Effective Cones** | **yes — reproduced** |
| 2608.14780 | Catastrophic Inflation in the Axiverse | yes |

## 1. arXiv:2310.06820 "Counting Calabi-Yau Threefolds" — COMPLETE MATCH

Every combinatorial count in the paper reproduced exactly (36 numbers).

### Table 1 (h11 = 1..5), favorable + non-favorable
| h11 | # polys | # FRSTs | # FRST classes |
|---|---|---|---|
| 1 | 5+0 ✓ | 5+0 ✓ | 5+0 ✓ |
| 2 | 36+0 ✓ | 48+0 ✓ | 36+0 ✓ |
| 3 | 243+1 ✓ | 525+1 ✓ | 274+1 ✓ |
| 4 | 1185+12 ✓ | 5330+18 ✓ | 1760+14 ✓ |
| 5 | 4897+93 ✓ | 56714+336 ✓ | 11713+134 ✓ |

### Table 4 (h11 = 6,7), favorable only
| h11 | # polys | # FRSTs | # FRST classes |
|---|---|---|---|
| 6 | 16,608 ✓ | 584,281 ✓ | 74,503 ✓ |
| 7 | 48,221 ✓ | 5,990,333 ✓ | 467,283 ✓ |

Independently confirms Gendler et al. against the competing count of ref.
[oxford], which quotes 4,896 and 16,607 favorable polytopes at h11=5 and 6;
we get 4,897 and 16,608, agreeing with Gendler et al.

### Finding: what "FRST class" means, operationally
`cytools.ntfe.ntfe_frsts` enumerates FRSTs up to *two-face equivalence* and
agrees with brute force exactly (verified at h11=2: both give 39). The paper's
count is 36. The missing step is a quotient by **Aut(P)**: exactly 3 polytopes
at h11=2 carry a coordinate-swap automorphism that identifies their 2 classes
into 1. Adding the Aut(P) orbit quotient reproduces the paper at every h11
from 1 to 7. So: paper's "FRST class" = Aut(P)-orbit of the induced two-face
triangulation, not the two-face class alone.

## 2. arXiv:2603.11173 "Holes in Calabi-Yau Effective Cones" — ENUMERATION MATCHES

Non-trivial Hilbert basis elements of the toric effective cone E_V.

| h11 | geometries with non-trivial HB elts | paper | with strictly-interior (big) one | paper |
|---|---|---|---|---|
| 2 | 1 ✓ | 1 | 0 ✓ | 0 |
| 3 | 13 ✓ | 13 | 0 ✓ | 0 |
| 4 | 88 ✓ | 88 | 4 ✓ | 4 |
| 5 | 434 | (not quoted) | 4 ✓ | 4 |

h11=3: the h21 multiset matches exactly — {43,45,51,59,63,83,89,93,95,99,105,105,165}
with the same number of classes per geometry (45 and 105 carry two).

Where the paper prints the GLSM charge matrix, ours is **identical element for
element**:
 * X_{2,106}: [[1,1,1,3,0,2],[0,0,0,1,1,-2]] — hole (1,-1) ✓
 * X_{3,51}:  [[1,0,1,1,0,-2,0],[0,1,0,1,1,2,0],[0,0,0,0,0,1,1]] — hole (-1,1,1) ✓

8 of 13 class vectors at h11=3 agree verbatim. The other 5 differ by the
GL(h11,Z) divisor-basis convention: the paper's vectors mostly do not even lie
inside our cone, and never appear in our Hilbert basis while being absent from
our list — i.e. a basis drift in CYTools' "deterministic default basis"
between the paper's version and this fork, not a computational disagreement.

### Caveat: index ordering
The paper indexes polytopes by KS listing order ("the (n+1)th favorable
polytope"). Our local Parquet mirror is NOT in that order — X_{2,106} is the
paper's index 19 but our index 3. Matching is by (h11,h21) + charge matrix.

## 3. arXiv:2212.10573 "Moduli Space Reconstruction" — cross-validation
Its ensemble is "1464 four-dimensional reflexive polytopes with 2 <= h11 <= 4".
Our favorable counts: 36 + 243 + 1185 = **1464** exactly. Independent
confirmation of the favorability split from a second paper.

## 4. Performance work

Profiling the reproduction workloads (the real paper pipeline, not microbench
toys) found three wins. Measured as HEAD vs HEAD-with-these-reverted, in
isolated pristine trees, 3 interleaved rounds, min-of-3, on an idle machine.
**All checksums and all triangulation counts identical.**

| workload | baseline | optimized | delta |
|---|---|---|---|
| `h11(lattice="N")` | 1.08s | 0.88s | **-17.7%** |
| `h21(lattice="N")` | 1.07s | 0.93s | **-13.0%** |
| `is_favorable("N")` | 1.04s | 0.88s | **-16.0%** |

FRST enumeration (`all_triangulations`), and how the win scales:

| h11 | FRSTs | baseline | optimized | delta |
|---|---|---|---|---|
| 3 | 526 | 1.01s | 0.97s | -3.3% |
| 4 | 753 | 1.14s | 1.05s | -7.7% |
| 5 | 1,607 | 3.18s | 2.56s | -19.5% |
| 6 | 5,050 | 14.38s | 10.46s | -27.3% |
| 7 | 12,165 | 70.01s | 44.25s | **-36.8%** |

### The three changes
1. **`Polytope.hpq` short-circuit** (`polytope.py`). The Batyrev sum computed
   `len(f.interior_points()) * len(f.dual().interior_points())` for every face.
   Python evaluates both factors, so it built the dual polytope's face lattice
   even when the first factor is 0 — which is the case for **94-99.9%** of
   2-faces. Skip the dual when the face has no interior points.
2. **`PolytopeFace._process_points`** (`polytopeface.py`). Runs once per face
   and scans every ambient point: O(#faces x #points) frozenset subset tests
   (1.58M `issubset` calls, and 1.28M `ambient_poly` *property* lookups, for
   380 polytopes). Hoisted the property/dict/bound-method out of the loop and
   fused the two passes into one.
3. **`all_triangulations` star filter** (`triangulation.py`). The star filter
   ran *after* mapping every triangulation's indices to labels; at h11=5,
   **96.4%** of fine triangulations are non-star, so nearly all that mapping
   was waste. Moved the test ahead of the mapping.

### A wrong turn worth recording
The first version of (3) tested `star_idx in s` on the raw simplices — and made
things **5.5% slower**. Cause: `t.simplices` is a numpy ndarray, and
`x in <numpy row>` is **~27x slower** than `in` on a Python list (0.17s vs
0.0063s per 100k). The original code got fast list membership as an accidental
side effect of the label mapping it was doing. Fix: test the whole
triangulation in one vectorized op, `(simps == star_idx).any(axis=1).all()`.
That turned +5.5% into -18.5%. The lesson: moving a filter earlier is only a
win if the cheaper position is also a cheap *representation*.

### 4th change: vectorized two-face restriction (`Triangulation.simplices`)
The `on_faces_dim` restriction was a double loop over (face, simplex) doing a
frozenset intersection each time -- O(#faces x #simplices) set operations, ~10M
of them in a single h11=5 FRST sweep. Replaced with a boolean face-by-label
incidence matrix and one vectorized membership count, materializing only the
pairs that meet the dimension condition.

| h11 | triangulations | before | after | delta |
|---|---|---|---|---|
| 5 | 602 | 0.049s | 0.044s | **-10.4%** |
| 7 | 6,459 | 0.826s | 0.648s | **-21.6%** |

End-to-end on full FRST enumeration this is only ~2-5%, because that pipeline
is dominated by the native enumerator (~38%) and the LP regularity checks
(~23%). Reported both ways deliberately: the function is meaningfully faster,
the sweep is not transformed.

Verified by differential comparison over **163,715** restriction records
(`on_faces_dim` 1/2/3, `split_by_face` on and off, numpy and set output) against
the previous implementation: values identical. Side benefit -- the old path
leaked numpy `uint8` labels into the returned frozensets (which breaks
`json.dumps`); the new one yields uniform Python `int`.

## Where the remaining time goes (profile of the paper workload, h11=5)
| cost | share | touchable from Python? |
|---|---|---|
| `triangulumancer.all_triangulations` (native C++) | 38% | no |
| LP regularity checks (`highspy`) | ~6% | no (external solver) |
| `_secondary_cone_hyperplanes_native` | ~17% cum | already batched |
| two-face restriction | ~7% | **done (above)** |

Two avenues not pursued:
* `triangulumancer.all_connected_triangulations` exists and is not used. If the
  flip-connected component coincides with the regular triangulations for 4d
  reflexive polytopes, the entire secondary-cone + LP regularity pass (~23%)
  could be skipped. That is a mathematical assumption, not a refactor, so it
  needs proving before it can be relied on.
* `_secondary_cone_hyperplanes_native` still issues `dim+2` separate batched
  `np.linalg.det` / `np.delete` calls per triangulation (28,758 and 47,930 calls
  in the profiled run). Collapsing them into one call over a precomputed index
  array would cut numpy dispatch overhead; bounded by ~6% of the sweep.

## The real win: use `ntfe_frsts`, not `all_triangulations`, for class counts

Enumerate-then-dedup is the wrong algorithm for the "# FRST classes" column.
`cytools.ntfe.ntfe_frsts` returns one FRST per two-face class directly, via
secondary-cone geometry, so it never materializes the redundancy. Both routes
give **identical class sets** (verified at h11=3..7):

| h11 | FRSTs | classes | `all_triangulations`+dedup | `ntfe_frsts` | speedup |
|---|---|---|---|---|---|
| 3 | 526 | 306 | 1.09s | 2.04s | 0.5x |
| 4 | 497 | 256 | 0.76s | 1.25s | 0.6x |
| 5 | 1,178 | 212 | 2.30s | 1.22s | 1.9x |
| 6 | 3,714 | 173 | 9.93s | 0.74s | **13.4x** |
| 7 | 9,064 | 132 | 30.78s | 0.45s | **68.9x** |

Crossover is ~h11=5; below that the enumerate route wins. Full-database effect
on the paper's two hardest numbers (`classes_fast.py`, 9 workers):

| | enumerate-then-dedup | via `ntfe` | result |
|---|---|---|---|
| h11=6 classes | 276.4s | **79.7s** | 74,503 (paper: 74,503) |
| h11=7 classes | 5,824.9s | **306.2s** | 467,283 (paper: 467,283) |

97 minutes to 5 minutes. Caveat, stated plainly: this accelerates the *classes*
column only. The paper's "# FRSTs" column is a count of all FRSTs, so it still
requires full enumeration -- there is no shortcut to counting things you must
enumerate. Non-favorable class counts (1,068 at h11=6, 8,126 at h11=7) agree
between the two routes as well.

## 5th change: batched minors in `_secondary_cone_hyperplanes_native`
`dim+2` separate `np.delete` + `np.linalg.det` pairs per triangulation (28,758
det and 47,930 delete calls in the profiled sweep) collapsed into one fancy
index against a cached drop table plus a single batched `det`. Verified over
9,066 secondary-cone records with **zero** flint fallbacks in either version,
i.e. the batched floats round to bit-identical integers.

Honest size: **~1-2%** end-to-end (-1.6/-2.6/-0.1/-1.5/-2.1% at h11=3..7). The
determinants were already batched over facet pairs, so only numpy dispatch
overhead was recovered. Kept because it is verified and also removes two loops;
drop it without loss if you want a minimal diff in that numerically delicate
function.

## Next targets (profiled, with mechanism — not guesses)

Once you switch class counting to `ntfe_frsts`, the hot path moves. Profile of
the ntfe route (h11=7, 250 polytopes, 8.85s):

| cost | share | note |
|---|---|---|
| `Triangulation.__init__` | **3.07s cum (35%)** | 11,215 calls |
| `Polytope._process_points` + `ppl_hull` | 1.5s cum | **8,541 Polytope builds for 250 inputs** |
| LP (`highspy`) | 0.41s | external |
| `_secondary_cone_hyperplanes_native` | 1.27s cum | already batched |

**The lead: `ntfe.face_triangulations._as_2d_poly` builds a fresh `Polytope`
per 2-face.** Every 2-face of a 4d reflexive polytope lives in ZZ^4, so the
`ambient_dim() == 2` fast path never fires and each face pays a convex hull
(`ppl_hull`) plus `saturating_lattice_pts` — ~34 Polytope constructions per
input polytope. Two-faces of reflexive polytopes are overwhelmingly the same
few small shapes, so the geometric work is almost all redundant.

Care required: the docstring's warning is real — `labels=poly.labels` is
load-bearing, and labels differ per face. So memoize the *geometric* part
(hull / inequalities / saturating points) keyed on the canonical point set and
re-attach labels, rather than caching the `Polytope` object. The module already
has a persistent `_ineq_cache` for two-face inequalities, so the pattern exists;
it just does not cover construction.

### Ruled out
* **`all_connected_triangulations` is not a regularity shortcut.** Tested
  directly: it returns a *different, larger* set than the regular
  triangulations (796 vs 380 regular at h11=5; 175 vs 144 at h11=4), so it
  cannot replace the secondary-cone + LP regularity pass.
* **Python micro-optimization of the enumerate path is exhausted.** After the
  five changes above the profile is 38% native C++ enumerator and ~23%
  external LP solver; every remaining Python candidate measured 1-5%.

## Status: what of Gendler's work is reproduced

### arXiv:2310.06820 "Counting Calabi-Yau Threefolds"
| result | status |
|---|---|
| Table 1: # polytopes / # FRSTs / # FRST classes, h11=1..5 (30 numbers) | **exact** |
| Table 4: same three columns, h11=6,7 (6 numbers) | **exact** |
| Table 1 "# CYs" (Wall classes), h11=1 | **exact** (5 = 4 + 1; `GL(1,Z)={-1,1}`) |
| Table 1 "# CYs" (Wall classes), h11=2 | **matches** (29 = 27 + 2; stable over tested bounds) |
| Table 1 "# CYs", h11=3 | upper bound only -- see below |

The "# CYs" column is the paper's *headline* claim, and needs Wall data
(`kappa_ijk`, `c_2`) rather than combinatorics -- `wall_classes.py`. Two
geometries are equivalent iff some `Lambda` in `GL(h11,Z)` carries one pair to
the other. At h11=1 the search is exhaustive. At h11=2 the answer stabilizes
as the bound grows (30 -> 29 -> 29), which is strong reproduction evidence but
not a proof of inequivalence: `GL(2,Z)` is infinite.

**Why h11>=3 needs a different algorithm.** A bounded search can prove
equivalence (by exhibiting `Lambda`) but never inequivalence, so it can only
over-count. At h11=3, bound=1 gives 210 against the true 186. Brute force is
also exponential in h11. The paper's actual method is to use *GV invariants* to
guess candidate `Lambda` -- generators of the Mori cone with equal GV invariants
must map to each other, which collapses the search to a handful of candidates.
Implementing that is the next step for this column.

Also worth noting: the bucketing here must use only genuinely basis-independent
invariants. Sorted `kappa_iii`, `kappa.sum()`, `c2.sum()` are *not* invariant
under `GL(n,Z)` -- an early version of this script used them and inflated the
count by splitting equivalent geometries.

### arXiv:2603.11173 "Holes in Calabi-Yau Effective Cones" -- complete range
| h11 | geometries with non-trivial Hilbert basis elts | with a strictly-interior one | paper |
|---|---|---|---|
| 2 | 1 | 0 | 1 / 0 |
| 3 | 13 | 0 | 13 / 0 |
| 4 | 88 | 4 | 88 / 4 |
| 5 | 434 | 4 | -- / 4 |
| 6 | 1,587 | 9 | -- / 9 |

Every count the paper states is matched, across its full h11 <= 6 range.

### arXiv:2212.10573
Ensemble size "1464 polytopes with 2 <= h11 <= 4" == our favorable counts
36 + 243 + 1185. Independent confirmation of the favorability split.
