# CYTools Refactoring Roadmap

Living document. Each item has **evidence** (a measurement, not an opinion), an
**action**, and a **done-when**. Check items off as they land.

Baseline measured 2026-08-31 on `dvg/data-plane`, 30,387 LOC across `src/cytools`.
Safe-default test baseline: **387 passed, 15 skipped** (10 optional-backend
skips plus the 5 mathematically irreducible cases —
see Appendix B). All four gates green: `ruff check`, `ruff format --check`,
`ty check`, pytest.

---

## Phase 0 — Tooling (highest leverage; unblocks Phases 1–4)

The build workflow now has a dedicated quality job for Ruff formatting,
Pyflakes correctness, and ty. Broader modernization rules are deliberately
staged so that every CI gate is enforceable when introduced.

- [x] **0.1** Add `[tool.ruff]` to `pyproject.toml` (line-length, target-version,
      lint rule selection).
      *Done when:* `ruff check` and `ruff format --check` are runnable with
      committed config.
- [x] **0.2** Add `ruff` + `ty` to the dev dependency group.
      *Done when:* `uv run ruff --version` works from a clean sync.
- [x] **0.3** Add a `lint` job to `.github/workflows/build-test.yml`.
      *Done when:* CI fails on a formatting or lint regression.
- [ ] **0.4** Add `.git-blame-ignore-revs` after the bulk-format commit so it does not
      poison `git blame`.

## Phase 1 — Real defects surfaced by lint

- [x] **1.1** Make the syntax/Pyflakes lint floor clean across `src/`, removing
      unused imports and names and fixing the undefined face-enumeration state.
      The former wildcard-export feature modules now have explicit public
      namespaces, so no `F403` exceptions remain.
- [ ] **1.2** Enable the remaining Ruff families in reviewed batches. Import
      ordering is now enforced; the remaining semantic/style families must be
      enabled without unsafe fixes unless numerical tests justify them.
      *Done when:* the full intended rule set is green and enabled in CI.

### Defects fixed while clearing the gates

- `Fan.restricted_simps(padded=True, as_face_inds=False)` raised
  `TypeError: 'frozenset' object is not subscriptable`. The reduction step
  yields frozensets but the padding step did `simp + [simp[-1]]`. It only fired
  when a restricted simplex actually had two points — i.e. exactly when
  `padded` has work to do — which is why `to_dim=2` (triangles) hid it and
  `to_dim=1` exposes it. The method had **zero callers** in src, tests, or
  benchmarks, so nothing caught it. Fixed by normalising both branches to
  lists; covered by two new tests in `tests/test_fan.py`, confirmed failing
  against the old code.
- Five `type(x) == type(None)` / `type(x) == type([])` comparisons in
  `Uplift_functions.py` → `is None` / `isinstance`.
- A dead local in `Uplift_functions.py` that duplicated the array allocation
  passed inline to `milp()`.
- Two misplaced/dead `ty: ignore` directives (one sat on the imported name
  while the diagnostic was on the module line, so it suppressed nothing).

## Phase 2 — Formatting drift

**Evidence:** max line length by module — `f_theory/FT_CY.py` **465 chars**,
`f_theory/Uplift_functions.py` 212, everything else ≤165. `FT_CY.py` packs 4–5
parameters per line with no space after commas (`bool=True`, `,backend:`).

- [x] **2.1** Format the `f_theory/` subpackage (the outlier, ~3.7k lines).
- [x] **2.2** Format the remaining package in one isolated, behaviour-free
      commit; record its SHA in `.git-blame-ignore-revs`.
      *Done when:* `ruff format --check src/` passes and tests are unchanged.

## Phase 3 — API vocabulary (the biggest user-facing smell)

**Evidence:** ~110 return-type-switching flag parameters across 33 distinct
names, with the *same concept spelled several ways*:

| Concept | Spellings found |
|---|---|
| indices | `as_indices` (22×), `as_poly_indices`, `as_inds`, `as_index`, `as_face_inds`, `as_triang_indices`, `as_vertex_index` |
| numpy array | `as_np_array` (7×), `as_np_arr` (2×), `as_np_array_output` |
| prefix | `as_*` (24 names) vs `return_*` (5 names) for the same idea |

Verified same-concept, not homonyms: `calabiyau.py` *"map charges to np.array
(True) or leave as set"* vs `triangulation.py` *"Return the simplices as a numpy
array. Otherwise… frozensets."*

A user cannot guess the parameter name. Fix the vocabulary first; collapsing the
flags themselves (Phase 6) depends on it.

- [x] **3.1** Canonicalise the numpy-array flag on `as_np_array`; alias
      `as_np_arr` / `as_np_array_output` with a `DeprecationWarning`.
- [ ] **3.2** Canonicalise the indices flags on `as_indices` where the meaning is
      plain "indices"; keep the qualified ones (`as_poly_indices`) only where
      they genuinely name a *different index space*, and rename the rest.
- [ ] **3.3** Settle on the `as_*` prefix; alias the five `return_*` names.
      *Done when:* one spelling per concept, old names warn for one release.

## Phase 4 — Stringly-typed enums

**Evidence:** 77 enum-like parameters (`backend`, `format`, `method`,
`triang_method`, `lattice`, `action`) — 69 annotated bare `str`, 8 unannotated,
**0** constrained by `Literal`. `backend="cgl"` is a runtime failure or a silent
fallback, never a type error. The codebase already uses `Literal` — but only
inside `@overload` stubs, never where it would catch a typo.

- [x] **4.1** Define shared aliases in `cytools/_typing.py` (`Backend`,
      `IntnumFormat`, `Lattice`, …) from the values each function accepts.
- [ ] **4.2** Apply them to the public API surface.
      *Done when:* `ty` flags a bad literal at a call site.

## Phase 5 — Duplication

**Evidence:** `ToricVariety.intersection_numbers` (358 lines) and
`CalabiYau.intersection_numbers` (287 lines) are 41.9% line-similar — 108 lines
across 8 contiguous runs of ≥4 identical lines. Not just docstrings: the
signature (15 lines), a docstring block (19), and the caching/dispatch logic are
copy-pasted. (`fan.py`'s third version is a different algorithm — 3.5% — and is
*not* duplication.)

- [x] **5.1** Extracted the 56-line post-processing block (anticanonical sign
      convention, basis change, format conversion) into
      `utils.finalize_intersection_numbers`. Duplication in runs of >=4 lines:
      **108 -> 63**, and what remains is the signature plus docstring, which two
      classes sharing a public API are entitled to. **This surfaced a real bug**:
      the two copies had diverged on one line, and the `ToricVariety` spelling
      wrote the sign-flipped tensor under the *requested* format's cache key
      while the conversion step below reads the `"dok"` key. On a fresh object
      `intersection_numbers(zero_as_anticanonical=True, format="coo"|"dense")`
      raised `KeyError: (True, False, False, 'dok')`; a prior `format="dok"`
      call masked it. `CalabiYau` had the correct spelling. Verified with a
      144-combination differential (3 polytopes x both classes x every flag
      combination): 6 differences, all `KeyError -> correct value`, and the new
      cold path matches the previously-working warm path exactly. Covered by
      `test_intersection_numbers_anticanonical_cold_cache`, confirmed failing
      against the old key.

## Phase 6 — Collapse the remaining return-type flags

Depends on Phase 3. Same treatment as the triangulation enumerators: one return
type per function; the alternative becomes a separate named method or the
caller's job.

- [ ] **6.1** `as_list` on the lazy `ntfe_*` enumerators (4 functions) — note
      these default *eager*, so this is a breaking default flip.
- [x] **6.2** `raw_output` on `all_triangulations` made the
      `Iterator[Triangulation]` annotation a lie. Split into two entry points
      with one honest return type each — `all_triangulations()` and the new
      `all_triangulation_simplices()` — over a private heterogeneous
      implementation. This was the last `ty` error.
- [ ] **6.3** `as_indices` family (6 blocks) — split or alias.
      *Explicitly out of scope:* `fetch_polytopes(as_list=)`. Its eager default
      is a deliberate upstream decision (CHANGELOG 1.4.3) on the library's front
      door, and it is a bounded query, not an unbounded enumeration.

## Phase 7 — Test/CI coverage gaps

- [x] **7.1** Added a dedicated backend CI job (macOS) that installs SuiteSparse,
      syncs every compiled backend, reconciles the OpenMP runtimes and runs the
      suite — the configuration that exercises CHOLMOD and the GNN sampler in
      one process. The 6-way matrix stays on the safe default plus Normaliz, so
      it does not build scikit-sparse or download PyTorch six times.
- [x] **7.2** Documented in `INSTALL.md`: the SuiteSparse header prerequisite,
      the duplicate-OpenMP failure mode, the symlink fix, and why
      `KMP_DUPLICATE_LIB_OK` is not the answer. Backed by
      `cytools._backends.openmp`, which raises an actionable Python exception
      before importing PyTorch when a second runtime is already loaded. Losing
      one optional backend is preferable to aborting a notebook kernel and its
      unsaved state. Covered by `tests/test_openmp_guard.py`.
- [x] **7.3** Kept incompatible compiled backends out of the default dependency
      group. Local and matrix testing use the safe environment; the dedicated
      backend job opts into all three extras explicitly and reconciles libomp
      before importing both CHOLMOD and PyTorch.

### Why the backends are not required dependencies

Considered and rejected, with measurements:

- **`gnn`** pulls PyTorch: **492 MB installed**. A Calabi-Yau geometry library
  should not impose that on users who never touch the GNN sampler.
- **`performance`** (`scikit-sparse`) is **sdist-only**: the installed wheel is
  `Generator: setuptools` with a local platform tag, and the built extension
  hard-links `/opt/homebrew/opt/suite-sparse/lib/libcholmod.5.dylib` — an
  absolute, machine-specific path. Requiring it would make `pip install cytools`
  fail on any machine without the SuiteSparse headers. CHANGELOG 1.4.x records
  the deliberate move *to* an optional extra with a SciPy fallback; reversing it
  would be a regression.
- Making them required would also force the duplicate-OpenMP hazard on every
  install rather than only on the combination that opts into both.

A dedicated opt-in CI job gives backend coverage without imposing those costs
or native-runtime hazards on every development checkout.

## Phase 8 — Long tail

- [x] **8.1** Consolidated the triplicated import-time FPU mutation behind
      `cytools._backends.ppl`. Every PPL consumer now shares one import boundary,
      which restores `FE_TONEAREST` exactly once after the engine loads. The
      low-level call has an explicit C signature and failure handling; unit,
      subprocess, and architecture tests preserve the native workaround for
      x86-64 and older PPL builds without scattering process-state mutation
      through domain modules.
- [ ] **8.2** God functions: `fan.intersection_numbers` (390 lines),
      `normal_form` (317), `find_lattice_points` (310, 12 params),
      `Triangulation.__init__` (295), `fetch_polytopes` (262 lines, **23
      parameters**).
- [ ] **8.3** ~~Drop~~ **reconsider** the `num_2face_triangs` alias of
      `n_2face_triangs`. It is pinned by `test_ntfe_enumeration.py` and
      `test_architecture.py`, so it is a deliberate public alias, not an
      oversight. Removing it is an API decision for the maintainer, not a
      cleanup.
- [x] **8.4** Reviewed all 5 `except …: pass` sites. Four were already
      correctly narrow (`ValueError`, `OSError`, `LinAlgError`,
      `(TypeError, IndexError, KeyError)`); each now states why swallowing is
      correct. Only `calabiyau.py`'s forkserver-preload catch is broad, and it
      is genuinely best-effort — documented rather than narrowed.
- [x] **8.5** Docstring typo "triangularions" fixed in 2 files.

## Phase 9 — Pipeline performance

Measured 2026-08-31 on the paper-realistic payload (`triangulate` ->
`intersection_numbers` -> `toric_kahler_cone` -> `tip_of_stretched_cone` ->
`compute_divisor_volumes`), favorable KS polytopes, medians of n=3.

- [x] **9.0** Reuse one lazily created process pool across every nonempty
      source batch in a materialization. A fully cached scan starts no workers;
      pool reuse and shutdown are pinned mechanically.
- [ ] **9.1** **Make the optional CHOLMOD path easy to install — worth ~3.0x
      end to end, with no algorithm change.** `intersection_numbers` is ~77% of
      the payload and 89% of that is one `solve_linear_system` call, which
      falls through the `backend="all"` waterfall to SciPy's SuperLU when
      scikit-sparse is absent. On the real h11(N)=300 system (M 28576x15238,
      nnz 158080): **CHOLMOD 39.1 ms vs SuperLU 793.8 ms**, with a better
      residual (1.9e-08 vs 3.7e-08). End to end 1105.9 -> 365.0 ms at h11=300,
      428.2 -> 142.1 ms at h11=150. Keep it in the dedicated backend CI job and
      optional `performance` extra: Phase 7 records why compiled, hard-linked
      scikit-sparse and its OpenMP interaction cannot return to the portable
      default environment.
- [ ] **9.2** After 9.1, the cone stages are the next target and the only ones
      with a bad exponent: `toric_kahler_cone` h11^2.13 and
      `tip_of_stretched_cone` h11^1.92, against h11^1.19-1.28 for everything
      else. Together they are 15.7% of the payload at h11=150 and 23.9% at
      h11=300, so they dominate as h11 grows. `tip` is the larger half (18.6%
      at h11=300) and is LP-bound, not sparse-solve-bound, so CHOLMOD does not
      help it.
- [ ] **9.3** Do not re-optimize triangulation or intnum assembly.
      `_construct_intnum_equations_4d` is **5%** of `intersection_numbers`;
      the assembly work already landed. With CHOLMOD in place no single stage
      exceeds ~31%, so kernel-level wins are capped near 1.3x.

---

## Appendix A — what is already clean

Recorded so effort is not wasted re-checking:

- **0** mutable default arguments.
- **19** type-suppressions across 30k lines.
- **12** broad `except` clauses.
- **No** import-time class patching — already refactored to `lazy_method(...)`.

## Appendix B — the 5 irreducible test skips

Properties of the fixtures, not the environment. No install fixes them, and
removing them would assert invalid mathematics.

- **3 skips** — `test_intnum_assembly.py` `poly[2]` (6-vertex) is non-reflexive.
  Even with `config.enable_experimental_features()` the toric variety builds but
  `intersection_numbers(in_basis=False)` fails: *"The GLSM charge matrix can only
  be computed for reflexive polytopes."*
- **2 skips** — `poly[0]` and `poly[3]` are reflexive but give **singular** toric
  varieties, and the test asserts *integral intersection numbers when smooth*.
