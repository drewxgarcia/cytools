# CYTools Refactoring Roadmap

Living document. Each item has **evidence** (a measurement, not an opinion), an
**action**, and a **done-when**. Check items off as they land.

Baseline measured 2026-08-31 on `dvg/data-plane`, 30,387 LOC across `src/cytools`.
Test baseline: **387 passed, 5 skipped** (the 5 are mathematically irreducible —
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
      The three remaining wildcard-export compatibility modules have narrow,
      documented `F403` exceptions.
- [ ] **1.2** Enable the remaining Ruff families in reviewed batches. There are
      currently 256 non-line-length, non-ambiguous-variable findings outside
      the enforced floor; do not apply unsafe fixes without numerical tests.
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

- [ ] **5.1** Extract the shared cache-key/lookup/rounding scaffolding into one
      helper; leave the differing geometry in place.
      *Done when:* the identical runs are gone and both call one helper.

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

- [ ] **7.1** CI installs only `--extra normaliz`. The `gnn` and `performance`
      extras are never exercised. Today's finding: sksparse + torch together
      **abort the interpreter** (two OpenMP runtimes) — exactly the class of bug
      CI is blind to.
      *Done when:* a CI job installs both extras and runs the suite.
- [ ] **7.2** Document the libomp conflict and its fix (single shared runtime,
      not `KMP_DUPLICATE_LIB_OK`) in `INSTALL.md`.
- [ ] **7.3** A plain `uv sync` prunes the `gnn` and `performance` extras and
      restores torch's bundled `libomp.dylib`, silently returning the suite to
      15 skips and re-arming the abort. Local verification currently needs
      `uv sync --extra gnn --extra performance --extra normaliz` followed by
      re-pointing `torch/lib/libomp.dylib` at the homebrew runtime. Make this
      reproducible (a documented dev-setup target) rather than folklore.

## Phase 8 — Long tail

- [ ] **8.1** Consolidate the triplicated import-time FPU mutation
      (`ctypes.CDLL(None).fesetround(0)` in `polytope.py`, `cone.py`,
      `h_polytope.py`). **Do not simply delete**: none of the 9 native imports
      changes the rounding mode on arm64 macOS with current versions, but this
      may be a real fix for x86-64 or older `ppl`. Verify cross-platform, then
      move to one place.
- [ ] **8.2** God functions: `fan.intersection_numbers` (390 lines),
      `normal_form` (317), `find_lattice_points` (310, 12 params),
      `Triangulation.__init__` (295), `fetch_polytopes` (262 lines, **23
      parameters**).
- [ ] **8.3** Drop the redundant `num_2face_triangs` alias of `n_2face_triangs`.
- [ ] **8.4** Review the 5 `except …: pass` sites; narrow or comment each.
- [ ] **8.5** Docstring typo "triangularions" (2 files).

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
