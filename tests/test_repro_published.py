"""Published numbers, re-derived on every run.

This fork's strongest claim is that it reproduces the combinatorial counts of
arXiv:2310.06820 exactly. That claim lived in prose in a benchmark README,
where nothing could break it noticeably. Here it is executable.

The counts are *counts*, so they need a complete stratum rather than a sample:
the committed slice holds every 4D reflexive polytope with `h11(N) <= 5`, all
6,472 of them, which is what makes exact agreement meaningful. See
`tests/fixtures/build_ks_slice.py`.

Definitions are the paper's, and the Aut(P) quotient that turns a two-face
equivalence class into the paper's "FRST class" is imported from the
reproduction script rather than restated, so the test and the script cannot
drift apart.
"""

import pytest

from cytools.dataset import load_polytopes

#: Table 1 of arXiv:2310.06820, as `h11: (polytopes, FRSTs, FRST classes)`
#: with each entry `(favorable, non-favorable)`.
TABLE_1 = {
    1: ((5, 0), (5, 0), (5, 0)),
    2: ((36, 0), (48, 0), (36, 0)),
    3: ((243, 1), (525, 1), (274, 1)),
    4: ((1185, 12), (5330, 18), (1760, 14)),
    5: ((4897, 93), (56714, 336), (11713, 134)),
}


def favorability_split(h11: int) -> tuple[int, int]:
    """Favorable and non-favorable polytope counts at fixed N-lattice h11."""
    records = load_polytopes(h12=h11)
    favorable = sum(1 for r in records if r.polytope.is_favorable(lattice="N"))
    return favorable, len(records) - favorable


def triangulation_split(h11: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """FRST and FRST-class counts at fixed h11, split by favorability."""
    from benchmarks.repro.counting_cy import analyze

    frsts = [0, 0]
    classes = [0, 0]
    for record in load_polytopes(h12=h11):
        favorable, n_frsts, n_classes = analyze(record.polytope.vertices())
        index = 0 if favorable else 1
        frsts[index] += n_frsts
        classes[index] += n_classes
    return (frsts[0], frsts[1]), (classes[0], classes[1])


@pytest.mark.parametrize("h11", sorted(TABLE_1))
def test_polytope_counts_match_the_published_table(h11):
    """Column 1 of Table 1, every row.

    A wrong favorability verdict on a single polytope moves one of these by
    one, so the pair is a sharp check on `is_favorable` over 6,472 real
    polytopes -- not merely on the row count.
    """
    assert favorability_split(h11) == TABLE_1[h11][0]


@pytest.mark.parametrize("h11", [1, 2, 3])
def test_triangulation_counts_match_the_published_table(h11):
    """Columns 2 and 3 of Table 1, over the rows that are quick to enumerate.

    The class count is the interesting one: `ntfe` enumerates FRSTs up to
    two-face equivalence, and the paper additionally quotients by Aut(P). At
    h11 = 2 that is the difference between 39 two-face classes and the
    published 36, so this pins the definition and not just the arithmetic.
    """
    frsts, classes = triangulation_split(h11)
    assert (frsts, classes) == (TABLE_1[h11][1], TABLE_1[h11][2])


@pytest.mark.slow
@pytest.mark.parametrize("h11", [4, 5])
def test_triangulation_counts_match_the_published_table_at_larger_h11(h11):
    """The same check where enumeration is expensive: ~10 s at h11 = 4."""
    frsts, classes = triangulation_split(h11)
    assert (frsts, classes) == (TABLE_1[h11][1], TABLE_1[h11][2])


def test_the_moduli_space_paper_ensemble_has_the_size_it_reports():
    """A second paper agreeing with the first, on the same underlying split.

    arXiv:2212.10573 describes its ensemble as "1464 four-dimensional
    reflexive polytopes with 2 <= h11 <= 4". That total is the sum of the
    favorable counts above, so it is independent corroboration of the
    favorability split rather than a restatement of it.
    """
    assert sum(favorability_split(h11)[0] for h11 in (2, 3, 4)) == 1464


def test_the_slice_is_complete_for_the_strata_it_covers():
    """The premise every count above depends on.

    6,472 is the paper's 6,366 favorable plus 106 non-favorable. If the slice
    were ever regenerated from an incomplete database, the counts would drift
    quietly; this states the premise so it fails loudly instead.
    """
    total = sum(len(load_polytopes(h12=h11)) for h11 in TABLE_1)
    published = sum(sum(TABLE_1[h11][0]) for h11 in TABLE_1)

    assert total == published == 6472
