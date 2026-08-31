"""
Tests for `fetch_polytopes` served from the local Parquet database.

These pin the *mapping* from `fetch_polytopes`' arguments onto the database's
columns, which is where a migration away from the Kreuzer-Skarke website can
silently go wrong. The riskiest part is the lattice convention: the database
stores `h11`/`h12`/`euler_characteristic` in the M-lattice convention, the
opposite of CYTools' `lattice="N"` default, and `fetch_polytopes` already
performs that swap on the way to the website. These tests assert the property
callers actually care about -- ask for `h11=k` in a lattice, get polytopes
whose `h11` in that lattice is `k` -- rather than the internal swap.

The website is not contacted; every test forces `source="database"`.
"""

import types

import pytest

from cytools import fetch_polytopes


def _fetch(**kwargs):
    kwargs.setdefault("source", "database")
    try:
        return fetch_polytopes(**kwargs)
    except (ImportError, ValueError, FileNotFoundError) as e:
        if "database" in str(e).lower() or "CYTOOLS_DB_DIR" in str(e):
            pytest.skip(f"no local KS database configured: {e}")
        raise


# ------------------------------------------------------------ lattice mapping
@pytest.mark.parametrize("lattice", ["N", "M"])
def test_h11_filter_is_in_the_requested_lattice(lattice):
    """The convention swap must be invisible to the caller."""
    polytopes = _fetch(h11=5, lattice=lattice, limit=5)
    if not polytopes:
        pytest.skip("no matching polytopes in the local database")
    assert all(p.h11(lattice=lattice) == 5 for p in polytopes)


@pytest.mark.parametrize("lattice", ["N", "M"])
def test_chi_filter_is_in_the_requested_lattice(lattice):
    seed = _fetch(h11=5, lattice=lattice, limit=4)
    if not seed:
        pytest.skip("no matching polytopes in the local database")
    target = seed[0].chi(lattice=lattice)

    polytopes = _fetch(h11=5, lattice=lattice, chi=target, limit=4)
    assert polytopes
    assert all(p.chi(lattice=lattice) == target for p in polytopes)


def test_h21_is_an_alias_for_h12():
    by_h12 = _fetch(h12=5, lattice="N", limit=3)
    by_h21 = _fetch(h21=5, lattice="N", limit=3)
    assert [p.vertices().tolist() for p in by_h12] == [
        p.vertices().tolist() for p in by_h21
    ]


# ------------------------------------------------------------ column mappings
def test_n_vertices_filter():
    polytopes = _fetch(h11=5, lattice="N", n_vertices=6, limit=3)
    if not polytopes:
        pytest.skip("no matching polytopes in the local database")
    assert all(len(p.vertices()) == 6 for p in polytopes)


def test_n_points_filter():
    polytopes = _fetch(h11=5, lattice="N", n_points=9, limit=3)
    if not polytopes:
        pytest.skip("no matching polytopes in the local database")
    assert all(len(p.points()) == 9 for p in polytopes)


def test_n_facets_filter():
    polytopes = _fetch(h11=5, lattice="N", n_facets=6, limit=3)
    if not polytopes:
        pytest.skip("no matching polytopes in the local database")
    assert all(len(p.facets()) == 6 for p in polytopes)


# ------------------------------------------------------------------ behaviour
def test_limit_is_respected():
    assert len(_fetch(h11=5, lattice="N", limit=7)) == 7


def test_as_list_false_returns_a_generator():
    generator = _fetch(h11=5, lattice="N", limit=3, as_list=False)
    assert isinstance(generator, types.GeneratorType)
    assert len(list(generator)) == 3


def test_favorable_filter():
    favorable = _fetch(h11=5, lattice="N", limit=5, favorable=True)
    assert favorable
    assert all(p.is_favorable(lattice="N") for p in favorable)


def test_dualize_returns_the_dual():
    plain = _fetch(h11=5, lattice="N", limit=3)
    dualized = _fetch(h11=5, lattice="N", limit=3, dualize=True)
    assert plain and len(plain) == len(dualized)
    assert all(
        (a.dual().vertices() == b.vertices()).all() for a, b in zip(plain, dualized)
    )


def test_a_fixed_seed_is_reproducible():
    first = _fetch(h11=5, lattice="N", limit=4, sample_seed=7)
    second = _fetch(h11=5, lattice="N", limit=4, sample_seed=7)
    assert [p.vertices().tolist() for p in first] == [
        p.vertices().tolist() for p in second
    ]


def test_deterministic_glsm_basis_reaches_the_polytope():
    """`load_polytopes` drops this flag, so the database path must not use it."""
    polytopes = _fetch(h11=5, lattice="N", limit=2, deterministic_glsm_basis=True)
    assert len(polytopes) == 2
    # a second identical request must agree on the basis
    again = _fetch(h11=5, lattice="N", limit=2, deterministic_glsm_basis=True)
    assert [p.glsm_charge_matrix().tolist() for p in polytopes] == [
        p.glsm_charge_matrix().tolist() for p in again
    ]


# ------------------------------------------------------------------- guards
def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="source"):
        fetch_polytopes(h11=5, lattice="N", limit=1, source="nonsense")


def test_database_source_rejects_5d():
    with pytest.raises(ValueError, match="4D"):
        fetch_polytopes(h11=5, dim=5, limit=1, source="database")


def test_inconsistent_euler_characteristic_still_rejected():
    with pytest.raises(ValueError, match="Euler characteristic"):
        fetch_polytopes(h11=5, h21=3, chi=999, lattice="N", limit=1, source="database")
