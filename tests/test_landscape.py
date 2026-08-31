"""Tests for the column-oriented landscape scan API.

The tests that need real Kreuzer-Skarke data are skipped unless a local
database is configured (`CYTOOLS_DB_DIR`). The rest -- registry behaviour,
laziness, filter translation, guards -- run anywhere.
"""

import numpy as np
import pytest

from cytools import Geometry, quantities, quantity, scan, sweep
from cytools.landscape import _QUANTITIES, _scan_kwargs, _store_key
from cytools.store import Unsupported

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def derived(tmp_path, monkeypatch):
    """An isolated store, so tests never reuse each other's cached results."""
    monkeypatch.setenv("CYTOOLS_DERIVED_DIR", str(tmp_path / "derived"))
    return tmp_path / "derived"


def _vertices(n=4, n_vertices=None):
    """Real KS vertices, or skip."""
    ds = pytest.importorskip("cytools.dataset")
    try:
        batches = ds.scan_batches(n_vertices=n_vertices or [13, 14], n=n, batch_size=n)
        out = []
        for b in batches:
            out += [b.vertices(i) for i in range(len(b))]
        if not out:
            pytest.skip("no rows returned from the local KS database")
        return out
    except (ImportError, ValueError) as e:
        pytest.skip(f"no local KS database configured: {e}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_builtin_quantities_are_registered():
    names = set(_QUANTITIES)
    for expected in (
        "h11",
        "h21",
        "chi",
        "is_favorable",
        "n_intnums",
        "divisor_volumes",
        "kahler_point",
        "tip",
        "tip_backend",
    ):
        assert expected in names


def test_quantities_lists_source_and_description():
    df = quantities()
    assert list(df.columns) == ["name", "source", "parallel_safe", "description"]
    assert set(df["source"]) <= {"database", "computed"}
    row = df[df["name"] == "h11"].iloc[0]
    assert row["source"] == "database"
    assert row["description"]
    # Every built-in lives in cytools.landscape, so all are parallel safe.
    assert df["parallel_safe"].all()


def test_unknown_column_names_the_alternatives():
    with pytest.raises(ValueError, match="unknown column"):
        scan(["not_a_real_column"], n=1)


def test_no_columns_is_an_error():
    with pytest.raises(ValueError, match="no columns"):
        scan([], n=1)


def test_custom_quantity_registers_and_is_requestable():
    @quantity(name="_test_double_h11")
    def _double(g):
        """Twice h11, for testing."""
        return 2 * g.h11

    try:
        assert "_test_double_h11" in _QUANTITIES
        assert "_test_double_h11" in set(quantities()["name"])
    finally:
        del _QUANTITIES["_test_double_h11"]


def test_store_key_is_readable_and_order_independent():
    assert _store_key(["h11", "chi"]) == _store_key(["chi", "h11"]) == "chi-h11"


def test_store_key_stays_bounded_for_many_columns():
    key = _store_key([f"col_{i}" for i in range(40)])
    assert len(key) <= 80


# ---------------------------------------------------------------------------
# Filter translation: the public surface is N-lattice, the database is M
# ---------------------------------------------------------------------------


def test_n_lattice_filters_map_onto_the_m_lattice_columns():
    kw = _scan_kwargs(
        n=5, batch_size=8, db_dir=None, filters={"h11": 50, "h21": 3, "chi": -100}
    )
    assert kw["h12"] == 50  # N-lattice h11 is the database's h12
    assert kw["h11"] == 3  # N-lattice h21 is the database's h11
    assert kw["chi"] == 100  # N-lattice chi is the negated database chi


def test_iterable_filters_are_translated_elementwise():
    kw = _scan_kwargs(
        n=None,
        batch_size=8,
        db_dir=None,
        filters={"h11": range(50, 53), "chi": [-4, -6]},
    )
    assert list(kw["h12"]) == [50, 51, 52]
    assert kw["chi"] == [4, 6]


def test_unexpected_filter_is_rejected():
    with pytest.raises(TypeError, match="unexpected filter"):
        _scan_kwargs(n=1, batch_size=8, db_dir=None, filters={"h13": 4})


def test_unknown_moduli_mode_is_rejected_before_database_access(monkeypatch):
    import cytools.landscape as lm

    def fail_if_accessed(**_):
        raise AssertionError("database was accessed")

    monkeypatch.setattr(lm, "scan_batches", fail_if_accessed)
    with pytest.raises(ValueError, match="moduli must be"):
        scan(["h11"], n=1, moduli="boundary")


# ---------------------------------------------------------------------------
# Laziness: the point of the design
# ---------------------------------------------------------------------------


def test_geometry_memoizes_the_triangulation(monkeypatch):
    verts = _vertices(1)[0]
    from cytools import Polytope

    calls = []
    real = Polytope.triangulate

    def counting(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(Polytope, "triangulate", counting)

    g = Geometry(verts)
    g.triangulation
    g.triangulation
    g.toric_variety
    assert len(calls) == 1, "triangulation was rebuilt"


def test_database_columns_never_build_a_polytope(derived, monkeypatch):
    _vertices(1)  # skip early if there is no database
    import cytools.landscape as lm

    built = []
    real = lm.Geometry.__init__

    def counting(self, vertices):
        built.append(1)
        return real(self, vertices)

    monkeypatch.setattr(lm.Geometry, "__init__", counting)

    df = scan(["h11", "chi", "n_vertices"], n=50, n_vertices=[13], progress=False)
    assert len(df) == 50
    assert built == [], "a Geometry was constructed for a pure database read"


def test_a_non_favorable_polytope_is_not_triangulated_for_cy_columns(monkeypatch):
    """The whole favorability optimisation, stated as a property."""
    from cytools import Polytope

    verts = None
    for v in _vertices(60):
        if not Polytope(v).is_favorable(lattice="N"):
            verts = v
            break
    if verts is None:
        pytest.skip("no non-favorable polytope in the sample")

    calls = []
    real = Polytope.triangulate

    def counting(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(Polytope, "triangulate", counting)

    g = Geometry(verts)
    with pytest.raises(Unsupported):
        g.cy
    assert calls == [], "triangulated a polytope that cannot carry a CY"


# ---------------------------------------------------------------------------
# Scan semantics
# ---------------------------------------------------------------------------


def test_scan_computes_and_matches_direct_cytools_calls(derived):
    """Differential check: the library route must equal the manual route."""
    verts = _vertices(40, n_vertices=[14])
    from cytools import Polytope

    df = scan(
        ["is_favorable", "n_points", "n_simplices"],
        n=40,
        n_vertices=[14],
        workers=1,
        progress=False,
    )
    assert len(df) == 40

    # Recompute a few rows by hand and compare.
    checked = 0
    for v in verts[:6]:
        p = Polytope(v)
        expected_fav = bool(p.is_favorable(lattice="N"))
        expected_pts = len(p.points())
        g = Geometry(v)
        assert g.is_favorable == expected_fav
        assert g.n_points == expected_pts
        checked += 1
    assert checked == 6


def test_unsupported_rows_keep_the_columns_that_did_not_need_a_cy(derived):
    _vertices(1)
    df = scan(
        ["is_favorable", "n_points", "n_intnums", "n_cy_intnums"],
        n=120,
        n_vertices=[13],
        workers=1,
        progress=False,
    )
    if "unsupported" not in df:
        pytest.skip("no non-favorable geometry in the sample")

    bad = df[df["unsupported"].notna()]
    assert len(bad) > 0
    # These need only the polytope or the ambient variety, so they must
    # survive the skip rather than be discarded with it.
    assert bad["is_favorable"].notna().all()
    assert (bad["is_favorable"] == False).all()  # noqa: E712
    assert bad["n_points"].notna().all()
    assert bad["n_intnums"].notna().all()
    # This one genuinely needs the threefold, so it must be absent.
    assert bad["n_cy_intnums"].isna().all()


def test_ambient_and_cy_intersection_numbers_are_different_quantities(derived):
    """A conflation that has already bitten once, pinned."""
    _vertices(1)
    df = scan(
        ["is_favorable", "n_intnums", "n_cy_intnums"],
        n=120,
        n_vertices=[13],
        workers=1,
        progress=False,
    )
    both = df[df["n_cy_intnums"].notna()]
    if both.empty:
        pytest.skip("no favorable geometry in the sample")
    # The threefold carries strictly fewer than the ambient fourfold.
    assert (both["n_cy_intnums"] < both["n_intnums"]).all()


def test_scan_is_resumable_and_returns_the_same_rows(derived):
    _vertices(1)
    first = scan(
        ["is_favorable", "n_points"],
        n=60,
        n_vertices=[13],
        workers=1,
        progress=False,
    )
    second = scan(
        ["is_favorable", "n_points"],
        n=60,
        n_vertices=[13],
        workers=1,
        progress=False,
    )

    a = first.sort_values("ks_id").reset_index(drop=True)
    b = second.sort_values("ks_id").reset_index(drop=True)
    assert a["ks_id"].equals(b["ks_id"])
    assert a["n_points"].equals(b["n_points"])


def test_sweep_returns_counts_and_does_not_collect(derived):
    _vertices(1)
    out = sweep(
        ["is_favorable", "n_points"], n=40, n_vertices=[13], workers=1, progress=False
    )
    assert isinstance(out, dict)
    assert set(out) == {"requested", "computed", "skipped", "unsupported", "failed"}
    assert out["requested"] == 40


def test_scan_refuses_to_collect_an_unbounded_result(derived):
    _vertices(1)
    with pytest.raises(ValueError, match="cytools.sweep"):
        scan(["h11"], n=5000, n_vertices=[13], max_rows=100, progress=False)


def test_user_defined_quantities_fall_back_to_one_worker():
    """A spawned worker imports `cytools`, and nothing else.

    Registration is a side effect of importing the module that holds the
    `@quantity` call, so a column defined anywhere but `cytools.landscape` is
    simply absent from a worker's registry -- whether it came from a notebook
    or from an installed package. Running in-process is the only correct
    choice, and it must be taken before a `BrokenProcessPool` happens.
    """
    from cytools.landscape import _resolve_workers

    # Built-in: resolvable in a worker, so an explicit worker count stands.
    assert _resolve_workers(8, ["is_favorable"]) == 8

    @quantity(name="_test_local_col")
    def _local(g):
        """Defined in this test module."""
        return 1

    try:
        with pytest.warns(RuntimeWarning, match="run in the notebook process"):
            assert _resolve_workers(8, ["_test_local_col"]) == 1
        # Silent when no parallelism was asked for.
        assert _resolve_workers(None, ["_test_local_col"]) == 1
    finally:
        del _QUANTITIES["_test_local_col"]


def test_quantities_reports_parallel_safety_consistently():
    """The advertised flag must match what `_resolve_workers` actually does."""
    from cytools.landscape import _resolve_workers

    df = quantities()
    for _, row in df.iterrows():
        expected = 4 if row["parallel_safe"] else 1
        assert _resolve_workers(4, [row["name"]]) == expected, row["name"]


def test_auto_worker_count_matches_the_payload(monkeypatch):
    """Measured payload classes get distinct caps; explicit choices still win."""
    import cytools.landscape as landscape_module
    from cytools.landscape import _resolve_workers

    monkeypatch.setattr(landscape_module.os, "cpu_count", lambda: 12)

    assert _resolve_workers(None, ["n_intnums"]) == 4
    assert _resolve_workers(None, ["divisor_volumes"]) == 1
    assert _resolve_workers(None, ["cy_volume"]) == 1
    assert _resolve_workers(None, ["n_intnums", "divisor_volumes"]) == 1

    # `workers=` is an expert override, including for BLAS-heavy payloads.
    assert _resolve_workers(6, ["divisor_volumes"]) == 6


def test_store_defaults_to_a_cache_dir_when_unconfigured(monkeypatch):
    """Nothing should need configuring before a result can be computed."""
    monkeypatch.delenv("CYTOOLS_DERIVED_DIR", raising=False)
    from cytools.store import DerivedStore

    store = DerivedStore()
    assert store.root is not None
    assert "cytools" in str(store.root).lower()


# ---------------------------------------------------------------------------
# Synthetic notebook workflows: no database installation required
# ---------------------------------------------------------------------------


class _FakeBatch:
    def __init__(self):
        self.ks_ids = np.asarray([31, 7], dtype=np.int64)
        self.vertex_count = np.asarray([5, 6], dtype=np.int64)
        self.facet_count = np.asarray([5, 8], dtype=np.int64)
        self.point_count = np.asarray([6, 14], dtype=np.int64)
        self.dual_point_count = np.asarray([126, 22], dtype=np.int64)
        self.h11 = np.asarray([101, 11], dtype=np.int64)
        self.h12 = np.asarray([3, 9], dtype=np.int64)
        self.euler_characteristic = np.asarray([196, 4], dtype=np.int64)
        self._vertices = [
            np.full((5, 4), 31, dtype=np.int32),
            np.full((6, 4), 7, dtype=np.int32),
        ]

    def __len__(self):
        return len(self.ks_ids)

    def vertices(self, i):
        return self._vertices[i]


def _synthetic_scan(calls):
    def scan_batches(**kwargs):
        calls.append(kwargs)
        yield _FakeBatch()

    return scan_batches


def test_database_only_scan_maps_filters_and_preserves_order(monkeypatch):
    import cytools.landscape as lm

    calls = []
    monkeypatch.setattr(lm, "scan_batches", _synthetic_scan(calls))
    df = scan(
        ["h11", "h21", "chi", "n_vertices", "n_points", "n_facets"],
        n=2,
        h11=range(3, 10),
        h21=101,
        chi=[-196, -4],
        progress=False,
    )

    assert df["ks_id"].tolist() == [31, 7]
    assert df["h11"].tolist() == [3, 9]
    assert df["h21"].tolist() == [101, 11]
    assert df["chi"].tolist() == [-196, -4]
    assert df["n_points"].tolist() == [6, 14]
    assert df["n_facets"].tolist() == [5, 8]
    assert calls[0]["h12"] == range(3, 10)
    assert calls[0]["h11"] == 101
    assert calls[0]["chi"] == [196, 4]
    assert df.attrs["cytools"]["requested"] == 2


def test_notebook_quantity_is_cached_versioned_and_ordered(derived, monkeypatch):
    import cytools.landscape as lm

    scan_calls, quantity_calls = [], []
    monkeypatch.setattr(lm, "scan_batches", _synthetic_scan(scan_calls))

    @quantity(name="_test_notebook_marker")
    def marker(g):
        """First synthetic coordinate."""
        quantity_calls.append(1)
        return int(g._vertices[0, 0])

    try:
        first = scan(
            ["h11", "_test_notebook_marker"],
            n=2,
            workers=1,
            progress=False,
        )
        cached = scan(["_test_notebook_marker"], n=2, workers=1, progress=False)
        new_version = scan(
            ["_test_notebook_marker"],
            n=2,
            workers=1,
            version=2,
            progress=False,
        )

        assert first["ks_id"].tolist() == [31, 7]
        assert first["_test_notebook_marker"].tolist() == [31, 7]
        assert cached["ks_id"].tolist() == [31, 7]
        assert cached.attrs["cytools"]["computed"] == 0
        assert cached.attrs["cytools"]["skipped"] == 2
        assert new_version.attrs["cytools"]["computed"] == 2
        assert len(quantity_calls) == 4
    finally:
        del _QUANTITIES["_test_notebook_marker"]


def test_sweep_does_not_build_result_frames(derived, monkeypatch):
    import cytools.landscape as lm

    monkeypatch.setattr(lm, "scan_batches", _synthetic_scan([]))

    def should_not_collect(*args, **kwargs):
        raise AssertionError("sweep collected database columns")

    monkeypatch.setattr(lm, "_db_columns", should_not_collect)

    assert sweep(["h11"], n=2, progress=False)["requested"] == 2

    @quantity(name="_test_sweep_marker")
    def marker(g):
        """Synthetic computed value."""
        return int(g._vertices[0, 0])

    try:
        summary = sweep(["_test_sweep_marker"], n=2, workers=1, progress=False)
        assert summary["requested"] == 2
        assert summary["computed"] == 2
    finally:
        del _QUANTITIES["_test_sweep_marker"]


def test_payload_keeps_successful_columns_when_one_is_unsupported():
    from cytools.landscape import _Payload

    @quantity(name="_test_supported")
    def supported(g):
        """Always works."""
        return 42

    @quantity(name="_test_unsupported")
    def unsupported(g):
        """Never applies."""
        raise Unsupported("not defined for this geometry")

    try:
        out = _Payload(["_test_supported", "_test_unsupported"])(np.eye(4))
        assert out["_test_supported"] == 42
        assert "_test_unsupported" not in out
        assert out["unsupported"] == "not defined for this geometry"
    finally:
        del _QUANTITIES["_test_supported"]
        del _QUANTITIES["_test_unsupported"]


def test_sampled_moduli_seed_is_stable_and_distinguishes_triangulations():
    from cytools.landscape import _Payload

    vertices = np.arange(20, dtype=np.int32).reshape(5, 4)

    @quantity(name="_test_moduli_seed")
    def moduli_seed(g):
        """The internal seed, exposed only for this test."""
        return g._moduli_seed

    try:
        payload = _Payload(["_test_moduli_seed"], moduli="sampled")
        canonical = payload(vertices)["_test_moduli_seed"]
        assert payload(vertices)["_test_moduli_seed"] == canonical
        assert payload(vertices.astype(np.int64))["_test_moduli_seed"] == canonical
        assert payload((vertices, 7))["_test_moduli_seed"] != canonical
        assert _Payload(["_test_moduli_seed"])(vertices)["_test_moduli_seed"] is None
    finally:
        del _QUANTITIES["_test_moduli_seed"]


def test_sampled_kahler_direction_is_reproducible_and_interior():
    from cytools.landscape import _sample_kahler_direction

    class PositiveOrthant:
        @staticmethod
        def hyperplanes():
            return np.eye(3)

        @staticmethod
        def find_grading_vector():
            return np.ones(3)

        @staticmethod
        def tip_of_stretched_cone(*_, **__):
            return np.ones(3)

    cone = PositiveOrthant()
    first = _sample_kahler_direction(cone, 41)
    again = _sample_kahler_direction(cone, 41)
    other = _sample_kahler_direction(cone, 42)

    assert np.allclose(first, again)
    assert not np.allclose(first, other)
    assert np.all(first > 0)
    assert np.isclose(first.min(), 1)


def test_sampled_kahler_direction_handles_missing_grading_vector():
    from cytools.landscape import _sample_kahler_direction

    class UngradedCone:
        @staticmethod
        def hyperplanes():
            return np.eye(2)

        @staticmethod
        def find_grading_vector():
            return None

    assert _sample_kahler_direction(UngradedCone(), 1, start=np.ones(2)) is None


def test_moduli_modes_use_separate_caches(derived, monkeypatch):
    import cytools.landscape as lm
    from cytools.store import DerivedStore

    monkeypatch.setattr(lm, "scan_batches", _synthetic_scan([]))

    @quantity(name="_test_moduli_cache")
    def moduli_seed(g):
        """The selected mode's deterministic seed."""
        return g._moduli_seed

    try:
        tip_result = scan(
            ["_test_moduli_cache"], n=2, workers=1, progress=False, moduli="tip"
        )
        sampled_result = scan(
            ["_test_moduli_cache"],
            n=2,
            workers=1,
            progress=False,
            moduli="sampled",
        )

        assert tip_result.attrs["cytools"]["moduli"] == "tip"
        assert sampled_result.attrs["cytools"]["moduli"] == "sampled"
        assert set(DerivedStore(derived).quantities()) == {
            "_test_moduli_cache",
            "_test_moduli_cache-sampled",
        }
    finally:
        del _QUANTITIES["_test_moduli_cache"]


def test_quantity_registration_validates_public_schema():
    with pytest.raises(ValueError, match="valid Python identifiers"):
        quantity(name="not-a-column")(lambda g: 1)
    with pytest.raises(ValueError, match="source must"):
        quantity(name="valid_name", source="remote")(lambda g: 1)
    with pytest.raises(ValueError, match="reserved"):
        quantity(name="valid_name", source="database")(lambda g: 1)
    with pytest.raises(ValueError, match="cannot replace built-in"):
        quantity(name="h11")(lambda g: 1)


# ---------------------------------------------------------------------------
# Triangulation multiplicity: many Calabi-Yaus per polytope
# ---------------------------------------------------------------------------


def test_triangulations_must_be_at_least_one(derived):
    with pytest.raises(ValueError, match="at least 1"):
        scan(["n_simplices"], n=1, triangulations=0, progress=False)


def test_triangulations_on_database_only_columns_is_refused(derived):
    """N copies of a row that does not depend on the triangulation is a bug."""
    with pytest.raises(ValueError, match="identical row"):
        scan(["h11", "chi"], n=10, triangulations=5, progress=False)


def test_triangulation_index_zero_is_the_canonical_triangulation():
    """So raising `triangulations` reuses earlier work instead of voiding it."""
    from cytools import Polytope
    from cytools.landscape import _hash_bytes

    verts = _vertices(1)[0]
    canonical = np.asarray(
        Polytope(verts).triangulate(verbosity=0).simplices(), dtype=np.int64
    )
    canonical = canonical[np.lexsort(canonical.T[::-1])]

    g = Geometry(verts)  # no seed == index 0
    assert g.triangulation_hash == _hash_bytes(canonical.tobytes())


def test_the_same_seed_always_gives_the_same_triangulation():
    """The property the store's skip logic depends on."""
    verts = _vertices(1)[0]
    a = Geometry(verts, triangulation_seed=12345).triangulation_hash
    b = Geometry(verts, triangulation_seed=12345).triangulation_hash
    assert a == b


def test_different_seeds_give_different_triangulations():
    verts = _vertices(4, n_vertices=[15])[0]
    hashes = {
        Geometry(verts, triangulation_seed=s).triangulation_hash
        for s in (None, 11, 22, 33)
    }
    assert len(hashes) > 1, "the sampler returned one triangulation every time"


def test_geometry_ids_are_identity_at_index_zero_and_distinct_after():
    """Backward compatibility, and no collisions between triangulations."""
    from cytools.landscape import _mix

    assert _mix(4242, 0) == 4242
    ids = [_mix(4242, k) for k in range(16)]
    assert len(set(ids)) == 16


def test_scan_returns_one_row_per_triangulation_with_provenance(derived):
    _vertices(1, n_vertices=[15])  # skip early if there is no database
    df = scan(
        ["is_favorable", "n_simplices", "triangulation_hash"],
        n=8,
        n_vertices=[15],
        triangulations=3,
        workers=1,
        progress=False,
    )

    assert len(df) == 24, "expected 8 polytopes x 3 triangulations"
    assert {"polytope_id", "triangulation_index"} <= set(df.columns)
    assert sorted(df["triangulation_index"].unique()) == [0, 1, 2]
    assert df["polytope_id"].nunique() == 8
    # The row key must separate triangulations of the same polytope.
    assert df["ks_id"].nunique() == 24
    # At index 0 the row key is the polytope's own id.
    zero = df[df["triangulation_index"] == 0]
    assert (zero["ks_id"] == zero["polytope_id"]).all()


def test_triangulations_of_one_polytope_are_different_geometries(derived):
    """If the simplices differ, the derived quantities must differ too."""
    _vertices(1, n_vertices=[15])  # skip early if there is no database
    df = scan(
        ["is_favorable", "n_simplices", "triangulation_hash"],
        n=10,
        n_vertices=[15],
        triangulations=4,
        workers=1,
        progress=False,
    )
    per = df.groupby("polytope_id")["triangulation_hash"].nunique()
    assert per.max() > 1, "every triangulation of every polytope was identical"


def test_raising_triangulations_reuses_the_earlier_ones(derived):
    """Index 0..k-1 are already stored, so only the new ones are computed."""
    _vertices(1, n_vertices=[15])  # skip early if there is no database
    first = sweep(
        ["n_simplices"],
        n=6,
        n_vertices=[15],
        workers=1,
        progress=False,
        triangulations=2,
    )
    assert first["computed"] == 12

    second = sweep(
        ["n_simplices"],
        n=6,
        n_vertices=[15],
        workers=1,
        progress=False,
        triangulations=4,
    )
    assert second["skipped"] == 12, "did not reuse the first two triangulations"
    assert second["computed"] == 12, "did not compute exactly the two new ones"


# ---------------------------------------------------------------------------
# Volumes are contracted from a native sparse tensor, not a dense (h11)^3 one
# ---------------------------------------------------------------------------


def test_contract_kappa_handles_every_multiplicity_class():
    """The two two-equal key shapes need different ordering sets.

    Keys are sorted, so a repeated pair sits at (0,1) -> (x,x,y) or at (1,2) ->
    (x,y,y). Applying one ordering set to both double-counts one ordering and
    drops another; that produced divisor volumes wrong by 8e6 relative.
    Checked here against a dense contraction built from the same data.
    """
    from cytools.landscape import _PERMS_3, _contract_kappa

    n = 5
    rng = np.random.default_rng(0)
    keys = [(0, 1, 2), (1, 1, 3), (0, 2, 2), (4, 4, 4), (0, 1, 1), (2, 3, 4)]
    vals = rng.normal(size=len(keys))
    idx = np.array(keys, dtype=np.int32)
    val = np.asarray(vals, dtype=np.float64)
    t = rng.normal(size=n) + 3.0

    # dense reference: fill every distinct permutation with the same value
    dense = np.zeros((n, n, n))
    for key, v in zip(keys, vals):
        for p in _PERMS_3:
            dense[key[p[0]], key[p[1]], key[p[2]]] = v
    expected = (np.tensordot(dense, t, axes=([-1], [0])) @ t) / 2

    got = _contract_kappa(idx, val, t, n)
    assert np.allclose(got, expected, rtol=1e-12, atol=1e-12), f"{got} vs {expected}"


def test_volume_columns_match_the_library(derived):
    """Differential: the sparse path must reproduce cytools' dense result."""
    verts = _vertices(40, n_vertices=[13])
    checked = 0
    for v in verts:
        if checked >= 3:
            break
        g = Geometry(v)
        try:
            if not g.is_favorable or g.tip is None:
                continue
        except Exception:
            continue
        ref_dv = np.asarray(g.cy.compute_divisor_volumes(g.moduli_point), dtype=float)
        ref_cv = float(g.cy.compute_cy_volume(g.moduli_point))
        assert np.allclose(g.divisor_volumes, ref_dv, rtol=1e-8, atol=1e-8)
        assert abs(g.cy_volume - ref_cv) <= 1e-8 * max(abs(ref_cv), 1.0)
        checked += 1
    if checked == 0:
        pytest.skip("no favorable geometry with a tip in the sample")


def test_both_volume_columns_share_one_contraction(monkeypatch):
    """cy_volume is tau.t/3, so asking for both must contract only once.

    The sparse route is forced here: it is gated on h11 >= 150 in normal use,
    and the geometries cheap enough to test with sit well below that.
    """
    import cytools.landscape as lm

    monkeypatch.setattr(lm, "_SPARSE_KAPPA_MIN_H11", 0)

    verts = None
    for v in _vertices(40, n_vertices=[13]):
        g = Geometry(v)
        try:
            if g.is_favorable and g.tip is not None:
                verts = v
                break
        except Exception:
            continue
    if verts is None:
        pytest.skip("no favorable geometry with a tip in the sample")

    calls = []
    real = lm._contract_kappa

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(lm, "_contract_kappa", counting)

    g = Geometry(verts)
    g.divisor_volumes
    g.cy_volume
    assert len(calls) == 1, f"contracted {len(calls)} times"
