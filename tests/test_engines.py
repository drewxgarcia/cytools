"""Contracts for guarantee-driven computational engine selection."""

import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from cytools import Polytope, config
from cytools._backends.hull import ppl_hull, qhull_hull
from cytools._backends.lp import _ppl_feasibility, highs_feasibility
from cytools._backends.registry import (
    EXACT,
    RECOVERABLE,
    Engine,
    EngineUnavailable,
    GuaranteeViolation,
    Registry,
    UnknownEngine,
)


def _registry() -> Registry:
    return Registry(
        task="test_task",
        engines=(
            Engine(name="fast", run=lambda: "fast", provides=frozenset()),
            Engine(
                name="safe",
                run=lambda: "safe",
                provides=frozenset({EXACT, RECOVERABLE}),
            ),
            Engine(
                name="large",
                run=lambda: "large",
                provides=frozenset({EXACT, RECOVERABLE}),
                applies=lambda problem: problem.get("size", 0) >= 10,
            ),
        ),
    )


def test_resolution_filters_by_guarantee_before_preference():
    registry = _registry()

    assert registry.resolve().name == "fast"
    assert registry.resolve(need=(EXACT, RECOVERABLE)).name == "safe"


def test_overrides_are_scoped_and_restore_after_errors():
    registry = _registry()
    assert config.engine_overrides() == {}

    with pytest.raises(RuntimeError):
        with config.engines(test_task="safe"):
            assert registry.resolve(need=(EXACT,)).name == "safe"
            raise RuntimeError("body failed")

    assert config.engine_overrides() == {}
    assert registry.resolve().name == "fast"


def test_override_cannot_silently_weaken_a_call_site():
    registry = _registry()

    with config.engines(test_task="fast"):
        with pytest.raises(GuaranteeViolation, match="mathematically wrong"):
            registry.resolve(need=(EXACT,))

    with config.engines(test_task="fast", allow_weaker=True):
        with pytest.warns(RuntimeWarning, match="mathematically wrong"):
            assert registry.resolve(need=(EXACT,)).name == "fast"


def test_forced_engine_still_has_to_apply_to_the_problem():
    registry = _registry()

    with config.engines(test_task="large"):
        with pytest.raises(EngineUnavailable, match="does not support"):
            registry.resolve(need=(EXACT,), problem={"size": 3})


def test_problem_size_changes_preference_not_explicit_applicability():
    registry = Registry(
        task="preferred",
        engines=(
            Engine(
                name="small",
                run=lambda: None,
                preference=lambda problem: 0 if problem["size"] < 10 else 1,
            ),
            Engine(
                name="large",
                run=lambda: None,
                preference=lambda problem: 0 if problem["size"] >= 10 else 1,
            ),
        ),
    )

    assert registry.resolve(problem={"size": 3}).name == "small"
    assert registry.resolve(problem={"size": 30}).name == "large"
    assert registry.select("large", problem={"size": 3}).name == "large"


def test_bad_engine_and_bad_guarantee_are_actionable():
    registry = _registry()

    with config.engines(test_task="missing"):
        with pytest.raises(UnknownEngine, match="Registered"):
            registry.resolve()

    with pytest.raises(ValueError, match="Unknown guarantee"):
        registry.resolve(need=("wishful_thinking",))


def test_public_introspection_reports_every_task():
    available = config.available_engines()
    assert set(available) == {
        "convex_hull",
        "interior_point",
        "stretched_tip",
        "triangulate",
        "linear_solve",
    }
    assert all(isinstance(names, tuple) for names in available.values())


def test_polytope_hull_keeps_engine_objects_out_of_domain_state():
    points = [[1, 0], [0, 1], [-1, -1], [0, 0]]
    polytope = Polytope(points)

    assert "_poly_optimal" not in vars(polytope)
    assert isinstance(polytope._vertices_optimal, np.ndarray)

    restored = pickle.loads(pickle.dumps(polytope))
    assert restored == polytope
    assert restored.vertices().tolist() == polytope.vertices().tolist()


def test_floating_hull_requires_an_explicit_guarantee_downgrade():
    points = [[1, 0], [0, 1], [-1, -1], [0, 0]]

    with config.engines(convex_hull="qhull"):
        with pytest.raises(GuaranteeViolation, match="mathematically wrong"):
            Polytope(points)

    with config.engines(convex_hull="qhull", allow_weaker=True):
        with pytest.warns(RuntimeWarning, match="mathematically wrong"):
            approximate = Polytope(points)
    assert approximate.vertices().shape == (3, 2)

    # The established public selector remains a drop-in compatibility path.
    assert Polytope(points, backend="qhull").vertices().shape == (3, 2)


def test_palp_is_permitted_explicitly_but_warns_that_it_can_abort():
    """EXACT is refused when missing; RECOVERABLE is warned about.

    The split is the point. Dropping exactness changes the answer, so it is
    blocked. Dropping recoverability risks the process but not the
    mathematics, and a caller who named PALP has accepted that trade.
    """
    points = [[1, 0], [0, 1], [-1, -1], [0, 0]]

    with pytest.warns(RuntimeWarning, match="aborts the process"):
        explicit = Polytope(points, backend="palp")

    assert explicit.vertices().shape == (3, 2)
    assert explicit.inequalities().tolist() == Polytope(points).inequalities().tolist()


def test_qhull_reconstructs_lattice_facets_without_truncating_normals():
    points = np.array([[1, 0], [0, 1], [-1, -1], [0, 0]])
    approximate, _ = qhull_hull(points)
    exact, _ = ppl_hull(points)

    assert {tuple(row) for row in approximate} == {tuple(row) for row in exact}


def test_numerical_infeasibility_is_confirmed_exactly(monkeypatch):
    calls = 0

    def exact_check(hyperplanes, c, ambient_dim, lower_bound):
        nonlocal calls
        calls += 1
        return _ppl_feasibility(hyperplanes, c, ambient_dim, lower_bound)

    monkeypatch.setattr("cytools._backends.lp._ppl_feasibility", exact_check)
    result = highs_feasibility([[1, 0], [-1, 0]], 1, 2)

    assert result is None
    assert calls == 1


def test_exact_fallback_stays_off_the_feasible_fast_path(monkeypatch):
    def unexpected_exact_check(*args, **kwargs):
        raise AssertionError("exact fallback ran for a feasible LP")

    monkeypatch.setattr("cytools._backends.lp._ppl_feasibility", unexpected_exact_check)
    result = highs_feasibility([[1, 0], [0, 1]], 1, 2)

    assert result is not None
    assert np.all(np.asarray(result) >= 1)


def test_feasibility_engines_accept_sparse_mapping_rows():
    point = highs_feasibility([{0: 1}, {1: 1}], 1, 2)

    assert point is not None
    assert np.all(np.asarray(point) >= 1)
    assert highs_feasibility([{0: 1}, {0: -1}], 1, 2) is None


@pytest.mark.parametrize("engine_name", ["qhull", "fine"])
def test_triangulation_override_preserves_star_contract(engine_name):
    polytope = Polytope([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1], [0, 0, 0]])

    with config.engines(triangulate=engine_name, allow_weaker=True):
        with pytest.warns(RuntimeWarning, match="mathematically wrong"):
            triangulation = polytope.triangulate(make_star=True)

    assert triangulation.is_star()


def test_high_dimensional_hull_does_not_abort_the_interpreter():
    """The automatic path must not reach an engine that calls abort().

    PALP is a C program with compile-time array bounds (``CEQ_Nmax``,
    ``EQUA_Nmax``) and aborts past them. Reached through a Python extension
    that is SIGABRT: the interpreter dies with no traceback, taking unsaved
    notebook state with it, and no ``except`` can catch it. PALP used to be
    the *default* hull engine above four dimensions, so this configuration --
    9-dimensional, 40 points -- killed ``Polytope()`` outright.

    Run in a subprocess precisely because the failure mode is process death:
    an in-process regression test for this would take the suite down with it
    rather than reporting.
    """
    program = textwrap.dedent("""
        import numpy as np
        from cytools import Polytope

        rng = np.random.default_rng(1)
        dim, target = 9, 40
        seen = {
            tuple(int(x) for x in p)
            for p in np.vstack([np.eye(dim, dtype=int), -np.ones((1, dim), dtype=int)])
        }
        while len(seen) < target:
            seen.add(tuple(int(x) for x in rng.integers(-4, 5, size=dim)))

        print(len(Polytope(np.array(sorted(seen))).inequalities()))
    """)

    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=600
    )

    assert result.returncode == 0, (
        "Polytope() died on a 9-dimensional configuration "
        f"(returncode {result.returncode}); a hull engine aborted the process."
    )
    assert int(result.stdout.strip().splitlines()[-1]) > 0
