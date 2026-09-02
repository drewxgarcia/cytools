"""Contracts for guarantee-driven computational engine selection."""

import pickle

import numpy as np
import pytest

from cytools import Polytope, config
from cytools._backends.hull import ppl_hull, qhull_hull
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
        with pytest.raises(EngineUnavailable, match="does not apply"):
            registry.resolve(need=(EXACT,), problem={"size": 3})


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

    # The established public selector is itself explicit and remains a
    # drop-in compatibility path.
    assert Polytope(points, backend="qhull").vertices().shape == (3, 2)


def test_qhull_reconstructs_lattice_facets_without_truncating_normals():
    points = np.array([[1, 0], [0, 1], [-1, -1], [0, 0]])
    approximate, _ = qhull_hull(points)
    exact, _ = ppl_hull(points)

    assert {tuple(row) for row in approximate} == {tuple(row) for row in exact}
