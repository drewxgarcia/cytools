"""Mechanical checks for the package boundaries in ARCHITECTURE.md."""

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "src" / "cytools"
PUBLIC_FACADE = PACKAGE / "__init__.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def test_internal_modules_do_not_import_from_public_facade():
    violations = []
    for path in PACKAGE.rglob("*.py"):
        if path == PUBLIC_FACADE:
            continue
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.ImportFrom) and node.module == "cytools":
                violations.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
            elif isinstance(node, ast.Import):
                if any(alias.name == "cytools" for alias in node.names):
                    violations.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
    assert not violations, "internal imports through public facade: " + ", ".join(
        violations
    )


def test_backend_modules_do_not_import_domain_objects():
    violations = []
    backend_dir = PACKAGE / "_backends"
    for path in backend_dir.glob("*.py"):
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if node.module.startswith("cytools.") and not node.module.startswith(
                ("cytools._backends", "cytools._typing")
            ):
                violations.append(f"{path.name}:{node.lineno}:{node.module}")
    assert not violations, "backend imports domain code: " + ", ".join(violations)


def test_optional_engines_are_not_imported_when_adapter_modules_load():
    forbidden = {"PyNormaliz", "extremalrays"}
    violations = []
    for path in (PACKAGE / "_backends").glob("*.py"):
        for node in _tree(path).body:
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name.split(".", 1)[0] in forbidden for name in names):
                violations.append(f"{path.name}:{node.lineno}")
    assert not violations, "optional engine imported eagerly: " + ", ".join(violations)


def test_ppl_is_imported_only_through_its_compatibility_boundary():
    boundary = PACKAGE / "_backends" / "ppl.py"
    violations = []
    for path in PACKAGE.rglob("*.py"):
        if path == boundary:
            continue
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Import) and any(
                alias.name == "ppl" for alias in node.names
            ):
                violations.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and node.module == "ppl":
                violations.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
    assert not violations, "PPL imported outside compatibility boundary: " + ", ".join(
        violations
    )


def test_feature_modules_do_not_mutate_domain_classes():
    domain_classes = {"Cone", "Polytope", "PolytopeFace", "Triangulation"}
    violations = []

    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(_tree(path)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]

            violations.extend(
                f"{path.relative_to(PACKAGE)}:{target.lineno}"
                for target in targets
                if isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in domain_classes
            )

            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "setattr"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in domain_classes
            ):
                violations.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")

    assert not violations, "runtime domain-class mutation: " + ", ".join(violations)


def test_package_import_keeps_numerical_and_domain_modules_lazy():
    code = """
import sys
import cytools

heavy_roots = {
    "cvxopt", "cygv", "flint", "highspy", "latticepts", "numba",
    "numpy", "ortools", "pandas", "ppl", "pyarrow", "pypalp",
    "qpsolvers", "regfans", "requests", "scipy", "sympy",
    "triangulumancer",
}
assert heavy_roots.isdisjoint(sys.modules)
assert {
    name for name in sys.modules
    if name == "cytools" or name.startswith("cytools.")
} <= {"cytools", "cytools._version"}

assert "cytools.ntfe" not in sys.modules
assert "cytools.ntfe.face_triangulations" not in sys.modules
assert "cytools.vector_config" not in sys.modules

Polytope = cytools.Polytope
assert "cytools.triangulation" not in sys.modules
assert "cytools.polytopeface" not in sys.modules
assert "cytools.cone" not in sys.modules
assert "cytools.toricvariety" not in sys.modules
assert "cytools.calabiyau" not in sys.modules

assert "ntfe_frts" in cytools.Polytope.__dict__
assert "vc" in cytools.Polytope.__dict__

poly = Polytope([[1, 0], [0, 1], [-1, -1]])
assert callable(poly.face_triangs)
assert "cytools.ntfe.face_triangulations" in sys.modules
assert "cytools.vector_config" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_domain_modules_do_not_eagerly_import_their_peers():
    code = """
import sys

from cytools.triangulation import Triangulation

assert Triangulation.__name__ == "Triangulation"
assert "cytools.cone" not in sys.modules
assert "cytools.polytopeface" not in sys.modules
assert "cytools.toricvariety" not in sys.modules
assert "cytools.calabiyau" not in sys.modules

from cytools.toricvariety import ToricVariety

assert ToricVariety.__name__ == "ToricVariety"
assert "cytools.cone" not in sys.modules
assert "cytools.calabiyau" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_every_lazy_method_target_resolves_to_a_callable():
    from cytools.cone import Cone
    from cytools.polytope import Polytope
    from cytools.polytopeface import PolytopeFace
    from cytools.triangulation import Triangulation

    methods = {
        Cone: ("vc",),
        Polytope: (
            "get_bdry",
            "face_triangs",
            "n_2face_triangs",
            "num_2face_triangs",
            "grow_ft",
            "grow_frt",
            "expanded_secondary_fan",
            "triangfaces_to_frt",
            "triangfaces_to_frst",
            "triangface_ineqs",
            "ntfe_hypers",
            "ntfe_cones",
            "ntfe_frts",
            "ntfe_frsts",
            "vc",
        ),
        PolytopeFace: ("_2d_frt_subfan_ineqs",),
        Triangulation: (
            "_2d_frt_cone_ineqs",
            "_2d_s_cone_ineqs",
            "vc",
            "fan",
        ),
    }

    for owner, names in methods.items():
        for name in names:
            assert callable(getattr(owner, name)), (
                f"cannot resolve {owner.__name__}.{name}"
            )


def test_lazy_method_resolves_its_target_once(monkeypatch):
    """Warm calls must not repeat module import and target validation."""
    import types

    import cytools._extensions as extensions

    calls = []

    def implementation(self, value):
        return self.offset + value

    def fake_import(name):
        calls.append(name)
        return types.SimpleNamespace(implementation=implementation)

    monkeypatch.setattr(extensions, "import_module", fake_import)

    class Example:
        method = extensions.LazyMethod("example.feature", "implementation")

        def __init__(self, offset):
            self.offset = offset

    assert Example(2).method(3) == 5
    assert Example(10).method(4) == 14
    assert calls == ["example.feature"]
