"""Mechanical checks for the package boundaries in ARCHITECTURE.md."""

import ast
from pathlib import Path
import subprocess
import sys


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
    assert not violations, (
        "internal imports through public facade: " + ", ".join(violations)
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

            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in domain_classes
                ):
                    violations.append(
                        f"{path.relative_to(PACKAGE)}:{target.lineno}"
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


def test_package_import_keeps_feature_modules_lazy():
    code = """
import sys
import cytools

assert "cytools.ntfe" not in sys.modules
assert "cytools.ntfe.face_triangulations" not in sys.modules
assert "cytools.vector_config" not in sys.modules

assert "ntfe_frts" in cytools.Polytope.__dict__
assert "vc" in cytools.Polytope.__dict__

poly = cytools.Polytope([[1, 0], [0, 1], [-1, -1]])
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
