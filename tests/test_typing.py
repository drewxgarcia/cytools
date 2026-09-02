"""Public choice types stay aligned with the runtime contracts they describe."""

from inspect import Parameter, signature
from typing import get_args, get_type_hints

from cytools._typing import (
    AutomorphismAction,
    ExtremalityMethod,
    ExtremalRaysMethod,
    FaceTriangulationMethod,
    InteriorPointBackend,
    IntnumFormat,
    InvariantFormat,
    InvariantKind,
    Lattice,
    LinearSolverBackend,
    NormalFormBackend,
    PointednessBackend,
    PolytopeBackend,
    PolytopeFormat,
    PolytopeInputType,
    PolytopeSource,
    ProcessStartMethod,
    RandomTriangulationBackend,
    SecondaryConeBackend,
    StretchedConeBackend,
    TriangulationBackend,
)
from cytools.calabiyau import CalabiYau, configure_gv_subprocess
from cytools.cone import Cone, is_extremal
from cytools.f_theory.Uplift_functions import (
    find_trilayer_vertex_polytope,
    find_trilayer_vertex_vertices,
)
from cytools.ntfe.face_triangulations import face_triangs
from cytools.ntfe.ntfe import triangface_ineqs
from cytools.polytope import Polytope
from cytools.polytopeface import PolytopeFace
from cytools.toricvariety import ToricVariety
from cytools.triangulation import Triangulation
from cytools.utils import fetch_polytopes, read_polytopes, solve_linear_system
from cytools.vector_config.fan import Fan


def test_choice_aliases_are_exact():
    assert get_args(Lattice) == ("N", "M")
    assert get_args(LinearSolverBackend) == ("all", "sksparse", "scipy")
    assert get_args(IntnumFormat) == ("dok", "coo", "dense")
    assert get_args(PolytopeSource) == ("auto", "database", "web")
    assert get_args(PolytopeFormat) == ("ks", "ws")
    assert get_args(PolytopeInputType) == ("file", "str")
    assert get_args(PolytopeBackend) == ("ppl", "qhull", "palp")
    assert get_args(AutomorphismAction) == ("left", "right")
    assert get_args(NormalFormBackend) == ("native", "palp")
    assert get_args(TriangulationBackend) == ("cgal", "qhull", "topcom")
    assert get_args(RandomTriangulationBackend) == ("cgal", "qhull")
    assert get_args(SecondaryConeBackend) == ("native", "topcom")
    assert get_args(ExtremalRaysMethod) == (
        "extremalrays",
        "legacy",
        "lp",
        "nnls",
    )
    assert get_args(ExtremalityMethod) == ("lp", "nnls")
    assert get_args(PointednessBackend) == ("dual", "null", "lp", "nnls")
    assert get_args(StretchedConeBackend) == (
        "mosek",
        "osqp",
        "cvxopt",
        "highs",
        "glop",
    )
    assert get_args(InteriorPointBackend) == (
        "highs",
        "glop",
        "scip",
        "cpsat",
        "mosek",
        "osqp",
        "cvxopt",
    )
    assert get_args(FaceTriangulationMethod) == (
        "fast",
        "fair",
        "grow2d",
        "dualgnn",
    )
    assert get_args(InvariantFormat) == ("dok", "coo")
    assert get_args(InvariantKind) == ("gv", "gw")
    assert get_args(ProcessStartMethod) == ("fork", "forkserver", "spawn")


def test_notebook_facing_annotations_use_the_shared_aliases():
    assert fetch_polytopes.__annotations__["lattice"] is Lattice
    assert fetch_polytopes.__annotations__["source"] is PolytopeSource
    assert read_polytopes.__annotations__["input_type"] is PolytopeInputType
    assert read_polytopes.__annotations__["format"] is PolytopeFormat
    assert solve_linear_system.__annotations__["backend"] is LinearSolverBackend


def test_domain_annotations_use_operation_specific_aliases():
    namespace = {
        "CalabiYau": CalabiYau,
        "Cone": Cone,
        "Polytope": Polytope,
        "PolytopeFace": PolytopeFace,
        "ToricVariety": ToricVariety,
        "Triangulation": Triangulation,
    }

    def choice(function, parameter):
        return get_type_hints(function, globalns=function.__globals__ | namespace)[
            parameter
        ]

    assert choice(Polytope.__init__, "backend") == PolytopeBackend | None
    assert choice(Polytope.automorphisms, "action") is AutomorphismAction
    assert choice(Polytope.normal_form, "backend") is NormalFormBackend
    assert choice(Polytope.triangulate, "backend") is TriangulationBackend
    assert (
        choice(Polytope.random_triangulations_fast, "backend")
        is RandomTriangulationBackend
    )
    assert choice(Polytope.hpq, "lattice") is Lattice
    assert choice(PolytopeFace.triangulate, "backend") is TriangulationBackend
    assert choice(Triangulation.__init__, "backend") is TriangulationBackend
    assert (
        choice(Triangulation.secondary_cone, "backend") == SecondaryConeBackend | None
    )
    assert choice(Cone.extremal_rays, "method") is ExtremalRaysMethod
    assert choice(Cone.is_pointed, "backend") is PointednessBackend
    assert choice(Cone.tip_of_stretched_cone, "backend") == (
        StretchedConeBackend | None
    )
    assert choice(Cone.find_interior_point, "backend") == (InteriorPointBackend | None)
    assert choice(is_extremal, "method") is ExtremalityMethod
    assert choice(ToricVariety.intersection_numbers, "format") is IntnumFormat
    assert choice(ToricVariety.intersection_numbers, "backend") is LinearSolverBackend
    assert choice(CalabiYau.intersection_numbers, "format") is IntnumFormat
    assert choice(CalabiYau.compute_gvs, "format") == InvariantFormat | None
    assert choice(configure_gv_subprocess, "method") is ProcessStartMethod
    assert choice(face_triangs, "triang_method") is FaceTriangulationMethod
    assert choice(triangface_ineqs, "triang_method") is FaceTriangulationMethod


def test_index_flags_have_one_canonical_spelling():
    migrated = {
        Triangulation.points: "as_triang_indices",
        Fan.cones: "as_inds",
        Fan.restricted_simps: "as_face_inds",
        find_trilayer_vertex_polytope: "as_index",
        find_trilayer_vertex_vertices: "as_vertex_index",
    }

    for function, legacy_name in migrated.items():
        parameters = signature(function).parameters
        assert "as_indices" in parameters
        if function is Fan.cones:
            # regfans' virtual method contract fixes the legacy positional
            # slot; the canonical spelling is therefore the keyword-only one.
            assert parameters["as_indices"].kind is Parameter.KEYWORD_ONLY
        else:
            assert parameters[legacy_name].kind is Parameter.KEYWORD_ONLY
