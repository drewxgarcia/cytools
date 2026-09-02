import builtins

import numpy as np
import pytest

import cytools.cone as cone_module
from cytools import Cone


def _canonical_face_rays(face):
    return tuple(sorted(tuple(ray) for ray in face.extremal_rays().tolist()))


def test_ambient_dimension():
    c = Cone([[0, 1, 0], [1, 1, 0]])
    assert c.ambient_dimension() == 3


def test_dimension():
    c = Cone([[0, 1, 0], [1, 1, 0]])
    assert c.dimension() == 2


def test_dual_cone():
    c = Cone([[0, 1], [1, 1]])
    assert len(c.dual_cone().rays()) == 2


def test_extremal_rays():
    c = Cone([[0, 1], [1, 1], [1, 0]])
    assert len(c.extremal_rays()) == 2


def test_face_lattice_simplicial_4d():
    c = Cone(np.eye(4, dtype=int))

    all_faces = c.face_lattice()
    all_faces_with_self = c.face_lattice(include_self=True)

    assert [len(fs) for fs in all_faces] == [4, 6, 4, 1]
    assert [len(fs) for fs in all_faces_with_self] == [1, 4, 6, 4, 1]
    assert all_faces_with_self[0][0] is c
    assert c.face_lattice(0) == (c,)
    assert c.face_lattice(4)[0].dimension() == 0
    assert all(f.dimension() == 2 for f in c.face_lattice(2))
    assert isinstance(c.facets(), list)
    assert {_canonical_face_rays(f) for f in c.facets()} == {
        _canonical_face_rays(f) for f in c.face_lattice(1)
    }
    assert c.face_lattice(2)[0] is c.face_lattice(include_self=True)[2][0]


def test_face_lattice_nonsimplicial_3d():
    c = Cone([[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]])

    expected_facets = {
        ((-1, 0, 1), (0, -1, 1)),
        ((-1, 0, 1), (0, 1, 1)),
        ((0, -1, 1), (1, 0, 1)),
        ((0, 1, 1), (1, 0, 1)),
    }
    expected_rays = {
        ((-1, 0, 1),),
        ((0, -1, 1),),
        ((0, 1, 1),),
        ((1, 0, 1),),
    }

    assert len(c.face_lattice(1)) == 4
    assert len(c.face_lattice(2)) == 4
    assert {_canonical_face_rays(f) for f in c.face_lattice(1)} == expected_facets
    assert {_canonical_face_rays(f) for f in c.face_lattice(2)} == expected_rays


def test_face_lattice_non_solid_pointed():
    c = Cone([[1, 0, 0], [0, 1, 0]])

    assert c.is_pointed()
    assert not c.is_solid()
    assert len(c.face_lattice()) == 2
    assert len(c.face_lattice(1)) == 2
    assert {_canonical_face_rays(f) for f in c.face_lattice(1)} == {
        ((1, 0, 0),),
        ((0, 1, 0),),
    }
    assert isinstance(c.facets(), list)
    assert {_canonical_face_rays(f) for f in c.facets()} == {
        _canonical_face_rays(f) for f in c.face_lattice(1)
    }


def test_face_lattice_one_dimensional_cone():
    c = Cone([[1, 0]])

    assert c.face_lattice()[-1][0].dimension() == 0
    assert c.face_lattice(include_self=True)[0] == (c,)
    assert c.face_lattice(1)[0].dimension() == 0
    assert c.facets()[0].dimension() == 0


def test_face_lattice_non_pointed_not_implemented():
    c = Cone([[1, 0], [0, 1], [-1, 0]])

    with pytest.raises(NotImplementedError):
        c.face_lattice()


def test_facets_non_pointed_still_supported():
    c = Cone([[1, 0], [0, 1], [-1, 0]])

    facets = c.facets()

    assert len(facets) == 1
    assert facets[0].dimension() == 1
    assert facets[0].contains([1, 0])
    assert facets[0].contains([-1, 0])
    assert not facets[0].contains([0, 1])


def find_interior_point():
    c = Cone([[3, 2], [5, 3]])
    pt = c.find_interior_point()
    assert c.contains(pt)


@pytest.mark.requires_dependency("cvxopt")
def test_cvxopt_backend_when_installed():
    c = Cone([[1, 0], [0, 1]])
    tip = c.tip_of_stretched_cone(backend="cvxopt")
    assert tip is not None
    assert np.allclose(tip, [1.0, 1.0], atol=1e-5)


def test_missing_cvxopt_extra_is_actionable(monkeypatch):
    import qpsolvers

    monkeypatch.setattr(
        qpsolvers,
        "available_solvers",
        [solver for solver in qpsolvers.available_solvers if solver != "cvxopt"],
    )
    with pytest.raises(ImportError, match=r"cytools-workbench\[cvxopt\]"):
        Cone([[1, 0], [0, 1]]).tip_of_stretched_cone(backend="cvxopt")


def test_find_lattice_points():
    c = Cone([[3, 2], [5, 3]])
    pts = c.find_lattice_points(min_points=20)
    assert len(pts) >= 20


def test_find_lattice_points_min_points_exceeds_old_default_coord_bound():
    c = Cone([[1]])
    pts = c.find_lattice_points(min_points=1002, fast_mode=False, deg_window=1000)
    assert len(pts) >= 1002
    assert pts[-1][0] >= 1001


def test_find_lattice_points_finite_coord_bound_exhausted():
    c = Cone([[1]])
    with pytest.raises(ValueError, match="finite max_coord=1"):
        c.find_lattice_points(min_points=3, fast_mode=False, max_coord=1, deg_window=10)


@pytest.mark.requires_dependency("PyNormaliz")
def test_hilbert_basis():
    c = Cone([[1, 3], [2, 1]])
    hb = c.hilbert_basis()
    assert {tuple(row) for row in hb} == {(1, 1), (1, 2), (1, 3), (2, 1)}


def test_hilbert_basis_missing_extra_is_actionable(monkeypatch):
    real_import = builtins.__import__

    def without_pynormaliz(name, *args, **kwargs):
        if name == "PyNormaliz":
            raise ModuleNotFoundError("hidden for test", name="PyNormaliz")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pynormaliz)
    with pytest.raises(ImportError, match=r"cytools-workbench\[normaliz\]"):
        Cone([[1, 3], [2, 1]]).hilbert_basis()


def test_intersection():
    c1 = Cone([[1, 0], [1, 2]])
    c2 = Cone([[0, 1], [2, 1]])
    c3 = c1.intersection(c2)
    assert len(c3.rays()) == 2


def test_intersection_rejects_a_single_cone_in_another_dimension():
    c2 = Cone([[1, 0], [0, 1]])
    c3 = Cone([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="same dimension"):
        c2.intersection(c3)


def test_is_pointed():
    c1 = Cone([[1, 0], [0, 1]])
    c2 = Cone([[1, 0], [0, 1], [-1, 0]])
    assert c1.is_pointed()
    assert not c2.is_pointed()


def test_is_simplicial():
    c1 = Cone([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    c2 = Cone([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, -1]])
    assert c1.is_simplicial()
    assert not c2.is_simplicial()


def test_is_smooth():
    c1 = Cone([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    c2 = Cone([[2, 0, 1], [0, 1, 0], [1, 0, 2]])
    assert c1.is_smooth()
    assert not c2.is_smooth()


def test_is_solid():
    c1 = Cone([[1, 0], [0, 1]])
    c2 = Cone([[1, 0, 0], [0, 1, 0]])
    assert c1.is_solid()
    assert not c2.is_solid()


def test_tip_of_stretched_cone():
    c = Cone([[3, 2], [5, 3]])
    tip_arr = c.tip_of_stretched_cone(1)
    assert tip_arr is not None
    tip = tip_arr.tolist()
    assert np.isclose(tip, [8.0, 5.0]).all()


def test_equality():
    c1 = Cone([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    c2 = Cone([[2, 0, 1], [0, 1, 0], [1, 0, 2]])
    assert c1 == c1
    assert c1 != c2


def test_dimension_is_lazy_not_computed_on_construction():
    """Constructing a Cone from rays must not compute its dimension.

    The rank is an exact integer elimination on the ray matrix -- ~(1500 x 200)
    for a Mori cone at h11 ~ 200 -- and the Kahler-cone/tip path that dominates
    ensemble scans never asks for it. Computing it eagerly in `__init__` cost
    26% of that path. This pins the laziness so it cannot regress.

    The argument got stronger when the rank became exact. Measured on integer
    matrices of the shapes that arise here, `exact_rank` costs about 6x
    `numpy.linalg.matrix_rank` -- 130 ms against 20 ms at (2320 x 300) -- which
    is the price of an answer that is right on ill-conditioned input. Paying it
    once, on demand, is affordable; paying it in every constructor is not.
    """
    calls = []
    real = cone_module.exact_rank

    def spy(M, *a, **k):
        calls.append(np.asarray(M).shape)
        return real(M, *a, **k)

    rays = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cone_module, "exact_rank", spy)
        c = Cone(rays)
        assert calls == [], f"__init__ computed a rank: {calls}"
        assert c.dimension() == 3  # computed on demand
        assert len(calls) == 1
        assert c.dimension() == 3  # and cached thereafter
        assert len(calls) == 1


def test_repr_reports_dimension_when_unset():
    """`__repr__` must go through the accessor, not the raw cached field.

    With a lazily-unset `_dim`, interpolating the field directly renders
    "A None-dimensional ..." on a freshly constructed cone.
    """
    c = Cone([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert "A 3-dimensional" in repr(c)
    assert "None" not in repr(c)


def test_rays_of_hyperplane_cone_leaves_dimension_consistent():
    """The rays() path also defers the rank; dim() must still be the true rank."""
    c = Cone(hyperplanes=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    rays = c.rays()
    assert c.dimension() == np.linalg.matrix_rank(rays)
    assert "None" not in repr(c)
