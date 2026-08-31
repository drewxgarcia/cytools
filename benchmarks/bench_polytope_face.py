"""
Benchmarks for the PolytopeFace class — complete coverage of all 16 public methods.

Methods covered:
  ambient_poly, labels, labels_bdry, labels_int, labels_vertices,
  dimension, ambient_dimension,
  points (default and optimal=True), interior_points, boundary_points,
  vertices, as_polytope, dual_face, faces, triangulate

Fixture design
--------------
``tiny_faces``
    Faces of 20 polytopes from the 5v tier.  Fast calibration.
    All dimension layers (0-d through (d-1)-d faces).

``bulk_faces``
    Faces of 20 polytopes from the bulk (13-17v) tier.  Primary fixture.
    These polytopes have significantly more faces and lattice points.

``reflexive_faces``
    Faces of 20 reflexive polytopes (h11≤4).  Required for dual_face()
    which raises NotImplementedError on non-reflexive polytopes.

Run fast suite:
    pytest benchmarks/bench_polytope_face.py --benchmark-only -m "not slow"

Run full suite:
    pytest benchmarks/bench_polytope_face.py --benchmark-only
"""

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_faces(polys, dims=None):
    """Collect PolytopeFace objects from a list of Polytope objects.

    dims: list of face dimensions to collect (default: all dims > 0)
    Returns a flat list of (polytope, face) pairs.
    """
    results = []
    for p in polys:
        try:
            d = p.dimension()
            face_dims = dims if dims is not None else range(1, d)
            for fd in face_dims:
                for f in p.faces(fd):
                    results.append(f)
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# Module-scope fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_faces(tiny_poly_objects):
    """All faces of 20 tiny (5v) polytopes — fast calibration."""
    return _collect_faces(tiny_poly_objects)


@pytest.fixture(scope="module")
def bulk_faces(bulk_poly_objects):
    """All faces of 20 bulk (13-17v) polytopes — primary fixture.

    These polytopes have many more faces and lattice points than tiny ones.
    """
    return _collect_faces(bulk_poly_objects)


@pytest.fixture(scope="module")
def reflexive_faces(reflexive_poly_objects):
    """All faces of 20 reflexive polytopes — needed for dual_face()."""
    return _collect_faces(reflexive_poly_objects)


@pytest.fixture(scope="module")
def bulk_codim1_faces(bulk_poly_objects):
    """Codim-1 (facet) faces of bulk polytopes.

    Facets are the most expensive face type to work with (most points,
    largest sub-face lattice).  Collect only the facets for focused tests.
    """
    results = []
    for p in bulk_poly_objects:
        try:
            d = p.dimension()
            results.extend(p.faces(d - 1))
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# 1. Trivial properties (O(1) cache hits after first construction)
# ---------------------------------------------------------------------------

class TestTrivialProperties:
    """Properties that are set at construction time or are O(1) lookups."""

    def test_ambient_poly_tiny(self, benchmark, tiny_faces):
        def go():
            return [f.ambient_poly for f in tiny_faces]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_labels_vertices_tiny(self, benchmark, tiny_faces):
        def go():
            return [f.labels_vertices for f in tiny_faces]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_dimension_tiny(self, benchmark, tiny_faces):
        def go():
            return [f.dimension() for f in tiny_faces]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_ambient_dimension_tiny(self, benchmark, tiny_faces):
        def go():
            return [f.ambient_dimension() for f in tiny_faces]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_ambient_poly_bulk(self, benchmark, bulk_faces):
        def go():
            return [f.ambient_poly for f in bulk_faces]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 2. Point labels (trigger _process_points on first call)
# ---------------------------------------------------------------------------

class TestPointLabels:
    """Label access triggers _process_points() on first call.

    These tests measure the cost of classifying all lattice points of a face
    (interior vs boundary) by checking inequality saturation against the
    ambient polytope's inequalities.
    """

    def test_labels_tiny(self, benchmark, tiny_faces):
        def go():
            return [f.labels for f in tiny_faces]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_labels_bdry_tiny(self, benchmark, tiny_faces):
        def go():
            return [f.labels_bdry for f in tiny_faces]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_labels_int_tiny(self, benchmark, tiny_faces):
        def go():
            return [f.labels_int for f in tiny_faces]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_labels_bulk(self, benchmark, bulk_faces):
        def go():
            return [f.labels for f in bulk_faces]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_labels_bdry_bulk(self, benchmark, bulk_faces):
        def go():
            return [f.labels_bdry for f in bulk_faces]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_labels_int_bulk(self, benchmark, bulk_faces):
        def go():
            return [f.labels_int for f in bulk_faces]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 3. Point coordinate access
# ---------------------------------------------------------------------------

class TestPointCoordinates:
    """points(), interior_points(), boundary_points(), vertices().

    points(optimal=True) adds LLL lattice reduction — tracks that overhead.
    """

    def test_points_default_tiny(self, benchmark, tiny_faces):
        def go():
            return [f.points() for f in tiny_faces]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_points_optimal_tiny(self, benchmark, tiny_faces):
        """points(optimal=True) applies LLL reduction — O(n³) per face."""
        def go():
            return [f.points(optimal=True) for f in tiny_faces]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_vertices_tiny(self, benchmark, tiny_faces):
        def go():
            return [f.vertices() for f in tiny_faces]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_interior_points_tiny(self, benchmark, tiny_faces):
        def go():
            return [f.interior_points() for f in tiny_faces]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_boundary_points_tiny(self, benchmark, tiny_faces):
        def go():
            return [f.boundary_points() for f in tiny_faces]
        benchmark.pedantic(go, rounds=5, iterations=1)

    def test_points_default_bulk(self, benchmark, bulk_faces):
        def go():
            return [f.points() for f in bulk_faces]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_points_optimal_bulk(self, benchmark, bulk_faces):
        """LLL reduction on faces of bulk polytopes — higher point counts."""
        def go():
            return [f.points(optimal=True) for f in bulk_faces]
        benchmark.pedantic(go, rounds=1, iterations=1)

    def test_vertices_bulk(self, benchmark, bulk_faces):
        def go():
            return [f.vertices() for f in bulk_faces]
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 4. as_polytope — constructs a new Polytope object per face
# ---------------------------------------------------------------------------

class TestAsPolytope:
    """as_polytope() creates a new Polytope from the face's points.

    This is the most expensive single PolytopeFace operation — it runs the
    full polytope construction (convex hull, inequalities, etc.) for each face.
    Bulk codim-1 faces have the most points and drive the worst case.
    """

    def test_as_polytope_tiny(self, benchmark, tiny_faces):
        def go():
            return [f.as_polytope() for f in tiny_faces]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_as_polytope_bulk_facets(self, benchmark, bulk_codim1_faces):
        """Facets of bulk polytopes — most points per face, most expensive."""
        def go():
            return [f.as_polytope() for f in bulk_codim1_faces]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 5. dual_face — reflexive polytopes only
# ---------------------------------------------------------------------------

class TestDualFace:
    """dual_face() requires a reflexive ambient polytope.

    Constructs the dual polytope and maps this face to its dual via
    inequality matching.  Reflexivity is guaranteed by h11-filtered fixtures.
    """

    def test_dual_face_tiny_reflexive(self, benchmark, reflexive_faces):
        def go():
            results = []
            for f in reflexive_faces:
                try:
                    results.append(f.dual_face())
                except Exception:
                    pass
            return results
        benchmark.pedantic(go, rounds=3, iterations=1)


# ---------------------------------------------------------------------------
# 6. sub-faces of a face
# ---------------------------------------------------------------------------

class TestSubFaces:
    """faces() recursively enumerates sub-faces of a PolytopeFace.

    Filters all ambient polytope faces by inequality saturation — scales
    with total face lattice size.
    """

    def test_faces_of_faces_tiny(self, benchmark, tiny_poly_objects):
        """Sub-faces of codim-1 faces (facets) of tiny polytopes."""
        facets = []
        for p in tiny_poly_objects:
            try:
                facets.extend(p.faces(p.dimension() - 1))
            except Exception:
                pass

        def go():
            return [f.faces() for f in facets]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_faces_of_faces_bulk(self, benchmark, bulk_codim1_faces):
        """Sub-faces of codim-1 faces of bulk polytopes."""
        def go():
            return [f.faces() for f in bulk_codim1_faces]
        benchmark.pedantic(go, rounds=1, iterations=1)


# ---------------------------------------------------------------------------
# 7. triangulate — triangulation of a face
# ---------------------------------------------------------------------------

class TestTriangulate:
    """triangulate() runs an external triangulation solver on the face's points.

    Even for a 2D face this is non-trivial at bulk complexity.
    Use 2D faces to keep the test tractable at bulk tier.
    """

    def test_triangulate_2d_faces_tiny(self, benchmark, tiny_poly_objects):
        """Triangulate all 2D faces of tiny polytopes."""
        faces_2d = []
        for p in tiny_poly_objects:
            try:
                faces_2d.extend(p.faces(2))
            except Exception:
                pass

        def go():
            return [f.triangulate() for f in faces_2d]
        benchmark.pedantic(go, rounds=3, iterations=1)

    def test_triangulate_2d_faces_bulk(self, benchmark, bulk_poly_objects):
        """Triangulate all 2D faces of bulk polytopes."""
        faces_2d = []
        for p in bulk_poly_objects:
            try:
                faces_2d.extend(p.faces(2))
            except Exception:
                pass

        def go():
            return [f.triangulate() for f in faces_2d]
        benchmark.pedantic(go, rounds=1, iterations=1)

    @pytest.mark.slow
    def test_triangulate_3d_faces_bulk(self, benchmark, bulk_poly_objects):
        """Triangulate all 3D faces of bulk polytopes — much more expensive."""
        faces_3d = []
        for p in bulk_poly_objects:
            try:
                faces_3d.extend(p.faces(3))
            except Exception:
                pass

        def go():
            return [f.triangulate() for f in faces_3d]
        benchmark.pedantic(go, rounds=1, iterations=1)
