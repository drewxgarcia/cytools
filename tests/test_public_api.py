import threading

import cytools


def test_version_aliases_agree():
    assert cytools.__version__ == cytools.version


def test_public_api_is_explicit():
    expected = {
        "Cone",
        "Geometry",
        "HPolytope",
        "Polytope",
        "load_polytopes",
        "scan",
        "sweep",
    }
    assert expected <= set(cytools.__all__)


def test_import_does_not_start_an_update_thread():
    assert not any(
        thread.name == "cytools-update-check" for thread in threading.enumerate()
    )
