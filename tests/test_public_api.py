import threading

import cytools


def test_version_aliases_agree():
    assert cytools.__version__ == cytools.version
    assert cytools.__upstream_version__ == cytools.upstream_version == "1.4.12"


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


def test_feature_namespaces_export_only_supported_names():
    import cytools.f_theory as f_theory
    import cytools.ntfe as ntfe
    import cytools.vector_config as vector_config

    assert set(ntfe.__all__) == {
        "cone_of_permissible_heights",
        "expanded_secondary_fan",
        "face_triangulations",
        "ntfe_cones",
        "ntfe_frsts",
        "ntfe_frts",
        "ntfe_hypers",
        "triangface_ineqs",
        "triangfaces_to_frst",
        "triangfaces_to_frt",
    }
    assert set(vector_config.__all__) == {"Fan", "VectorConfiguration"}
    assert set(f_theory.__all__) == {
        "CY_orientifold",
        "F_Theory_Uplift",
        "fetch_F_Theory_uplifts",
        "fetch_nef_partition_uplifts",
        "fetch_orientifolds",
    }

    assert not hasattr(ntfe, "np")
    assert not hasattr(vector_config, "regfans")
    assert not hasattr(f_theory, "Polytope")
