"""Focused regressions for lightweight runtime and diagnostics helpers."""

import builtins
import types

import requests


def test_known_bad_version_warns_even_when_the_update_request_fails(
    monkeypatch, capsys
):
    import cytools._updates as updates

    monkeypatch.setattr(updates, "versions_with_serious_bugs", (updates.version,))

    def offline(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", offline)
    updates.check_for_updates()

    assert "contains a serious bug" in capsys.readouterr().out


def test_broken_mosek_import_reports_the_original_failure(monkeypatch):
    import cytools.config as config

    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "mosek":
            raise OSError("unloadable libmosek")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    monkeypatch.setattr(config, "_mosek_error", None)
    monkeypatch.setattr(config, "_mosek_is_activated", None)

    config.check_mosek_license(silent=True)

    assert config._mosek_is_activated is False
    assert "OSError: unloadable libmosek" in config._mosek_error


def test_boundary_edges_ignore_orientation():
    from cytools.helpers.basic_geometry import get_bdry

    owner = types.SimpleNamespace(
        triangulate=lambda: types.SimpleNamespace(
            simplices=lambda: [[0, 1, 2], [2, 1, 3]]
        )
    )

    assert get_bdry(owner) == {
        frozenset((0, 1)),
        frozenset((0, 2)),
        frozenset((1, 3)),
        frozenset((2, 3)),
    }
