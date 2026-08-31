"""Tests for the native PPL floating-point compatibility boundary."""

import ctypes
import subprocess
import sys

import pytest

from cytools._backends import fpu


class _FakeSetter:
    def __init__(self, result: int = 0) -> None:
        self.argtypes = None
        self.restype = None
        self.result = result
        self.calls: list[int] = []

    def __call__(self, mode: int) -> int:
        self.calls.append(mode)
        return self.result


class _FakeLibc:
    def __init__(self, result: int = 0) -> None:
        self.fesetround = _FakeSetter(result)


def test_reset_rounding_mode_calls_c_api_safely(monkeypatch):
    libc = _FakeLibc()
    monkeypatch.setattr(fpu.ctypes, "CDLL", lambda _: libc)

    fpu.reset_rounding_mode()

    assert libc.fesetround.calls == [0]
    assert libc.fesetround.argtypes == [ctypes.c_int]
    assert libc.fesetround.restype is ctypes.c_int


def test_reset_rounding_mode_reports_native_failure(monkeypatch):
    libc = _FakeLibc(result=1)
    monkeypatch.setattr(fpu.ctypes, "CDLL", lambda _: libc)

    with pytest.raises(RuntimeError, match="restore the process FPU"):
        fpu.reset_rounding_mode()


def test_ppl_boundary_resets_rounding_once_for_all_domain_modules():
    code = """
from cytools._backends import fpu

calls = []
fpu.reset_rounding_mode = lambda: calls.append(None)

from cytools.polytope import Polytope
from cytools.cone import Cone
from cytools.h_polytope import HPolytope

assert Polytope.__name__ == "Polytope"
assert Cone.__name__ == "Cone"
assert HPolytope.__name__ == "HPolytope"
assert len(calls) == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
