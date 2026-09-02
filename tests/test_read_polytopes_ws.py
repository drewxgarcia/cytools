"""
Regression tests for weight-system input to `read_polytopes`.

`format="ws"` is reached in production by `fetch_polytopes` for every
non-4-dimensional request (`format=("ks" if dim == 4 else "ws")`), so it is
live code despite having no direct call site in the tree.

Reading from a file used to wrap the *file name* in a `StringIO` and hand that
to PALP as a weight system, so the same data gave different answers depending
on whether it arrived as a string or as a file.
"""

import gc
from typing import cast

import pytest

from cytools._typing import PolytopeFormat
from cytools.utils import read_polytopes

#: The quintic, and a second weight system, one per line.
WS_TEXT = "1 1 1 1 1 5\n1 1 1 3 6 12\n"


@pytest.fixture
def ws_file(tmp_path):
    path = tmp_path / "weights.txt"
    path.write_text(WS_TEXT)
    return str(path)


def _vertices(polytopes):
    return [p.vertices().tolist() for p in polytopes]


def test_file_input_matches_string_input(ws_file):
    """The regression: identical data, identical polytopes, either way in."""
    from_string = read_polytopes(WS_TEXT, input_type="str", format="ws", as_list=True)
    from_file = read_polytopes(ws_file, input_type="file", format="ws", as_list=True)
    assert _vertices(from_file) == _vertices(from_string)


def test_file_input_yields_every_weight_system(ws_file):
    """Both lines are read -- an EOF probe on the same handle skipped every other one."""
    polytopes = list(
        read_polytopes(ws_file, input_type="file", format="ws", as_list=True)
    )
    assert len(polytopes) == 2


def test_limit_is_respected(ws_file):
    polytopes = list(
        read_polytopes(ws_file, input_type="file", format="ws", as_list=True, limit=1)
    )
    assert len(polytopes) == 1


def test_trailing_blank_lines_terminate_cleanly(tmp_path):
    """A blank line must stop the read: PALP aborts the process if handed one."""
    path = tmp_path / "trailing.txt"
    path.write_text(WS_TEXT + "\n\n\n")
    polytopes = list(
        read_polytopes(str(path), input_type="file", format="ws", as_list=True)
    )
    assert len(polytopes) == 2


def test_does_not_fall_through_into_the_ks_reader(ws_file):
    """The `ws` branch used to run on into the `ks` loop once it finished."""
    polytopes = read_polytopes(ws_file, input_type="file", format="ws", as_list=True)
    assert all(p.dimension() == 4 for p in polytopes)


def test_file_handle_is_closed(ws_file):
    """Exhausting the generator must not leave the file open."""
    generator = read_polytopes(ws_file, input_type="file", format="ws", as_list=False)
    list(generator)
    gc.collect()
    open_paths = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, type(open(ws_file))) and not obj.closed:
                open_paths.append(getattr(obj, "name", None))
        except (ReferenceError, TypeError):
            continue
    assert ws_file not in open_paths


def test_unknown_format_is_rejected(ws_file):
    with pytest.raises(ValueError, match="Unsupported format"):
        read_polytopes(
            ws_file,
            input_type="file",
            format=cast(PolytopeFormat, "nonsense"),
            as_list=True,
        )
