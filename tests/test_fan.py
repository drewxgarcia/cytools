import numpy as np

from cytools import Polytope


def fan_fixture():
    p = Polytope(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -6, -9]]
    )
    return p.triangulate().fan()


def test_intersection_numbers_call_order_digits():
    fan_after = fan_fixture()
    fan_fresh = fan_fixture()

    fan_after.intersection_numbers(digits=0, symmetrize=False)
    after = fan_after.intersection_numbers(symmetrize=False)
    fresh = fan_fresh.intersection_numbers(symmetrize=False)

    assert after == fresh
    assert len(after) == 121
    assert np.isclose(after[(1, 1, 2, 6)], 0.5)
    assert np.isclose(after[(1, 1, 6, 6)], 1 / 6)
    assert np.isclose(after[(2, 2, 2, 2)], 121.5)


def test_intersection_numbers_call_order_eps():
    fan_after = fan_fixture()
    fan_fresh = fan_fixture()

    fan_after.intersection_numbers(eps=0.6, digits=None, symmetrize=False)
    after = fan_after.intersection_numbers(symmetrize=False)
    fresh = fan_fresh.intersection_numbers(symmetrize=False)

    assert after == fresh
    assert len(after) == 121
    assert (1, 1, 2, 6) in after
    assert np.isclose(after[(1, 1, 3, 6)], 1 / 3)


def test_mori_rays_after_low_precision_intersection_numbers():
    fan_after = fan_fixture()
    fan_fresh = fan_fixture()

    fan_after.intersection_numbers(digits=0, symmetrize=False)

    after = sorted(map(tuple, fan_after.mori_rays().tolist()))
    fresh = sorted(map(tuple, fan_fresh.mori_rays().tolist()))

    assert after == fresh


# ---------------------------------------------------------------------------
# restricted_simps
# ---------------------------------------------------------------------------


def test_restricted_simps_pads_label_simplices():
    """`padded=True` must work on the label path, not just on face indices.

    Regression test: the reduction step yields frozensets, but the padding step
    indexed and concatenated them (`simp + [simp[-1]]`), which raised
    `TypeError: 'frozenset' object is not subscriptable`. It only fired when a
    restricted simplex actually had two points -- i.e. exactly when `padded`
    has work to do -- so `to_dim=1` is the trigger, and `to_dim=2` (triangles)
    is not.
    """
    fan = fan_fixture()

    unpadded = fan.restricted_simps(to_dim=1, padded=False, as_face_inds=False)
    assert any(len(simp) == 2 for face in unpadded for simp in face), (
        "fixture no longer produces 2-point restricted simplices, "
        "so this test would pass vacuously"
    )

    padded = fan.restricted_simps(to_dim=1, padded=True, as_face_inds=False)
    assert all(len(simp) >= 3 for face in padded for simp in face)

    # padding duplicates the last entry rather than inventing a new point
    for face in padded:
        for simp in face:
            assert len(set(simp)) <= 2


def test_restricted_simps_label_and_index_paths_agree():
    """Both output spaces must describe the same simplices."""
    fan = fan_fixture()
    by_label = fan.restricted_simps(to_dim=2, padded=False, as_face_inds=False)
    by_index = fan.restricted_simps(to_dim=2, padded=False, as_face_inds=True)

    assert len(by_label) == len(by_index)
    for labels, inds in zip(by_label, by_index):
        assert [len(s) for s in labels] == [len(s) for s in inds]
