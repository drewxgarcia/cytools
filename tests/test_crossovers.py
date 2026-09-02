"""Recorded engine measurements support the automatic ordering policy."""

import json
from pathlib import Path

from cytools._backends.crossovers import CROSSOVERS
from cytools._backends.engines import CONVEX_HULL

ARTIFACT = (
    Path(__file__).parents[1] / "benchmarks" / "artifacts" / "engine_crossovers.json"
)


def test_convex_hull_order_matches_the_isolated_measurement():
    report = json.loads(ARTIFACT.read_text())
    results = report["tasks"]["convex_hull"]

    # PPL beats PALP on every measured representative through dimension five.
    for dim in range(2, 6):
        for extra in (0, 12, 30):
            cell = results[f"{dim}x{extra}"]
            assert cell["ppl"]["status"] == cell["palp"]["status"] == "ok"
            assert cell["ppl"]["seconds"] < cell["palp"]["seconds"]

    # At larger configurations PALP can terminate the interpreter, so a speed
    # win in one dimension-six cell cannot make it an automatic engine.
    assert results["8x30"]["palp"]["status"] == "aborted"
    assert CONVEX_HULL.names().index("ppl") < CONVEX_HULL.names().index("palp")


def test_unmeasured_crossovers_are_explicitly_labelled():
    assert CROSSOVERS["stretched_tip.osqp_to_mosek"].measured is False
