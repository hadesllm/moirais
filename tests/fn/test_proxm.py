"""Tests for moirais.fn.proxm — proximity cost."""
from moirais.fn.proxm import proxm


def test_proxm_smoke():
    r = proxm([0.0], [[1.0], [3.0]])
    assert r.name == "proximity_cost"
    assert r.value == 1.0
    assert r.extra["closest"] == 0


def test_proxm_manhattan():
    r = proxm([0, 0], [[1, 1], [2, 2]], metric="manhattan")
    assert r.extra["distances"][0] == 2.0
