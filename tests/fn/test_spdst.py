"""Tests for moirais.fn.spdst — spatial distance."""
from moirais.fn.spdst import spdst


def test_spdst_euclidean():
    r = spdst([0, 0], [3, 4])
    assert r.name == "spatial_distance"
    assert abs(r.value - 5.0) < 1e-10


def test_spdst_manhattan():
    r = spdst([0, 0], [3, 4], metric="manhattan")
    assert r.value == 7.0
