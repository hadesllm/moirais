"""Tests for moirais.fn.ocslt — OC scaling."""
import numpy as np
from moirais.fn.ocslt import ocslt


def test_ocslt_basic():
    V = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1]], dtype=float)
    r = ocslt(V, n_dims=1)
    assert r.value["ideal_points"].shape == (4, 1)
    assert 0 <= r.value["classification_rate"] <= 1


def test_ocslt_2d():
    rng = np.random.default_rng(42)
    V = rng.integers(0, 2, (10, 5)).astype(float)
    r = ocslt(V, n_dims=2)
    assert r.value["ideal_points"].shape == (10, 2)


def test_ocslt_extra():
    V = np.array([[1, 0], [0, 1]], dtype=float)
    r = ocslt(V)
    assert r.extra["n_legislators"] == 2
