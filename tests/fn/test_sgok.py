"""Tests for ordinary kriging."""
import numpy as np
from moirais.fn.sgok import sgok


def test_sgok_smoke():
    rng = np.random.default_rng(1)
    coords = rng.uniform(0, 10, (15, 2))
    Z = coords[:, 0] + rng.normal(0, 0.1, 15)
    r = sgok(Z, coords, np.array([5.0, 5.0]))
    assert r.name == "ordinary_kriging"
    assert "predictions" in r.extra
    assert "variances" in r.extra


def test_sgok_weights_sum():
    coords = np.array([[0, 0], [2, 0], [0, 2]], dtype=float)
    Z = np.array([1.0, 2.0, 3.0])
    r = sgok(Z, coords, np.array([1.0, 1.0]))
    w = r.extra["weights"][0]
    assert abs(w.sum() - 1.0) < 1e-6
