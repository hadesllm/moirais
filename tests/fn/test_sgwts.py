"""Tests for spatial weights matrix."""
import numpy as np
from moirais.fn.sgwts import sgwts


def test_sgwts_knn():
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]], dtype=float)
    r = sgwts(coords, method="knn", k=2)
    assert r.name == "spatial_weights_matrix"
    assert "W" in r.extra
    assert "W_row" in r.extra
    W_row = r.extra["W_row"]
    assert np.allclose(W_row.sum(axis=1), 1.0)


def test_sgwts_distance():
    coords = np.array([[0, 0], [1, 0], [0, 1], [5, 5]], dtype=float)
    r = sgwts(coords, method="distance", bandwidth=2.0)
    assert r.extra["W"][0, 3] == 0.0
