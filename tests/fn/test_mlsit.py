"""Tests for moirais.fn.mlsit -- MLSMU6 single iteration."""

import numpy as np
from moirais.fn.mlsit import mlsmu6_single_iteration, mlsit


def test_mlsit_smoke():
    X = np.array([[0, 0], [1, 1]], dtype=float)
    Y = np.array([[0.5, 0], [0, 0.5], [1, 0.5]], dtype=float)
    D = np.array([[0.5, 0.7, 1.1], [0.8, 1.2, 0.5]], dtype=float)
    r = mlsit(X, Y, D)
    assert r.name == "mlsmu6_single_iteration"
    X_new, Y_new = r.value
    assert X_new.shape == (2, 2)
    assert Y_new.shape == (3, 2)
    assert "stress" in r.extra


def test_mlsit_alias():
    assert mlsit is mlsmu6_single_iteration
