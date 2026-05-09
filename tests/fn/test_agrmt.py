"""Tests for moirais.fn.agrmt — Agreement score."""
import numpy as np
from moirais.fn.agrmt import agrmt


def test_agrmt_perfect():
    V = np.array([[1, 1, 0], [1, 1, 0]], dtype=float)
    r = agrmt(V)
    assert r.value["agreement_matrix"][0, 1] == 1.0


def test_agrmt_opposite():
    V = np.array([[1, 1, 0], [0, 0, 1]], dtype=float)
    r = agrmt(V)
    assert r.value["agreement_matrix"][0, 1] == 0.0


def test_agrmt_stats():
    V = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]], dtype=float)
    r = agrmt(V)
    assert 0.0 <= r.extra["mean_agreement"] <= 1.0
    assert r.extra["n_legislators"] == 3
