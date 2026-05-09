"""Tests for moirais.fn.smcub — SMACOF unfolding."""
import numpy as np
from moirais.fn.smcub import smcub


def test_smcub_basic():
    D = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1], [1.5, 2.5, 0.5]])
    r = smcub(D, n_dims=2)
    assert r.coordinates.shape == (7, 2)
    assert r.stress >= 0


def test_smcub_square():
    D = np.array([[0, 1], [1, 0]])
    r = smcub(D, n_dims=1)
    assert r.coordinates.shape == (4, 1)


def test_smcub_convergence():
    rng = np.random.default_rng(42)
    D = rng.random((5, 3)) * 2
    r = smcub(D, n_dims=2, max_iter=500)
    assert r.stress < 1.0
