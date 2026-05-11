"""Tests for morie.fn.smvot -- simulate spatial voting."""
import numpy as np
from morie.fn.smvot import simulate_voting_spatial, smvot


def test_alias():
    assert smvot is simulate_voting_spatial


def test_smoke():
    X = np.array([[-0.5], [0.0], [0.5]])
    bills = np.array([[0.0], [0.3]])
    r = simulate_voting_spatial(X, bills)
    assert r.name == "simulate_voting_spatial"
    assert r.extra["n_legislators"] == 3
    assert r.extra["n_bills"] == 2
    assert r.extra["vote_matrix"].shape == (3, 2)
