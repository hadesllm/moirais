"""Tests for morie.fn.nopar -- NOMINATE parameters."""
import numpy as np
from morie.fn.nopar import nominate_parameters, nopar


def test_alias():
    assert nopar is nominate_parameters


def test_smoke():
    votes = np.random.default_rng(42).choice([0, 1], (10, 5)).astype(float)
    X = np.random.default_rng(42).standard_normal((10, 2))
    nv = np.random.default_rng(42).standard_normal(5)
    mid = np.random.default_rng(42).standard_normal(5)
    r = nominate_parameters(votes, X, nv, mid)
    assert r.name == "nominate_parameters"
    assert "n_bills" in r.extra
    assert r.extra["n_bills"] == 5
