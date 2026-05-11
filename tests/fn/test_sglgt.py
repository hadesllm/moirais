"""Tests for spatial logistic regression."""
import numpy as np
from morie.fn.sglgt import sglgt


def test_sglgt_smoke():
    rng = np.random.default_rng(19)
    n = 30
    coords = rng.uniform(0, 10, (n, 2))
    X = np.column_stack([np.ones(n), coords[:, 0]])
    prob = 1.0 / (1.0 + np.exp(-(X @ np.array([-1.0, 0.3]))))
    Y = rng.binomial(1, prob).astype(float)
    r = sglgt(Y, X, coords)
    assert r.name == "spatial_logistic"
    assert "beta" in r.extra
    assert "fitted_prob" in r.extra


def test_cheatsheet():
    from morie.fn.sglgt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
