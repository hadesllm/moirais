"""Tests for alpha-NOMINATE."""
import numpy as np
from moirais.fn.anom import anom


def test_anom_smoke():
    rng = np.random.default_rng(42)
    votes = (rng.random((8, 10)) > 0.4).astype(float)
    r = anom(votes, n_dims=1, n_samples=20, burn_in=10)
    assert r.name == "alpha_nominate_estimate"
    assert 0.0 <= r.extra["alpha_mean"] <= 1.0
    assert "ideal_points" in r.extra


def test_cheatsheet():
    from moirais.fn.anom import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
