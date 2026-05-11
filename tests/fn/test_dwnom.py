"""Tests for DW-NOMINATE."""
import numpy as np
from morie.fn.dwnom import dwnom


def test_dwnom_smoke():
    rng = np.random.default_rng(42)
    votes = (rng.random((15, 20)) > 0.4).astype(float)
    votes[rng.random(votes.shape) < 0.1] = np.nan
    r = dwnom(votes, n_dims=1, max_iter=10)
    assert r.name == "dw_nominate_estimate"
    assert "ideal_points" in r.extra
    assert r.extra["gmp"] > 0.0


def test_cheatsheet():
    from morie.fn.dwnom import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
