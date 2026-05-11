"""Tests for NOMINATE bootstrap."""
import numpy as np
from morie.fn.nombs import nombs


def test_nombs_smoke():
    rng = np.random.default_rng(42)
    n_leg, n_votes = 10, 15
    votes = (rng.random((n_leg, n_votes)) > 0.4).astype(float)
    X = rng.standard_normal((n_leg, 1))
    nv = rng.standard_normal((n_votes, 1))
    nv /= np.abs(nv) + 1e-6
    cp = np.zeros(n_votes)
    r = nombs(votes, X, nv, cp, n_boot=5)
    assert r.name == "nominate_bootstrap_se"
    assert "se_ideal_points" in r.extra
    assert r.extra["se_ideal_points"].shape == (n_leg, 1)


def test_cheatsheet():
    from morie.fn.nombs import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
