"""Tests for simulation envelope."""
import numpy as np
from moirais.fn.sgenv import sgenv


def test_sgenv_smoke():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 10, (30, 2))
    stat_fn = lambda p: np.array([p[:, 0].mean()])
    r = sgenv(pts, (0, 10, 0, 10), stat_fn, n_sim=19, seed=42)
    assert r.name == "simulation_envelope"
    assert "envelope_lo" in r.extra
    assert "envelope_hi" in r.extra


def test_cheatsheet():
    from moirais.fn.sgenv import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
