"""Tests for average covariance block."""
import numpy as np
from moirais.fn.sgavgc import sgavgc


def test_sgavgc_smoke():
    block = np.array([[0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5]], dtype=float)
    r = sgavgc(block)
    assert r.name == "average_covariance_block"
    assert r.statistic > 0
    assert r.extra["n_points"] == 4


def test_cheatsheet():
    from moirais.fn.sgavgc import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
