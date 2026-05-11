"""Tests for kriging prediction error."""
import numpy as np
from morie.fn.sgkpe import sgkpe


def test_sgkpe_smoke():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.1, 1.9, 3.2, 3.8])
    r = sgkpe(obs, pred)
    assert r.name == "kriging_prediction_error"
    assert r.statistic >= 0
    assert "mae" in r.extra
    assert "me" in r.extra


def test_cheatsheet():
    from morie.fn.sgkpe import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
