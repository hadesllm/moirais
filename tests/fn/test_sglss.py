"""Tests for squared error loss."""
import numpy as np
from moirais.fn.sglss import sglss


def test_sglss_smoke():
    pred = np.array([1.0, 2.0, 3.0])
    obs = np.array([1.1, 1.9, 3.2])
    r = sglss(pred, obs)
    assert r.name == "squared_error_loss"
    assert r.statistic >= 0
    assert "losses" in r.extra
    assert len(r.extra["losses"]) == 3


def test_cheatsheet():
    from moirais.fn.sglss import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
