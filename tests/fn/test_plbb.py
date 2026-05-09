"""Tests for moirais.fn.plbb — plot Blackbox result."""
import numpy as np
from moirais.fn.plbb import plbb


def test_plbb_smoke():
    X = np.array([[1, 2], [3, 4], [5, 6.0]])
    r = plbb(X)
    assert r.name == "plot_blackbox_result"
    assert r.value == 3
    assert r.extra["x"] == [1, 3, 5]


def test_cheatsheet():
    from moirais.fn.plbb import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
