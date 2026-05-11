"""Tests for morie.fn.sephy — separating hyperplane."""
import numpy as np
from morie.fn.sephy import sephy


def test_sephy_smoke():
    X = np.array([[0, 0], [1, 0], [3, 0], [4, 0]])
    labels = np.array([0, 0, 1, 1])
    r = sephy(X, labels)
    assert r.name == "separating_hyperplane"
    assert "normal" in r.extra
    assert "midpoint" in r.extra


def test_cheatsheet():
    from morie.fn.sephy import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
