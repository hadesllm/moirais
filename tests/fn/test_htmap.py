"""Tests for moirais.fn.htmap — heatmap issue weights."""
import numpy as np
from moirais.fn.htmap import htmap


def test_htmap_smoke():
    W = np.array([[0.8, 0.1], [0.2, 0.9], [0.5, 0.5]])
    r = htmap(W)
    assert r.name == "heatmap_issue_weights"
    assert r.value == 3
    assert r.extra["n_dims"] == 2


def test_cheatsheet():
    from moirais.fn.htmap import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
