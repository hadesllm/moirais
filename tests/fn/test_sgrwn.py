"""Tests for row normalize weights."""
import numpy as np
from moirais.fn.sgrwn import sgrwn


def test_sgrwn_smoke():
    W = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    r = sgrwn(W)
    assert r.name == "row_normalize_weights"
    W_norm = r.extra["W_normalized"]
    assert np.allclose(W_norm.sum(axis=1), 1.0)


def test_cheatsheet():
    from moirais.fn.sgrwn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
