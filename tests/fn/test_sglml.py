"""Tests for LM spatial tests."""
import numpy as np
from moirais.fn.sglml import sglml


def test_sglml_smoke():
    rng = np.random.default_rng(14)
    n = 20
    resid = rng.normal(0, 1, n)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i-1), min(n, i+2)):
            if i != j:
                W[i, j] = 0.5
    r = sglml(resid, W)
    assert r.name == "lm_test_spatial"
    assert "lm_lag" in r.extra
    assert "lm_error" in r.extra
    assert "p_lag" in r.extra


def test_cheatsheet():
    from moirais.fn.sglml import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
