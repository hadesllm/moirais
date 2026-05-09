"""Tests for bcaci (BCa bootstrap CI)."""
import numpy as np
from moirais.fn.bcaci import bca_ci


def test_bca_ci_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(100)
    r = bca_ci(data, stat_fn=np.mean, n_boot=500, seed=42)
    assert r.extra["lower"] < r.extra["upper"]
    assert r.extra["lower"] < 0.5


def test_cheatsheet():
    from moirais.fn.bcaci import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
