"""Tests for bootstrap scaling."""
import numpy as np
from moirais.fn.btsca import btsca


def test_btsca_smoke():
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((20, 4)) + np.array([1, 3, 5, 7])
    r = btsca(Z, scale_fn="am", n_boot=10)
    assert r.name == "bootstrap_scaling_se"
    assert "se_positions" in r.extra
    assert "ci_lower" in r.extra


def test_cheatsheet():
    from moirais.fn.btsca import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
