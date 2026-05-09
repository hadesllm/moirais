"""Tests for kriging cross-validation."""
import numpy as np
from moirais.fn.sgkvl import sgkvl


def test_sgkvl_smoke():
    rng = np.random.default_rng(5)
    coords = rng.uniform(0, 10, (15, 2))
    Z = coords[:, 0] + rng.normal(0, 0.5, 15)
    r = sgkvl(Z, coords)
    assert r.name == "kriging_cross_validation"
    assert r.statistic >= 0
    assert "mae" in r.extra
    assert len(r.extra["errors"]) == 15


def test_cheatsheet():
    from moirais.fn.sgkvl import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
