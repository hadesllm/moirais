"""Tests for spatial ACF."""
import numpy as np
from moirais.fn.sgacf import sgacf


def test_sgacf_smoke():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (50, 2))
    Z = rng.normal(0, 1, 50)
    r = sgacf(Z, coords, n_lags=5)
    assert r.name == "spatial_acf"
    assert "lag_distances" in r.extra
    assert "acf_values" in r.extra
    assert len(r.extra["acf_values"]) == 5


def test_cheatsheet():
    from moirais.fn.sgacf import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
