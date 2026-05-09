"""Tests for conditional CDF via indicator kriging."""
import numpy as np
from moirais.fn.sgccdf import sgccdf


def test_sgccdf_smoke():
    rng = np.random.default_rng(8)
    coords = rng.uniform(0, 5, (20, 2))
    Z = rng.normal(5, 2, 20)
    r = sgccdf(Z, coords, np.array([2.5, 2.5]))
    assert r.name == "conditional_cdf_indicator"
    assert "cdf_values" in r.extra
    cdf = r.extra["cdf_values"]
    assert np.all(np.diff(cdf) >= -1e-10)


def test_cheatsheet():
    from moirais.fn.sgccdf import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
