"""Tests for morie.fn.pstdn -- posterior density."""
import numpy as np
from morie.fn.pstdn import posterior_density_data, pstdn


def test_alias():
    assert pstdn is posterior_density_data


def test_smoke():
    chain = np.random.default_rng(42).standard_normal(500)
    r = posterior_density_data(chain)
    assert r.name == "posterior_density_data"
    assert "grid" in r.extra
    assert "density" in r.extra
    assert len(r.extra["grid"]) == 200
    assert r.extra["bandwidth"] > 0
