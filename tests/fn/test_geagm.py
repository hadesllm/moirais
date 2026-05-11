"""Test geagm."""
import numpy as np
import pytest
from morie.fn.geagm import geagm


def test_geagm_basic():
    rng = np.random.default_rng(42)
    gdp = rng.uniform(1000, 100000, 20)
    trade = rng.uniform(100, 50000, 20)
    r = geagm(gdp=gdp, trade=trade, n=20)
    assert isinstance(r.value, float) and np.isfinite(r.value)
    assert r.value > 0
    assert r.value == pytest.approx(np.mean(gdp), rel=1e-10)


def test_geagm_extra():
    rng = np.random.default_rng(42)
    gdp = rng.uniform(1000, 100000, 20)
    trade = rng.uniform(100, 50000, 20)
    r = geagm(gdp=gdp, trade=trade, n=20)
    assert isinstance(r.name, str) and len(r.name) > 0
    assert r.extra["n"] == 20
    assert r.extra["total_trade"] == pytest.approx(np.sum(trade), rel=1e-10)
