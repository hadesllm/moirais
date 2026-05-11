"""Test wlhme."""
import numpy as np
import pytest
from morie.fn.wlhme import wlhme


def test_wlhme_basic():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlhme(abundance=abund, coords=coords, n=20)
    assert isinstance(r.value, float) and np.isfinite(r.value)
    assert r.value > 0
    assert r.value == pytest.approx(np.mean(abund), rel=1e-10)


def test_wlhme_extra():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlhme(abundance=abund, coords=coords, n=20)
    assert isinstance(r.name, str) and len(r.name) > 0
    assert r.extra["n"] == 20
    assert r.extra["total"] == int(np.sum(abund))
