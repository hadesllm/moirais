"""Test tsstk."""
import numpy as np
import pytest
from moirais.fn.tsstk import tsstk


def test_tsstk_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tsstk(data=data, coords=coords, n=20, t=5)
    assert isinstance(r.value, float)
    assert np.isfinite(r.value), "Space-time K statistic must be finite"
    assert abs(r.value) < 3.0, f"Mean of standard normal data {r.value} implausibly far from 0"


def test_tsstk_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tsstk(data=data, coords=coords, n=20, t=5)
    assert r.name
