"""Test abbac."""
import numpy as np
import pytest
from morie.fn.abbac import abbac


def test_abbac_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abbac(data=data, coords=coords, n=20)
    assert isinstance(r.value, float) and np.isfinite(r.value)
    assert r.value == pytest.approx(np.mean(data), rel=1e-10)


def test_abbac_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abbac(data=data, coords=coords, n=20)
    assert isinstance(r.name, str) and len(r.name) > 0
