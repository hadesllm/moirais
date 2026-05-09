"""Test wqwqi."""
import numpy as np
import pytest
from moirais.fn.wqwqi import wqwqi


def test_wqwqi_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqwqi(data=data, coords=coords, n=20)
    assert r.value is not None


def test_wqwqi_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqwqi(data=data, coords=coords, n=20)
    assert r.name
