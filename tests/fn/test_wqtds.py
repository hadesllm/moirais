"""Test wqtds."""
import numpy as np
import pytest
from morie.fn.wqtds import wqtds


def test_wqtds_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqtds(data=data, coords=coords, n=20)
    assert r.value is not None


def test_wqtds_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqtds(data=data, coords=coords, n=20)
    assert r.name
