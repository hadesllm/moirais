"""Test wqpo4."""
import numpy as np
import pytest
from morie.fn.wqpo4 import wqpo4


def test_wqpo4_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqpo4(data=data, coords=coords, n=20)
    assert r.value is not None


def test_wqpo4_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqpo4(data=data, coords=coords, n=20)
    assert r.name
