"""Test wqso4."""
import numpy as np
import pytest
from morie.fn.wqso4 import wqso4


def test_wqso4_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqso4(data=data, coords=coords, n=20)
    assert r.value is not None


def test_wqso4_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqso4(data=data, coords=coords, n=20)
    assert r.name
