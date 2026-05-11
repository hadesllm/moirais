"""Test nbl90."""
import numpy as np
import pytest
from morie.fn.nbl90 import nbl90


def test_nbl90_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbl90(data=data, coords=coords, n=20)
    assert r.value is not None


def test_nbl90_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbl90(data=data, coords=coords, n=20)
    assert r.name
