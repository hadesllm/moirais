"""Test nbl10."""
import numpy as np
import pytest
from moirais.fn.nbl10 import nbl10


def test_nbl10_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbl10(data=data, coords=coords, n=20)
    assert r.value is not None


def test_nbl10_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbl10(data=data, coords=coords, n=20)
    assert r.name
