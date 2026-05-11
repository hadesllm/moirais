"""Test gcco2."""
import numpy as np
import pytest
from morie.fn.gcco2 import gcco2


def test_gcco2_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcco2(data=data, coords=coords, n=30)
    assert r.value is not None


def test_gcco2_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcco2(data=data, coords=coords, n=30)
    assert r.name
