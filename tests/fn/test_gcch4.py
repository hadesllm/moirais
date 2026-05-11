"""Test gcch4."""
import numpy as np
import pytest
from morie.fn.gcch4 import gcch4


def test_gcch4_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcch4(data=data, coords=coords, n=30)
    assert r.value is not None


def test_gcch4_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcch4(data=data, coords=coords, n=30)
    assert r.name
