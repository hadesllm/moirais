"""Test gcupw."""
import numpy as np
import pytest
from morie.fn.gcupw import gcupw


def test_gcupw_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcupw(data=data, coords=coords, n=30)
    assert r.value is not None


def test_gcupw_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcupw(data=data, coords=coords, n=30)
    assert r.name
