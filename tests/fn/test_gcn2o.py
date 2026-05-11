"""Test gcn2o."""
import numpy as np
import pytest
from morie.fn.gcn2o import gcn2o


def test_gcn2o_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcn2o(data=data, coords=coords, n=30)
    assert r.value is not None


def test_gcn2o_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcn2o(data=data, coords=coords, n=30)
    assert r.name
