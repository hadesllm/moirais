"""Test gchwv."""
import numpy as np
import pytest
from morie.fn.gchwv import gchwv


def test_gchwv_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gchwv(data=data, coords=coords, n=30)
    assert r.value is not None


def test_gchwv_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gchwv(data=data, coords=coords, n=30)
    assert r.name
