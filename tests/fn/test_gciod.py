"""Test gciod."""
import numpy as np
import pytest
from morie.fn.gciod import gciod


def test_gciod_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gciod(data=data, coords=coords, n=30)
    assert r.value is not None


def test_gciod_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gciod(data=data, coords=coords, n=30)
    assert r.name
