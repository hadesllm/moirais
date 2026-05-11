"""Test gcstd."""
import numpy as np
import pytest
from morie.fn.gcstd import gcstd


def test_gcstd_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcstd(data=data, coords=coords, n=30)
    assert r.value is not None


def test_gcstd_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcstd(data=data, coords=coords, n=30)
    assert r.name
