"""Test gcslr."""
import numpy as np
import pytest
from morie.fn.gcslr import gcslr


def test_gcslr_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcslr(data=data, coords=coords, n=30)
    assert r.value is not None


def test_gcslr_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcslr(data=data, coords=coords, n=30)
    assert r.name
