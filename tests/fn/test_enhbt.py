"""Test enhbt."""
import numpy as np
import pytest
from morie.fn.enhbt import enhbt


def test_enhbt_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enhbt(data=data, coords=coords, n=30)
    assert r.value is not None


def test_enhbt_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enhbt(data=data, coords=coords, n=30)
    assert r.name
