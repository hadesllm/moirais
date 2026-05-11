"""Test entrd2."""
import numpy as np
import pytest
from morie.fn.entrd2 import entrd2


def test_entrd2_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = entrd2(data=data, coords=coords, n=30)
    assert r.value is not None


def test_entrd2_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = entrd2(data=data, coords=coords, n=30)
    assert r.name
