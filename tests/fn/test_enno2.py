"""Test enno2."""
import numpy as np
import pytest
from morie.fn.enno2 import enno2


def test_enno2_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enno2(data=data, coords=coords, n=30)
    assert r.value is not None


def test_enno2_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enno2(data=data, coords=coords, n=30)
    assert r.name
