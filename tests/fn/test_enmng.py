"""Test enmng."""
import numpy as np
import pytest
from moirais.fn.enmng import enmng


def test_enmng_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enmng(data=data, coords=coords, n=30)
    assert r.value is not None


def test_enmng_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enmng(data=data, coords=coords, n=30)
    assert r.name
