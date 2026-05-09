"""Test enwtl."""
import numpy as np
import pytest
from moirais.fn.enwtl import enwtl


def test_enwtl_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enwtl(data=data, coords=coords, n=30)
    assert r.value is not None


def test_enwtl_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enwtl(data=data, coords=coords, n=30)
    assert r.name
