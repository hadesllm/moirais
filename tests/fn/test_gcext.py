"""Test gcext."""
import numpy as np
import pytest
from moirais.fn.gcext import gcext


def test_gcext_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcext(data=data, coords=coords, n=30)
    assert r.value is not None


def test_gcext_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcext(data=data, coords=coords, n=30)
    assert r.name
