"""Test gcaer."""
import numpy as np
import pytest
from morie.fn.gcaer import gcaer


def test_gcaer_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcaer(data=data, coords=coords, n=30)
    assert r.value is not None


def test_gcaer_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcaer(data=data, coords=coords, n=30)
    assert r.name
