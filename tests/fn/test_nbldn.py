"""Test nbldn."""
import numpy as np
import pytest
from morie.fn.nbldn import nbldn


def test_nbldn_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbldn(data=data, coords=coords, n=20)
    assert r.value is not None


def test_nbldn_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbldn(data=data, coords=coords, n=20)
    assert r.name
