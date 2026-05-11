"""Test nbgrd."""
import numpy as np
import pytest
from morie.fn.nbgrd import nbgrd


def test_nbgrd_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbgrd(data=data, coords=coords, n=20)
    assert r.value is not None


def test_nbgrd_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbgrd(data=data, coords=coords, n=20)
    assert r.name
