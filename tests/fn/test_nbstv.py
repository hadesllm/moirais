"""Test nbstv."""
import numpy as np
import pytest
from morie.fn.nbstv import nbstv


def test_nbstv_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbstv(data=data, coords=coords, n=20)
    assert r.value is not None


def test_nbstv_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbstv(data=data, coords=coords, n=20)
    assert r.name
