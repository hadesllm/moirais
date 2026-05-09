"""Test nbind."""
import numpy as np
import pytest
from moirais.fn.nbind import nbind


def test_nbind_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbind(data=data, coords=coords, n=20)
    assert r.value is not None


def test_nbind_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbind(data=data, coords=coords, n=20)
    assert r.name
