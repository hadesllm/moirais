"""Test nbzne."""
import numpy as np
import pytest
from morie.fn.nbzne import nbzne


def test_nbzne_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbzne(data=data, coords=coords, n=20)
    assert r.value is not None


def test_nbzne_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbzne(data=data, coords=coords, n=20)
    assert r.name
