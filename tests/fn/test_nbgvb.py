"""Test nbgvb."""
import numpy as np
import pytest
from moirais.fn.nbgvb import nbgvb


def test_nbgvb_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbgvb(data=data, coords=coords, n=20)
    assert r.value is not None


def test_nbgvb_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbgvb(data=data, coords=coords, n=20)
    assert r.name
