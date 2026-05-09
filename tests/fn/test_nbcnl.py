"""Test nbcnl."""
import numpy as np
import pytest
from moirais.fn.nbcnl import nbcnl


def test_nbcnl_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbcnl(data=data, coords=coords, n=20)
    assert r.value is not None


def test_nbcnl_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbcnl(data=data, coords=coords, n=20)
    assert r.name
