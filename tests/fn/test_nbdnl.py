"""Test nbdnl."""
import numpy as np
import pytest
from morie.fn.nbdnl import nbdnl


def test_nbdnl_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbdnl(data=data, coords=coords, n=20)
    assert r.value is not None


def test_nbdnl_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbdnl(data=data, coords=coords, n=20)
    assert r.name
