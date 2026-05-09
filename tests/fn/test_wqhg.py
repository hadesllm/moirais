"""Test wqhg."""
import numpy as np
import pytest
from moirais.fn.wqhg import wqhg


def test_wqhg_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqhg(data=data, coords=coords, n=20)
    assert r.value is not None


def test_wqhg_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqhg(data=data, coords=coords, n=20)
    assert r.name
