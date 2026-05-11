"""Test wqno2."""
import numpy as np
import pytest
from morie.fn.wqno2 import wqno2


def test_wqno2_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqno2(data=data, coords=coords, n=20)
    assert r.value is not None


def test_wqno2_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqno2(data=data, coords=coords, n=20)
    assert r.name
