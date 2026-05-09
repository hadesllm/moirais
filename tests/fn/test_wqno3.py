"""Test wqno3."""
import numpy as np
import pytest
from moirais.fn.wqno3 import wqno3


def test_wqno3_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqno3(data=data, coords=coords, n=20)
    assert r.value is not None


def test_wqno3_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqno3(data=data, coords=coords, n=20)
    assert r.name
