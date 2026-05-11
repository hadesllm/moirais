"""Test wqsec."""
import numpy as np
import pytest
from morie.fn.wqsec import wqsec


def test_wqsec_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqsec(data=data, coords=coords, n=20)
    assert r.value is not None


def test_wqsec_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqsec(data=data, coords=coords, n=20)
    assert r.name
