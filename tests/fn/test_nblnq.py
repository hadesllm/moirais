"""Test nblnq."""
import numpy as np
import pytest
from morie.fn.nblnq import nblnq


def test_nblnq_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nblnq(data=data, coords=coords, n=20)
    assert r.value is not None


def test_nblnq_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nblnq(data=data, coords=coords, n=20)
    assert r.name
