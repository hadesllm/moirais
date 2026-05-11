"""Test nbleq."""
import numpy as np
import pytest
from morie.fn.nbleq import nbleq


def test_nbleq_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbleq(data=data, coords=coords, n=20)
    assert r.value is not None


def test_nbleq_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbleq(data=data, coords=coords, n=20)
    assert r.name
