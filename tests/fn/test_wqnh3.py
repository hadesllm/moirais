"""Test wqnh3."""
import numpy as np
import pytest
from moirais.fn.wqnh3 import wqnh3


def test_wqnh3_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqnh3(data=data, coords=coords, n=20)
    assert r.value is not None


def test_wqnh3_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqnh3(data=data, coords=coords, n=20)
    assert r.name
