"""Test wqeut."""
import numpy as np
import pytest
from moirais.fn.wqeut import wqeut


def test_wqeut_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqeut(data=data, coords=coords, n=20)
    assert r.value is not None


def test_wqeut_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 14, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wqeut(data=data, coords=coords, n=20)
    assert r.name
