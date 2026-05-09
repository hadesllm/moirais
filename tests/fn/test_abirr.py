"""Test abirr."""
import numpy as np
import pytest
from moirais.fn.abirr import abirr


def test_abirr_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abirr(data=data, coords=coords, n=20)
    assert r.value is not None


def test_abirr_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abirr(data=data, coords=coords, n=20)
    assert r.name
