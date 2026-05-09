"""Test tsspn."""
import numpy as np
import pytest
from moirais.fn.tsspn import tsspn


def test_tsspn_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tsspn(data=data, coords=coords, n=20, t=5)
    assert r.value is not None


def test_tsspn_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 5))
    coords = rng.uniform(0, 100, (20, 2))
    r = tsspn(data=data, coords=coords, n=20, t=5)
    assert r.name
