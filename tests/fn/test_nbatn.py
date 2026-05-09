"""Test nbatn."""
import numpy as np
import pytest
from moirais.fn.nbatn import nbatn


def test_nbatn_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbatn(data=data, coords=coords, n=20)
    assert r.value is not None


def test_nbatn_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbatn(data=data, coords=coords, n=20)
    assert r.name
