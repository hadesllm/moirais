"""Test envlc."""
import numpy as np
import pytest
from morie.fn.envlc import envlc


def test_envlc_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = envlc(data=data, coords=coords, n=30)
    assert r.value is not None


def test_envlc_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = envlc(data=data, coords=coords, n=30)
    assert r.name
