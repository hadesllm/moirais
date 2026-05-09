"""Test lambrt."""
import numpy as np
import pytest
from moirais.fn.lambrt import lambrt


def test_lambrt_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = lambrt(coords=coords, n=20)
    assert r.value is not None


def test_lambrt_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = lambrt(coords=coords, n=20)
    assert r.name
