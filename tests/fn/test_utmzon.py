"""Test utmzon."""
import numpy as np
import pytest
from morie.fn.utmzon import utmzon


def test_utmzon_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = utmzon(coords=coords, n=20)
    assert r.value is not None


def test_utmzon_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = utmzon(coords=coords, n=20)
    assert r.name
