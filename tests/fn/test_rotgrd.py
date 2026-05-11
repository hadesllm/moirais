"""Test rotgrd."""
import numpy as np
import pytest
from morie.fn.rotgrd import rotgrd


def test_rotgrd_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = rotgrd(coords=coords, n=20)
    assert r.value is not None


def test_rotgrd_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = rotgrd(coords=coords, n=20)
    assert r.name
