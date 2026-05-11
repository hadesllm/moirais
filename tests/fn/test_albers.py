"""Test albers."""
import numpy as np
import pytest
from morie.fn.albers import albers


def test_albers_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = albers(coords=coords, n=20)
    assert r.value is not None


def test_albers_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = albers(coords=coords, n=20)
    assert r.name
