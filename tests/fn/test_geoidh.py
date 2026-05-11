"""Test geoidh."""
import numpy as np
import pytest
from morie.fn.geoidh import geoidh


def test_geoidh_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = geoidh(coords=coords, n=20)
    assert r.value is not None


def test_geoidh_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = geoidh(coords=coords, n=20)
    assert r.name
