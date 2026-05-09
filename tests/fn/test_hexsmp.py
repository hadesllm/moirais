"""Test hexsmp."""
import numpy as np
import pytest
from moirais.fn.hexsmp import hexsmp


def test_hexsmp_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = hexsmp(coords=coords, n=20)
    assert r.value is not None


def test_hexsmp_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = hexsmp(coords=coords, n=20)
    assert r.name
