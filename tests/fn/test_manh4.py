"""Test manh4."""
import numpy as np
import pytest
from morie.fn.manh4 import manh4


def test_manh4_basic():
    rng = np.random.default_rng(42)
    depth = rng.uniform(0, 5000, 20)
    temp = rng.uniform(-2, 30, 20)
    sal = rng.uniform(30, 40, 20)
    r = manh4(depth=depth, temp=temp, salinity=sal, n=20)
    assert r.value is not None


def test_manh4_description():
    rng = np.random.default_rng(42)
    depth = rng.uniform(0, 5000, 20)
    temp = rng.uniform(-2, 30, 20)
    sal = rng.uniform(30, 40, 20)
    r = manh4(depth=depth, temp=temp, salinity=sal, n=20)
    assert r.name
