"""Test maben."""
import numpy as np
import pytest
from moirais.fn.maben import maben


def test_maben_basic():
    rng = np.random.default_rng(42)
    depth = rng.uniform(0, 5000, 20)
    temp = rng.uniform(-2, 30, 20)
    sal = rng.uniform(30, 40, 20)
    r = maben(depth=depth, temp=temp, salinity=sal, n=20)
    assert r.value is not None


def test_maben_description():
    rng = np.random.default_rng(42)
    depth = rng.uniform(0, 5000, 20)
    temp = rng.uniform(-2, 30, 20)
    sal = rng.uniform(30, 40, 20)
    r = maben(depth=depth, temp=temp, salinity=sal, n=20)
    assert r.name
