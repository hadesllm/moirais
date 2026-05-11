"""Test marov."""
import numpy as np
import pytest
from morie.fn.marov import marov


def test_marov_basic():
    rng = np.random.default_rng(42)
    depth = rng.uniform(0, 5000, 20)
    temp = rng.uniform(-2, 30, 20)
    sal = rng.uniform(30, 40, 20)
    r = marov(depth=depth, temp=temp, salinity=sal, n=20)
    assert r.value is not None


def test_marov_description():
    rng = np.random.default_rng(42)
    depth = rng.uniform(0, 5000, 20)
    temp = rng.uniform(-2, 30, 20)
    sal = rng.uniform(30, 40, 20)
    r = marov(depth=depth, temp=temp, salinity=sal, n=20)
    assert r.name
