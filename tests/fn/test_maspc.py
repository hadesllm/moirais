"""Test maspc."""
import numpy as np
import pytest
from morie.fn.maspc import maspc


def test_maspc_basic():
    rng = np.random.default_rng(42)
    depth = rng.uniform(0, 5000, 20)
    temp = rng.uniform(-2, 30, 20)
    sal = rng.uniform(30, 40, 20)
    r = maspc(depth=depth, temp=temp, salinity=sal, n=20)
    assert r.value is not None


def test_maspc_description():
    rng = np.random.default_rng(42)
    depth = rng.uniform(0, 5000, 20)
    temp = rng.uniform(-2, 30, 20)
    sal = rng.uniform(30, 40, 20)
    r = maspc(depth=depth, temp=temp, salinity=sal, n=20)
    assert r.name
