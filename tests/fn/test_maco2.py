"""Test maco2."""
import numpy as np
import pytest
from morie.fn.maco2 import maco2


def test_maco2_basic():
    rng = np.random.default_rng(42)
    depth = rng.uniform(0, 5000, 20)
    temp = rng.uniform(-2, 30, 20)
    sal = rng.uniform(30, 40, 20)
    r = maco2(depth=depth, temp=temp, salinity=sal, n=20)
    assert r.value is not None


def test_maco2_description():
    rng = np.random.default_rng(42)
    depth = rng.uniform(0, 5000, 20)
    temp = rng.uniform(-2, 30, 20)
    sal = rng.uniform(30, 40, 20)
    r = maco2(depth=depth, temp=temp, salinity=sal, n=20)
    assert r.name
