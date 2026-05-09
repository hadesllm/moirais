"""Test machla."""
import numpy as np
import pytest
from moirais.fn.machla import machla


def test_machla_basic():
    rng = np.random.default_rng(42)
    depth = rng.uniform(0, 5000, 20)
    temp = rng.uniform(-2, 30, 20)
    sal = rng.uniform(30, 40, 20)
    r = machla(depth=depth, temp=temp, salinity=sal, n=20)
    assert r.value is not None


def test_machla_description():
    rng = np.random.default_rng(42)
    depth = rng.uniform(0, 5000, 20)
    temp = rng.uniform(-2, 30, 20)
    sal = rng.uniform(30, 40, 20)
    r = machla(depth=depth, temp=temp, salinity=sal, n=20)
    assert r.name
