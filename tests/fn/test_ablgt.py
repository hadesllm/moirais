"""Test ablgt."""
import numpy as np
import pytest
from morie.fn.ablgt import ablgt


def test_ablgt_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = ablgt(data=data, coords=coords, n=20)
    assert r.value is not None


def test_ablgt_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = ablgt(data=data, coords=coords, n=20)
    assert r.name
