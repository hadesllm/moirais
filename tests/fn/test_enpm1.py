"""Test enpm1."""
import numpy as np
import pytest
from moirais.fn.enpm1 import enpm1


def test_enpm1_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enpm1(data=data, coords=coords, n=30)
    assert r.value is not None


def test_enpm1_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enpm1(data=data, coords=coords, n=30)
    assert r.name
