"""Test abfng."""
import numpy as np
import pytest
from morie.fn.abfng import abfng


def test_abfng_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abfng(data=data, coords=coords, n=20)
    assert r.value is not None


def test_abfng_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abfng(data=data, coords=coords, n=20)
    assert r.name
