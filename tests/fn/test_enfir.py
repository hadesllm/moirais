"""Test enfir."""
import numpy as np
import pytest
from morie.fn.enfir import enfir


def test_enfir_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enfir(data=data, coords=coords, n=30)
    assert r.value is not None


def test_enfir_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enfir(data=data, coords=coords, n=30)
    assert r.name
