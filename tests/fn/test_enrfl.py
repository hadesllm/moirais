"""Test enrfl."""
import numpy as np
import pytest
from moirais.fn.enrfl import enrfl


def test_enrfl_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enrfl(data=data, coords=coords, n=30)
    assert r.value is not None


def test_enrfl_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enrfl(data=data, coords=coords, n=30)
    assert r.name
