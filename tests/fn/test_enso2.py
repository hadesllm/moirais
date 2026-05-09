"""Test enso2."""
import numpy as np
import pytest
from moirais.fn.enso2 import enso2


def test_enso2_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enso2(data=data, coords=coords, n=30)
    assert r.value is not None


def test_enso2_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enso2(data=data, coords=coords, n=30)
    assert r.name
