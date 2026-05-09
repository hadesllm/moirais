"""Test enfrg."""
import numpy as np
import pytest
from moirais.fn.enfrg import enfrg


def test_enfrg_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enfrg(data=data, coords=coords, n=30)
    assert r.value is not None


def test_enfrg_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enfrg(data=data, coords=coords, n=30)
    assert r.name
