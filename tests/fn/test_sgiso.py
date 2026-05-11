"""Tests for isotropy test."""
import numpy as np
from morie.fn.sgiso import sgiso


def test_sgiso_smoke():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (100, 2))
    Z = rng.normal(0, 1, 100)
    r = sgiso(Z, coords)
    assert r.name == "isotropy_test"
    assert "isotropic" in r.extra
    assert "directions" in r.extra


def test_sgiso_isotropic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (200, 2))
    Z = rng.normal(0, 1, 200)
    r = sgiso(Z, coords)
    assert isinstance(r.value, float)
