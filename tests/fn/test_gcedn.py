"""Test gcedn."""
import numpy as np
import pytest
from moirais.fn.gcedn import gcedn


def test_gcedn_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcedn(data=data, coords=coords, n=30)
    assert r.value is not None


def test_gcedn_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = gcedn(data=data, coords=coords, n=30)
    assert r.name
