"""Test abradn."""
import numpy as np
import pytest
from moirais.fn.abradn import abradn


def test_abradn_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abradn(data=data, coords=coords, n=20)
    assert r.value is not None


def test_abradn_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 100, (20, 2))
    r = abradn(data=data, coords=coords, n=20)
    assert r.name
