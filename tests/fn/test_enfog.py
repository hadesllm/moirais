"""Test enfog."""
import numpy as np
import pytest
from moirais.fn.enfog import enfog


def test_enfog_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enfog(data=data, coords=coords, n=30)
    assert r.value is not None


def test_enfog_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(30)
    coords = rng.uniform(0, 100, (30, 2))
    r = enfog(data=data, coords=coords, n=30)
    assert r.name
