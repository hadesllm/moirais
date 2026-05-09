"""Test nbcog."""
import numpy as np
import pytest
from moirais.fn.nbcog import nbcog


def test_nbcog_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbcog(data=data, coords=coords, n=20)
    assert r.value is not None


def test_nbcog_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(30, 90, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = nbcog(data=data, coords=coords, n=20)
    assert r.name
